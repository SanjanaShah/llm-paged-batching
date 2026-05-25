import asyncio
import random
import time
from typing import Callable, List, Optional

from src.cache_manager import CacheManager
from src.metrics import MetricsTracker
from src.models import SequenceRequest, SequenceStatus
from src.scheduler import AetherserveScheduler, SchedulerOutputs

# Simulated forward-pass latency constants.
# Prefill is compute-bound: cost grows with number of input tokens (quadratic attention).
# Decode is memory-bandwidth-bound: cost is nearly flat per step regardless of batch size,
# because we're streaming KV-cache reads from HBM rather than recomputing attention.
_PREFILL_COST_PER_TOKEN_S = 0.003   # 3 ms / prompt token
_DECODE_STEP_LATENCY_S = 0.020      # 20 ms / decode iteration (flat, bandwidth-bound)
_DECODE_JITTER_S = 0.002            # ±2 ms stochastic kernel overhead


class AetherEngine:
    """
    Asynchronous LLM inference engine with continuous batching and virtual KV-cache.

    Design mirrors vLLM's three-layer architecture:
      CacheManager  — physical block allocation (PagedAttention abstraction)
      Scheduler     — iteration-level batch composition with preemption
      Engine loop   — drives simulated forward passes with realistic latency profiles

    The engine is intentionally model-agnostic: token generation is pluggable via
    _generate_token(). Replace the mock with a real HuggingFace / Triton backend
    without touching scheduler or cache logic.
    """

    def __init__(
        self,
        num_gpu_blocks: int = 128,
        num_cpu_blocks: int = 256,
        max_batch_size: int = 8,
    ) -> None:
        self.cache_manager = CacheManager(
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
        )
        self.scheduler = AetherserveScheduler(
            cache_manager=self.cache_manager,
            max_batch_size=max_batch_size,
        )
        self.metrics = MetricsTracker()
        self._step_count: int = 0
        self._token_callbacks: List[Callable[[str, int], None]] = []

    def add_request(self, request: SequenceRequest) -> None:
        self.metrics.on_request_arrived(
            request_id=request.request_id,
            prompt_len=len(request.prompt_tokens),
            max_output_tokens=request.max_output_tokens,
            arrival_time=request.arrival_time,
        )
        self.scheduler.add_request(request)

    def register_token_callback(self, cb: Callable[[str, int], None]) -> None:
        """Register a function called on every generated token (request_id, token_id)."""
        self._token_callbacks.append(cb)

    async def step(self) -> Optional[SchedulerOutputs]:
        """
        Execute one iteration of the continuous batching loop.

        A single step produces exactly one new token per decode-phase sequence
        and processes the full prefill for newly-admitted sequences. This is the
        core property that makes continuous batching efficient: decode steps have
        flat cost, so batching more sequences together is nearly free.
        """
        outputs = self.scheduler.schedule()

        for req in outputs.swapped_out:
            self.metrics.on_swap_out()
        for req in outputs.swapped_in:
            self.metrics.on_swap_in()

        if outputs.is_empty:
            if self.scheduler.has_unfinished_requests():
                await asyncio.sleep(0.001)
            return outputs

        self._step_count += 1
        step_start = time.monotonic()

        # Separate newly-admitted (prefill) from continuing (decode) sequences.
        # Prefill sequences have no generated tokens yet.
        prefill_seqs = [r for r in outputs.scheduled if not r.generated_tokens]
        decode_seqs = [r for r in outputs.scheduled if r.generated_tokens]

        # Simulate prefill: compute-bound, cost proportional to prompt length.
        if prefill_seqs:
            total_prompt_tokens = sum(len(r.prompt_tokens) for r in prefill_seqs)
            await asyncio.sleep(_PREFILL_COST_PER_TOKEN_S * total_prompt_tokens)

        # Simulate decode: memory-bandwidth-bound, flat cost per step.
        if decode_seqs:
            jitter = random.uniform(-_DECODE_JITTER_S, _DECODE_JITTER_S)
            await asyncio.sleep(max(0.001, _DECODE_STEP_LATENCY_S + jitter))

        now = time.monotonic()

        for req in outputs.scheduled:
            if req.status == SequenceStatus.FINISHED:
                continue

            token = self._generate_token(req)
            req.generated_tokens.append(token)

            if len(req.generated_tokens) == 1:
                req.first_token_time = now
                self.metrics.on_first_token(req.request_id, now)

            for cb in self._token_callbacks:
                cb(req.request_id, token)

            if req.is_finished:
                req.status = SequenceStatus.FINISHED
                req.finish_time = now
                self.metrics.on_request_finished(req.request_id, len(req.generated_tokens), now)
                self.cache_manager.free(req.request_id)

        for req in outputs.preempted:
            if req.status != SequenceStatus.FINISHED:
                self.metrics.on_preemption()

        step_duration = time.monotonic() - step_start
        self.metrics.on_step_complete(len(outputs.scheduled), step_duration)
        return outputs

    def _generate_token(self, req: SequenceRequest) -> int:
        """
        Deterministic mock token generation. In a real engine, this is replaced
        by model.forward() + argmax/sampling over the vocabulary logit distribution.
        The determinism allows test reproducibility without a real model.
        """
        return (hash(req.request_id) + len(req.generated_tokens)) % 50256

    async def run_until_complete(self) -> None:
        """Drive the engine loop until all queued requests finish."""
        while self.scheduler.has_unfinished_requests():
            result = await self.step()
            if result is None:
                break

    def memory_stats(self) -> dict:
        return self.cache_manager.memory_stats()

    def scheduler_stats(self) -> dict:
        return self.scheduler.stats()
