"""
AetherServe core inference engine.

Wires together: ModelBackend → token generation
               HierarchicalCacheManager → block allocation across 3 tiers
               HierarchicalAdaptiveScheduler → continuous batching + cascading eviction
               PredictivePrefetcher → proactive Tier 3 → Tier 2 prefetch

The engine is model-agnostic: swap MockBackend for TransformersBackend to run
real HuggingFace weights with zero changes to the scheduler or cache layer.

Latency simulation constants model the compute/memory-bandwidth distinction:
  Prefill is compute-bound → cost grows with total prompt tokens
  Decode is memory-bandwidth-bound → cost is nearly flat per step (we're
    streaming KV-cache reads from HBM, not recomputing attention)

When using TransformersBackend these constants are not used — the actual
model.forward() wall time provides the real latency signal.
"""

import asyncio
import random
import time
from typing import Callable, List, Optional

from src.backends import MockBackend, ModelBackend
from src.cache_manager import HierarchicalCacheManager
from src.metrics import MetricsTracker
from src.models import SequenceRequest, SequenceStatus
from src.prefetcher import PredictivePrefetcher
from src.scheduler import HierarchicalAdaptiveScheduler, SchedulerOutputs

# Callback signature: (request_id, token_id, token_text)
TokenCallback = Callable[[str, int, str], None]

_PREFILL_COST_PER_TOKEN_S = 0.003   # 3 ms / prompt token (mock only)
_DECODE_STEP_LATENCY_S = 0.020      # 20 ms / decode iteration (mock only)
_DECODE_JITTER_S = 0.002            # ±2 ms stochastic kernel overhead


class AetherEngine:
    """
    Asynchronous LLM inference engine with continuous batching,
    three-tier hierarchical KV-cache, and predictive prefetching.
    """

    def __init__(
        self,
        backend: Optional[ModelBackend] = None,
        num_gpu_blocks: int = 128,
        num_cpu_blocks: int = 256,
        num_storage_blocks: int = 0,
        max_batch_size: int = 8,
        quantize_storage: bool = True,
    ) -> None:
        self.backend: ModelBackend = backend or MockBackend()
        self.cache_manager = HierarchicalCacheManager(
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            num_storage_blocks=num_storage_blocks,
            quantize_storage=quantize_storage,
        )
        self.scheduler = HierarchicalAdaptiveScheduler(
            cache_manager=self.cache_manager,
            max_batch_size=max_batch_size,
        )
        self.prefetcher = PredictivePrefetcher(
            cache_manager=self.cache_manager,
            scheduler=self.scheduler,
        )
        self.metrics = MetricsTracker()
        self._step_count: int = 0
        self._token_callbacks: List[TokenCallback] = []

    def add_request(self, request: SequenceRequest) -> None:
        # If raw text was provided without tokens, tokenize via the backend
        if request.prompt_text and not request.prompt_tokens:
            request.prompt_tokens = self.backend.tokenize(request.prompt_text)
        self.metrics.on_request_arrived(
            request_id=request.request_id,
            prompt_len=len(request.prompt_tokens),
            max_output_tokens=request.max_output_tokens,
            arrival_time=request.arrival_time,
        )
        self.scheduler.add_request(request)

    def register_token_callback(self, cb: TokenCallback) -> None:
        self._token_callbacks.append(cb)

    async def step(self) -> Optional[SchedulerOutputs]:
        """
        One iteration of the continuous batching loop.

        Order of operations:
          1. Prefetcher tick — move Tier 3 → Tier 2 for upcoming requests
          2. Scheduler — determine this step's batch composition
          3. Simulate prefill latency (compute-bound) for new sequences
          4. Simulate decode latency (memory-bandwidth-bound) for ongoing ones
          5. Generate one token per scheduled sequence
          6. Record metrics, free finished sequences
        """
        # Proactive prefetch before scheduling so prefetched blocks are
        # already in CPU when the scheduler tries to elevate them to GPU
        self.prefetcher.tick()

        outputs = self.scheduler.schedule()

        # Track tier transfer events
        for req in outputs.swapped_cpu_out:
            self.metrics.on_tier_transfer("gpu", "cpu")
        for req in outputs.swapped_cpu_in:
            self.metrics.on_tier_transfer("cpu", "gpu")
        for req in outputs.swapped_storage_out:
            self.metrics.on_tier_transfer("cpu", "storage")
        for req in outputs.swapped_storage_in:
            self.metrics.on_tier_transfer("storage", "cpu")

        if outputs.is_empty:
            if self.scheduler.has_unfinished_requests():
                await asyncio.sleep(0.001)
            return outputs

        self._step_count += 1
        step_start = time.monotonic()

        using_mock = isinstance(self.backend, MockBackend)
        prefill_seqs = [r for r in outputs.scheduled if not r.generated_tokens]
        decode_seqs = [r for r in outputs.scheduled if r.generated_tokens]

        if using_mock:
            # Simulate latency with asyncio.sleep
            if prefill_seqs:
                total_prompt = sum(len(r.prompt_tokens) for r in prefill_seqs)
                await asyncio.sleep(_PREFILL_COST_PER_TOKEN_S * total_prompt)
            if decode_seqs:
                jitter = random.uniform(-_DECODE_JITTER_S, _DECODE_JITTER_S)
                await asyncio.sleep(max(0.001, _DECODE_STEP_LATENCY_S + jitter))

        now = time.monotonic()

        for req in outputs.scheduled:
            if req.status == SequenceStatus.FINISHED:
                continue

            token_id, token_text = self.backend.generate_next_token(
                req.prompt_tokens, req.generated_tokens
            )
            req.generated_tokens.append(token_id)

            if len(req.generated_tokens) == 1:
                req.first_token_time = now
                self.metrics.on_first_token(req.request_id, now)

            for cb in self._token_callbacks:
                cb(req.request_id, token_id, token_text)

            # EOS check: if backend exposes eos_token_id, stop early
            eos = getattr(self.backend, "eos_token_id", None)
            if req.is_finished or (eos is not None and token_id == eos):
                req.status = SequenceStatus.FINISHED
                req.finish_time = now
                self.metrics.on_request_finished(
                    req.request_id, len(req.generated_tokens), now
                )
                self.cache_manager.free(req.request_id)

        for req in outputs.preempted:
            if req.status not in (SequenceStatus.FINISHED, SequenceStatus.WAITING):
                self.metrics.on_preemption()

        step_duration = time.monotonic() - step_start
        self.metrics.on_step_complete(len(outputs.scheduled), step_duration)
        return outputs

    async def run_until_complete(self) -> None:
        while self.scheduler.has_unfinished_requests():
            result = await self.step()
            if result is None:
                break

    def memory_stats(self) -> dict:
        return self.cache_manager.memory_stats()

    def scheduler_stats(self) -> dict:
        return self.scheduler.stats()

    def cleanup(self) -> None:
        self.cache_manager.cleanup()
