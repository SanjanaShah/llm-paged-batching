import time
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Dict, List, Optional


@dataclass
class RequestMetrics:
    request_id: str
    prompt_len: int
    max_output_tokens: int
    arrival_time: float
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    tokens_generated: int = 0

    @property
    def ttft(self) -> Optional[float]:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def total_latency(self) -> Optional[float]:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time

    @property
    def generation_tps(self) -> Optional[float]:
        if self.finish_time is None or self.first_token_time is None:
            return None
        gen_time = self.finish_time - self.first_token_time
        return self.tokens_generated / gen_time if gen_time > 0 else float("inf")


class MetricsTracker:
    """
    Collects and aggregates per-request and system-level serving metrics.

    Mirrors the telemetry surface of production LLM serving systems:
    TTFT (time-to-first-token) captures prefill latency experienced by the user.
    ITL (inter-token latency) captures decode throughput.
    System TPS measures aggregate token generation capacity under load.
    Batch utilization tracks how efficiently the engine fills the batch window.
    """

    def __init__(self) -> None:
        self._requests: Dict[str, RequestMetrics] = {}
        self._system_start: float = time.monotonic()
        self._total_tokens: int = 0
        self._step_batch_sizes: List[int] = []
        self._step_latencies_s: List[float] = []
        self._preemption_count: int = 0
        self._swap_out_count: int = 0
        self._swap_in_count: int = 0

    def on_request_arrived(
        self, request_id: str, prompt_len: int, max_output_tokens: int, arrival_time: float
    ) -> None:
        self._requests[request_id] = RequestMetrics(
            request_id=request_id,
            prompt_len=prompt_len,
            max_output_tokens=max_output_tokens,
            arrival_time=arrival_time,
        )

    def on_first_token(self, request_id: str, timestamp: float) -> None:
        if request_id in self._requests:
            self._requests[request_id].first_token_time = timestamp

    def on_request_finished(self, request_id: str, tokens_generated: int, timestamp: float) -> None:
        if request_id in self._requests:
            m = self._requests[request_id]
            m.finish_time = timestamp
            m.tokens_generated = tokens_generated
        self._total_tokens += tokens_generated

    def on_step_complete(self, batch_size: int, step_duration_s: float) -> None:
        self._step_batch_sizes.append(batch_size)
        self._step_latencies_s.append(step_duration_s)

    def on_preemption(self) -> None:
        self._preemption_count += 1

    def on_swap_out(self) -> None:
        self._swap_out_count += 1

    def on_swap_in(self) -> None:
        self._swap_in_count += 1

    def summary(self) -> dict:
        finished = [r for r in self._requests.values() if r.finish_time is not None]
        elapsed = time.monotonic() - self._system_start

        ttfts = sorted([r.ttft for r in finished if r.ttft is not None])
        latencies = sorted([r.total_latency for r in finished if r.total_latency is not None])

        def pct(lst: list, p: float) -> Optional[float]:
            if not lst:
                return None
            idx = min(int(len(lst) * p), len(lst) - 1)
            return round(lst[idx] * 1000, 2)

        return {
            "elapsed_s": round(elapsed, 3),
            "total_requests_finished": len(finished),
            "total_tokens_generated": self._total_tokens,
            "system_tps": round(self._total_tokens / elapsed, 2) if elapsed > 0 else 0,
            "ttft_p50_ms": pct(ttfts, 0.50),
            "ttft_p90_ms": pct(ttfts, 0.90),
            "ttft_p99_ms": pct(ttfts, 0.99),
            "latency_mean_ms": round(mean(latencies) * 1000, 2) if latencies else None,
            "latency_p90_ms": pct(latencies, 0.90),
            "latency_p99_ms": pct(latencies, 0.99),
            "avg_batch_size": round(mean(self._step_batch_sizes), 2) if self._step_batch_sizes else 0,
            "total_steps": len(self._step_latencies_s),
            "preemptions": self._preemption_count,
            "swap_outs": self._swap_out_count,
            "swap_ins": self._swap_in_count,
        }

    def per_request_report(self) -> List[dict]:
        return [
            {
                "request_id": r.request_id,
                "prompt_len": r.prompt_len,
                "tokens_generated": r.tokens_generated,
                "ttft_ms": round(r.ttft * 1000, 2) if r.ttft else None,
                "total_latency_ms": round(r.total_latency * 1000, 2) if r.total_latency else None,
                "generation_tps": round(r.generation_tps, 2) if r.generation_tps else None,
            }
            for r in self._requests.values()
        ]
