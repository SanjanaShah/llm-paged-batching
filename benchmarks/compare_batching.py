"""
AetherServe Benchmark: Continuous Batching vs Naive Static Batching

Static batching waits for the entire batch to finish before starting the next.
Because decode cost is proportional to the *longest* sequence in the batch,
shorter sequences sit idle burning wall-clock time.

Continuous batching evicts finished sequences immediately and injects new ones
at the start of the very next iteration — no idle time for short sequences.

This benchmark quantifies the throughput gap across a realistic workload with
high output-length variance (the regime where the difference is largest).
"""
import asyncio
import random
import sys
import time
from pathlib import Path

# Allow running from project root or benchmarks/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine import AetherEngine, _DECODE_STEP_LATENCY_S, _PREFILL_COST_PER_TOKEN_S
from src.models import SequenceRequest


def _make_workload(n: int, seed: int = 42) -> list[SequenceRequest]:
    rng = random.Random(seed)
    return [
        SequenceRequest(
            request_id=f"req-{i:03d}",
            prompt_tokens=list(range(rng.randint(8, 64))),
            max_output_tokens=rng.randint(4, 128),
        )
        for i in range(n)
    ]


async def _run_continuous(requests: list[SequenceRequest]) -> dict:
    engine = AetherEngine(num_gpu_blocks=512, num_cpu_blocks=1024, max_batch_size=8)
    for r in requests:
        engine.add_request(r)
    await engine.run_until_complete()
    return engine.metrics.summary()


async def _run_naive_static(requests: list[SequenceRequest], batch_size: int = 8) -> dict:
    """
    Simulates static batching: collect a fixed batch, wait for all to finish
    before starting the next. Decode cost = max_output_tokens of longest sequence.
    This wastes cycles for every sequence shorter than the max.
    """
    start = time.monotonic()
    total_tokens = 0

    for i in range(0, len(requests), batch_size):
        batch = requests[i : i + batch_size]
        max_output = max(r.max_output_tokens for r in batch)
        total_prompt = sum(len(r.prompt_tokens) for r in batch)

        # Prefill: compute-bound, proportional to prompt tokens
        await asyncio.sleep(_PREFILL_COST_PER_TOKEN_S * total_prompt)

        # Decode: ALL sequences blocked on the *longest* sequence
        # (This is the fundamental inefficiency of static batching)
        await asyncio.sleep(_DECODE_STEP_LATENCY_S * max_output)

        for r in batch:
            total_tokens += r.max_output_tokens

    elapsed = time.monotonic() - start
    return {
        "elapsed_s": round(elapsed, 3),
        "total_tokens_generated": total_tokens,
        "system_tps": round(total_tokens / elapsed, 2),
        "avg_batch_size": batch_size,
    }


def _print_comparison(n: int, cb: dict, sb: dict) -> None:
    speedup = sb["elapsed_s"] / cb["elapsed_s"]
    tps_gain = (cb["system_tps"] - sb["system_tps"]) / sb["system_tps"] * 100

    print(f"\n{'=' * 68}")
    print(f"  AetherServe — Continuous vs Static Batching Benchmark")
    print(f"  Workload: {n} requests, output length 4–128 tokens (high variance)")
    print(f"{'=' * 68}")
    print(f"{'Metric':<42} {'Continuous':>12} {'Static':>12}")
    print(f"{'-' * 68}")
    print(f"{'Total wall time (s)':<42} {cb['elapsed_s']:>12.3f} {sb['elapsed_s']:>12.3f}")
    print(f"{'System throughput (tok/s)':<42} {cb['system_tps']:>12.2f} {sb['system_tps']:>12.2f}")
    print(f"{'Total tokens generated':<42} {cb['total_tokens_generated']:>12,} {sb['total_tokens_generated']:>12,}")
    if cb.get("ttft_p50_ms"):
        print(f"{'P50 TTFT (ms)':<42} {cb['ttft_p50_ms']:>12.2f} {'N/A':>12}")
    if cb.get("ttft_p99_ms"):
        print(f"{'P99 TTFT (ms)':<42} {cb['ttft_p99_ms']:>12.2f} {'N/A':>12}")
    if cb.get("avg_batch_size"):
        print(f"{'Avg batch utilization':<42} {cb['avg_batch_size']:>12.2f} {sb['avg_batch_size']:>12}")
    print(f"{'Preemptions':<42} {cb.get('preemptions', 0):>12} {'—':>12}")
    print(f"{'-' * 68}")
    print(f"  Speedup: {speedup:.2f}x  |  Throughput gain: +{tps_gain:.1f}%")
    print(f"{'=' * 68}\n")


async def main() -> None:
    n = 32
    print(f"[1/3] Generating workload ({n} requests)...")
    workload = _make_workload(n)

    def _clone(reqs):
        return [
            SequenceRequest(
                request_id=r.request_id,
                prompt_tokens=list(r.prompt_tokens),
                max_output_tokens=r.max_output_tokens,
            )
            for r in reqs
        ]

    print("[2/3] Running continuous batching (AetherServe)...")
    cb_results = await _run_continuous(_clone(workload))

    print("[3/3] Running naive static batching (batch_size=8)...")
    sb_results = await _run_naive_static(_clone(workload), batch_size=8)

    _print_comparison(n, cb_results, sb_results)


if __name__ == "__main__":
    asyncio.run(main())
