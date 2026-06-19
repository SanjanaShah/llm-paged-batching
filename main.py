#!/usr/bin/env python3
"""
AetherServe Production Driver.

Runs 4 concurrent requests through the 3-tier engine with a live telemetry
dashboard showing block counts moving across GPU / CPU / NVMe storage in
real time.

Model selection (at startup):
  - If `transformers` and `torch` are installed, loads Qwen/Qwen2.5-0.5B-Instruct
    and generates real text.
  - Falls back to MockBackend automatically (no download needed), which
    produces synthetic token IDs useful for benchmarking tier mechanics.

Memory is intentionally constrained to force evictions across all three tiers
under the 4-request load. Watch the TIER row in the dashboard change.
"""

import asyncio
import logging
import sys
import time
from typing import Dict, Tuple

from src.backends import MockBackend, ModelBackend
from src.engine import AetherEngine
from src.models import SequenceRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("AetherServe")


# ── Token stream router ───────────────────────────────────────────────────────

class TokenStreamRouter:
    """
    Routes generated tokens to per-request async queues.

    Each client coroutine consumes from its own queue via consume_stream(),
    which yields (token_id, token_text) tuples as they arrive. A None
    sentinel signals end-of-stream.

    asyncio.Queue is safe here because everything runs on one event loop
    and call_soon_threadsafe serializes puts from any thread context.
    """

    def __init__(self) -> None:
        self._streams: Dict[str, asyncio.Queue] = {}

    def register(self, request_id: str) -> None:
        self._streams[request_id] = asyncio.Queue()

    def push_token(self, request_id: str, token_id: int, token_text: str) -> None:
        if request_id in self._streams:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(
                self._streams[request_id].put_nowait, (token_id, token_text)
            )

    async def consume_stream(self, request_id: str):
        queue = self._streams.get(request_id)
        if not queue:
            return
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            yield item
            queue.task_done()

    def close(self, request_id: str) -> None:
        if request_id in self._streams:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(self._streams[request_id].put_nowait, None)


# ── Per-client consumer ───────────────────────────────────────────────────────

async def client_consumer(request_id: str, router: TokenStreamRouter) -> None:
    """
    Simulates a downstream client (e.g. WebSocket connection) consuming
    the token stream for one request. Prints each token as it arrives and
    records TTFT on first token.
    """
    start = time.monotonic()
    first = True
    count = 0
    text_buf = []

    async for _, token_text in router.consume_stream(request_id):
        if first:
            ttft_ms = (time.monotonic() - start) * 1000
            logger.info(f"[{request_id}] TTFT {ttft_ms:.0f}ms")
            first = False
        text_buf.append(token_text)
        count += 1

    duration = time.monotonic() - start
    tps = count / max(duration, 0.001)
    output = "".join(text_buf)
    logger.info(f"[{request_id}] Done — {count} tokens, {tps:.1f} tok/s")
    if output.strip():
        print(f"\n  [{request_id}] → {output[:120]}")


# ── Live telemetry dashboard ──────────────────────────────────────────────────

async def telemetry_loop(engine: AetherEngine, interval_s: float = 0.05) -> None:
    """
    Prints a single updating line showing tier utilization every `interval_s`.
    Uses \\r to overwrite the same terminal line so it looks like a live gauge.

    TIER  GPU:used/total | CPU:used/total | NVMe:used/total
    QUEUE waiting | running | cpu-swapped | storage-swapped
    """
    while engine.scheduler.has_unfinished_requests():
        m = engine.memory_stats()
        s = engine.scheduler_stats()
        xfers = engine.metrics._tier_transfers
        xfer_str = "  ".join(f"{f}→{t}:{c}" for (f, t), c in xfers.items()) or "none"
        line = (
            f"\r\033[33m[TIER]\033[0m "
            f"GPU {m['gpu_used_blocks']:>2}/{m['gpu_total_blocks']} "
            f"| CPU {m['cpu_used_blocks']:>2}/{m['cpu_total_blocks']} "
            f"| NVMe {m['storage_used_blocks']:>2}/{m['storage_total_blocks']} "
            f"  \033[36m[Q]\033[0m "
            f"W:{s['waiting']} R:{s['running']} "
            f"SC:{s['swapped_cpu']} SS:{s['swapped_storage']}"
            f"  \033[35m[XFER]\033[0m {xfer_str}"
            "          "  # trailing spaces to clear previous longer lines
        )
        sys.stdout.write(line)
        sys.stdout.flush()
        await asyncio.sleep(interval_s)
    sys.stdout.write("\r" + " " * 120 + "\r")  # clear the line
    sys.stdout.flush()


# ── Model loader ──────────────────────────────────────────────────────────────

def load_backend() -> Tuple[ModelBackend, str]:
    try:
        from src.backends import TransformersBackend
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        logger.info(f"Loading {model_id} — this may take a moment on first run...")
        backend = TransformersBackend(model_id)
        return backend, model_id
    except (ImportError, Exception) as e:
        logger.info(f"TransformersBackend unavailable ({e}), using MockBackend")
        return MockBackend(), "MockBackend"


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    logger.info("=" * 60)
    logger.info("  AetherServe  |  3-Tier Hierarchical KV-Cache Engine")
    logger.info("=" * 60)

    backend, model_name = load_backend()
    logger.info(f"Backend: {model_name}")

    # Tight block budget forces all three tiers into play under 4-request load
    engine = AetherEngine(
        backend=backend,
        num_gpu_blocks=8,
        num_cpu_blocks=12,
        num_storage_blocks=64,
        max_batch_size=3,
    )

    router = TokenStreamRouter()
    engine.register_token_callback(router.push_token)

    prompts = [
        ("Req-Alpha",   "What is PagedAttention and why does it eliminate GPU memory fragmentation?", 40),
        ("Req-Bravo",   "Write a haiku about NVMe storage latency.", 20),
        ("Req-Charlie", "Explain continuous batching in three sentences.", 35),
        ("Req-Delta",   "What is the difference between prefill and decode phases in LLM inference?", 45),
    ]

    client_tasks = []
    for rid, text, max_out in prompts:
        router.register(rid)
        req = SequenceRequest(
            request_id=rid,
            prompt_tokens=[],
            prompt_text=text,
            max_output_tokens=max_out,
        )
        engine.add_request(req)
        client_tasks.append(asyncio.create_task(client_consumer(rid, router)))
        logger.info(f"Queued [{rid}] prompt_len={len(engine.backend.tokenize(text))} max_out={max_out}")

    logger.info("\nRunning engine + telemetry in parallel...")
    start = time.monotonic()

    engine_task = asyncio.create_task(engine.run_until_complete())

    await asyncio.gather(engine_task, telemetry_loop(engine))

    # Close any streams that didn't get a FINISHED signal in the step loop
    for rid, _, _ in prompts:
        router.close(rid)

    await asyncio.gather(*client_tasks)

    elapsed = time.monotonic() - start
    summary = engine.metrics.summary()

    print("\n" + "=" * 60)
    print("  METRICS SUMMARY")
    print("=" * 60)
    print(f"  Wall time          : {elapsed:.3f}s")
    print(f"  Requests finished  : {summary['total_requests_finished']}")
    print(f"  Tokens generated   : {summary['total_tokens_generated']}")
    print(f"  System throughput  : {summary['system_tps']} tok/s")
    print(f"  TTFT p50 / p90     : {summary['ttft_p50_ms']}ms / {summary['ttft_p90_ms']}ms")
    print(f"  Latency mean / p99 : {summary['latency_mean_ms']}ms / {summary['latency_p99_ms']}ms")
    print(f"  Avg batch size     : {summary['avg_batch_size']}")
    print(f"  Steps              : {summary['total_steps']}")
    print(f"  Preemptions        : {summary['preemptions']}")
    print(f"  Tier transfers     : {summary['tier_transfers']}")
    print(f"  Prefetches (pred.) : {engine.prefetcher.total_prefetches}")

    mem = engine.memory_stats()
    if mem["storage_quantized"]:
        saved_kb = mem["storage_bytes_saved"] / 1024
        ratio = mem["storage_compression_ratio"]
        print(f"  NVMe compression   : {ratio:.1f}× INT8 | {saved_kb:.1f} KB saved")
    print("=" * 60)

    engine.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        sys.exit(130)
