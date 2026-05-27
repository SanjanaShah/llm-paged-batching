#!/usr/bin/env python3
"""
AetherServe Enterprise Production Driver.

This module sets up an asynchronous execution environment for continuous batching 
and virtual KV-cache scheduling, simulating high-throughput LLM serving infrastructure.
"""

import asyncio
import logging
import sys
import time
from typing import Dict, List

from src.engine import AetherEngine
from src.models import SequenceRequest, SequenceStatus

# --- PRODUCTION LOGGING ARCHITECTURE ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("AetherServe")


# --- REAL-TIME ASYNC STREAM HANDLER ---
class TokenStreamRouter:
    """Manages thread-safe asynchronous token delivery streams for concurrent users."""
    def __init__(self) -> None:
        self._streams: Dict[str, asyncio.Queue] = {}

    def register_request(self, request_id: str) -> None:
        """Initialize an isolated data stream queue for a unique request ID."""
        self._streams[request_id] = asyncio.Queue()

    def push_token(self, request_id: str, token_id: int) -> None:
        """Callback engine hook to intercept generated tokens and route them to queues."""
        if request_id in self._streams:
            # Safe, thread-friendly background push
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(self._streams[request_id].put_nowait, token_id)

    async def consume_stream(self, request_id: str):
        """Asynchronous generator yielding tokens sequentially as they are produced."""
        queue = self._streams.get(request_id)
        if not queue:
            return

        while True:
            token = await queue.get()
            if token is None:  # Sentinel value signaling end-of-stream
                queue.task_done()
                break
            yield token
            queue.task_done()

    def close_stream(self, request_id: str) -> None:
        """Inject an explicit termination signal into the designated user queue."""
        if request_id in self._streams:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(self._streams[request_id].put_nowait, None)


# --- CONCURRENT CLIENT SIMULATORS ---
async def simulate_user_client(request_id: str, router: TokenStreamRouter):
    """Simulates an external client consuming a live, real-time token stream from the server."""
    logger.info(f"📡 Client [{request_id}] initialized streaming connection channel.")
    
    tokens_received = 0
    start_time = time.monotonic()
    ttft_recorded = False

    async for token in router.consume_stream(request_id):
        if not ttft_recorded:
            ttft_ms = (time.monotonic() - start_time) * 1000
            logger.info(f"✨ Client [{request_id}] -> Time-To-First-Token (TTFT): {ttft_ms:.1f}ms")
            ttft_recorded = True
            
        tokens_received += 1
        # In a real microservice, this streams raw data chunks to WebSockets / HTTP
        print(f"\033[34m[STREAM-{request_id}]\033[0m Generated Token ID: {token}")

    duration = time.monotonic() - start_time
    tps = tokens_received / max(0.001, duration)
    logger.info(f"🛑 Client [{request_id}] Stream Finished. Tokens: {tokens_received}, Speed: {tps:.1f} TPS")


# --- MAIN ORCHESTRATION EVENT LOOP ---
async def main():
    logger.info("=== Starting AetherServe High-Performance Inference Engine ===")

    # 1. Initialize core system dependencies using bounded memory pressures
    # We restrict blocks to show preemption handling under load
    engine = AetherEngine(
        num_gpu_blocks=16,      # Forces memory saturation
        num_cpu_blocks=32,      # Limits backup swap space allocation
        max_batch_size=4        # Hard ceiling on batch matrix step width
    )
    router = TokenStreamRouter()

    # Bind the router to our engine callback system
    engine.register_token_callback(router.push_token)

    # 2. Design a heterogeneous test batch derived straight from your test conditions
    now = time.monotonic()
    raw_prompts = [
        ("User-Alpha",   20, 15),  # Medium prompt, mid output
        ("User-Bravo",   6,  10),  # Light prefill, low footprint
        ("User-Charlie", 64, 30),  # Heavy prompt - deliberately triggers preemption or swap pressure
        ("User-Delta",   2,  8),   # Fast execution turnaround
    ]

    requests: List[SequenceRequest] = []
    client_tasks = []

    # 3. Provision queues and register monitoring handles
    for rid, prompt_len, max_out in raw_prompts:
        req = SequenceRequest(
            request_id=rid,
            prompt_tokens=list(range(prompt_len)),
            max_output_tokens=max_out,
            arrival_time=now
        )
        requests.append(req)
        
        # Open routing queues before starting processing
        router.register_request(rid)
        
        # Launch independent client consumers on the async event loop
        client_tasks.append(asyncio.create_task(simulate_user_client(rid, router)))

    # 4. Inject inputs into the scheduling core
    for req in requests:
        engine.add_request(req)
        logger.info(f"📥 Queued Request [{req.request_id}] | Prompt: {len(req.prompt_tokens)} tokens")

    # 5. Execute the execution loop concurrently alongside our streaming clients
    logger.info("🎬 Launching continuous batching loop processing steps...")
    engine_start_time = time.monotonic()

    # Wrap request completion checks with tracking loops
    while engine.scheduler.has_unfinished_requests():
        # Step through scheduling, hardware simulation delay, and sampling loops
        outputs = await engine.step()
        
        if outputs:
            # Close queues for requests that finished during this specific step
            for req in outputs.scheduled:
                if req.status == SequenceStatus.FINISHED:
                    router.close_stream(req.request_id)

    # Reclaim dangling stream references
    for req in requests:
        router.close_stream(req.request_id)

    # Wait for consumer clients to flush their final tokens
    await asyncio.gather(*client_tasks)

    # 6. Collate System Metrics Output
    engine_duration = time.monotonic() - engine_start_time
    summary_stats = engine.metrics.summary()

    print("\n" + "="*50)
    print("       AETHERSERVE LIVE METRICS SUMMARY REPORT     ")
    print("="*50)
    print(f" Total Wall-Clock Operational Execution Time : {engine_duration:.3f} seconds")
    print(f" Total Completed Client Processing Requests  : {summary_stats.get('total_requests_finished', 0)}")
    print(f" Total System Text Tokens Fully Generated    : {summary_stats.get('total_tokens_generated', 0)}")
    print(f" Aggregated Engine Processing Performance    : {summary_stats.get('system_tps', 0.0):.2f} tokens/sec")
    print(f" Median Expected Response Latency (TTFT P50) : {summary_stats.get('ttft_p50_ms', 0.0):.2f} ms")
    print("="*50 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Engine shutdown initialized manually by operator interrupt.")
        sys.exit(130)