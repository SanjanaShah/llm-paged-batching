"""
Smoke test for the full engine loop.
Validates that all requests complete and metrics are populated.
"""
import asyncio
import time
import pytest
from src.models import SequenceRequest, SequenceStatus
from src.engine import AetherEngine


def _req(rid: str, prompt_len: int, max_output: int) -> SequenceRequest:
    return SequenceRequest(
        request_id=rid,
        prompt_tokens=list(range(prompt_len)),
        max_output_tokens=max_output,
    )


@pytest.mark.asyncio
async def test_all_requests_complete():
    engine = AetherEngine(num_gpu_blocks=64, num_cpu_blocks=128, max_batch_size=4)
    reqs = [
        _req("user-a", prompt_len=4, max_output=5),
        _req("user-b", prompt_len=2, max_output=2),
        _req("user-c", prompt_len=1, max_output=8),
    ]
    for r in reqs:
        engine.add_request(r)

    await engine.run_until_complete()

    for r in reqs:
        assert r.status == SequenceStatus.FINISHED
        assert len(r.generated_tokens) == r.max_output_tokens


@pytest.mark.asyncio
async def test_metrics_populated():
    engine = AetherEngine(num_gpu_blocks=64, num_cpu_blocks=128, max_batch_size=4)
    reqs = [_req(f"r{i}", 8, 4) for i in range(4)]
    for r in reqs:
        engine.add_request(r)

    await engine.run_until_complete()

    summary = engine.metrics.summary()
    assert summary["total_requests_finished"] == 4
    assert summary["total_tokens_generated"] == 16
    assert summary["system_tps"] > 0
    assert summary["ttft_p50_ms"] is not None


@pytest.mark.asyncio
async def test_variable_length_requests():
    """Continuous batching should handle heterogeneous output lengths correctly."""
    engine = AetherEngine(num_gpu_blocks=128, num_cpu_blocks=256, max_batch_size=8)
    lengths = [1, 5, 2, 20, 3, 15, 7, 30]
    reqs = [_req(f"r{i}", prompt_len=4, max_output=l) for i, l in enumerate(lengths)]
    for r in reqs:
        engine.add_request(r)

    await engine.run_until_complete()

    for r, expected in zip(reqs, lengths):
        assert len(r.generated_tokens) == expected


@pytest.mark.asyncio
async def test_token_callback_fires():
    engine = AetherEngine(num_gpu_blocks=64, num_cpu_blocks=128, max_batch_size=4)
    seen_tokens: dict = {}

    def on_token(rid: str, tok: int, text: str) -> None:
        seen_tokens.setdefault(rid, []).append(tok)

    engine.register_token_callback(on_token)
    reqs = [_req("cb-test", 4, 5)]
    for r in reqs:
        engine.add_request(r)

    await engine.run_until_complete()
    assert len(seen_tokens["cb-test"]) == 5


# Standalone runner (kept for backward compat with original test style)
async def _standalone():
    print("[Test] Initializing AetherServe Engine...")
    engine = AetherEngine()
    reqs = [
        _req("User-A", 4, 5),
        _req("User-B", 2, 2),
        _req("User-C", 1, 8),
    ]
    for r in reqs:
        engine.add_request(r)
    start = time.time()
    await engine.run_until_complete()
    elapsed = time.time() - start
    print(f"\n[Success] All requests processed in {elapsed:.2f}s")
    import json
    print(json.dumps(engine.metrics.summary(), indent=2))


if __name__ == "__main__":
    asyncio.run(_standalone())
