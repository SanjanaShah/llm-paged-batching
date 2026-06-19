"""
Tests for the cascading eviction logic in HierarchicalAdaptiveScheduler.

Validates that GPU OOM triggers the correct tier cascade:
  GPU full → swap oldest running req to CPU
  CPU also full → push coldest CPU-swapped req to Storage, then retry GPU→CPU
  All full → recompute (wipe + re-queue as WAITING)

Also validates the elevator restore path and predictive prefetcher.
"""
import time
import pytest
from src.models import SequenceRequest, SequenceStatus, BLOCK_SIZE
from src.cache_manager import HierarchicalCacheManager
from src.scheduler import HierarchicalAdaptiveScheduler
from src.prefetcher import PredictivePrefetcher


def _req(rid: str, prompt_len: int = BLOCK_SIZE, max_output: int = 8) -> SequenceRequest:
    return SequenceRequest(
        request_id=rid,
        prompt_tokens=list(range(prompt_len)),
        max_output_tokens=max_output,
    )


@pytest.fixture
def setup():
    """Tight memory config that forces tier evictions under moderate load."""
    cache = HierarchicalCacheManager(
        num_gpu_blocks=4,
        num_cpu_blocks=4,
        num_storage_blocks=16,
    )
    sched = HierarchicalAdaptiveScheduler(cache_manager=cache, max_batch_size=8)
    yield cache, sched
    cache.cleanup()


# ── Basic admission ───────────────────────────────────────────────────────────

def test_admits_within_capacity(setup):
    cache, sched = setup
    reqs = [_req(f"r{i}") for i in range(4)]
    for r in reqs:
        sched.add_request(r)
    out = sched.schedule()
    assert len(out.scheduled) == 4
    assert all(r.status == SequenceStatus.RUNNING for r in reqs)


# ── GPU OOM → CPU swap ────────────────────────────────────────────────────────

def test_gpu_oom_triggers_cpu_swap(setup):
    cache, sched = setup
    # Fill GPU: 4 blocks, each request uses 1 block
    reqs = [_req(f"r{i}", prompt_len=BLOCK_SIZE, max_output=BLOCK_SIZE + 4) for i in range(4)]
    for r in reqs:
        sched.add_request(r)
    sched.schedule()  # All 4 admitted, 4 GPU blocks used

    # Advance each request past its first block boundary → needs 2nd block
    for r in reqs:
        for _ in range(BLOCK_SIZE):
            r.generated_tokens.append(42)

    # Next schedule: 4 running requests each need a new GPU block, but 0 free
    out = sched.schedule()

    # At least some must have been swapped to CPU
    swapped = [r for r in reqs if r.status == SequenceStatus.SWAPPED_CPU]
    assert len(swapped) >= 1, "GPU OOM should trigger CPU swap"


# ── CPU OOM → cascade to Storage ─────────────────────────────────────────────

def test_cascade_to_storage_when_cpu_full():
    """
    With GPU and CPU both full, a running request's OOM must cascade the
    coldest CPU-swapped request down to Tier 3 storage.

    Setup (bypasses the normal allocation path to control tier state precisely):
      GPU: r1, r2 running (each 1 block → GPU fully saturated)
      CPU: r3, r4 injected directly via the CPU allocator (fills CPU)
    Then push r1 past its first block boundary so append_slot needs a new
    GPU block. GPU full → cascade: coldest CPU req goes to storage.
    """
    cache = HierarchicalCacheManager(
        num_gpu_blocks=2,
        num_cpu_blocks=2,
        num_storage_blocks=16,
    )
    sched = HierarchicalAdaptiveScheduler(cache_manager=cache, max_batch_size=8)

    # r1, r2 running on GPU (consumes all 2 GPU blocks)
    r1 = _req("r1", prompt_len=BLOCK_SIZE, max_output=BLOCK_SIZE + 4)
    r2 = _req("r2", prompt_len=BLOCK_SIZE, max_output=BLOCK_SIZE + 4)
    assert cache.allocate(r1)
    assert cache.allocate(r2)
    r1.status = r2.status = SequenceStatus.RUNNING
    sched.running_queue.extend([r1, r2])
    assert cache.memory_stats()["gpu_free_blocks"] == 0

    # r3, r4 injected directly into CPU tier (fills both CPU slots)
    # This simulates prior evictions without routing through GPU,
    # so no GPU blocks are freed and GPU stays at 0.
    r3 = _req("r3", prompt_len=BLOCK_SIZE, max_output=4)
    r4 = _req("r4", prompt_len=BLOCK_SIZE, max_output=4)
    cb3 = cache._cpu.allocate()
    cb4 = cache._cpu.allocate()
    assert cb3 is not None and cb4 is not None
    cache._cpu_table["r3"] = [cb3]
    cache._cpu_table["r4"] = [cb4]
    r3.last_accessed_time = 0.0   # r3 is "coldest" — will cascade first
    r4.last_accessed_time = 1e9
    r3.status = r4.status = SequenceStatus.SWAPPED_CPU
    sched.swapped_cpu_queue.extend([r3, r4])
    assert cache.memory_stats()["cpu_free_blocks"] == 0

    # Push r1 past its first block boundary so it needs a 2nd GPU block
    for _ in range(BLOCK_SIZE):
        r1.generated_tokens.append(42)

    out = sched.schedule()

    # r1 should have triggered: GPU OOM → CPU full → cascade r3 to storage
    assert len(sched.swapped_storage_queue) >= 1, (
        "Coldest CPU-swapped request must cascade to Tier 3 storage"
    )
    # r3 should be the one in storage (lowest last_accessed_time)
    assert r3.status == SequenceStatus.SWAPPED_STORAGE

    cache.cleanup()


# ── Elevator restore: Storage → CPU → GPU ─────────────────────────────────────

def test_elevator_restore_storage_to_gpu(setup):
    cache, sched = setup
    req = _req("r0", prompt_len=BLOCK_SIZE, max_output=4)
    sched.add_request(req)
    sched.schedule()

    # Manually push to storage tier
    cache.swap_gpu_to_cpu(req)
    req.status = SequenceStatus.SWAPPED_CPU
    sched.running_queue.remove(req)
    sched.swapped_cpu_queue.append(req)

    cache.swap_cpu_to_storage(req)
    req.status = SequenceStatus.SWAPPED_STORAGE
    sched.swapped_cpu_queue.remove(req)
    sched.swapped_storage_queue.append(req)

    assert req.status == SequenceStatus.SWAPPED_STORAGE

    # Schedule should restore: Storage → CPU → GPU in subsequent steps
    # Step 1: Storage → CPU
    out1 = sched.schedule()
    assert len(out1.swapped_storage_in) >= 1 or req.status in (
        SequenceStatus.SWAPPED_CPU,
        SequenceStatus.RUNNING,
    )

    # Step 2: CPU → GPU (if not already running)
    if req.status == SequenceStatus.SWAPPED_CPU:
        out2 = sched.schedule()
        assert req.status == SequenceStatus.RUNNING


# ── Predictive prefetcher ─────────────────────────────────────────────────────

def test_prefetcher_moves_storage_to_cpu_on_almost_done():
    cache = HierarchicalCacheManager(
        num_gpu_blocks=4,
        num_cpu_blocks=4,
        num_storage_blocks=16,
    )
    sched = HierarchicalAdaptiveScheduler(cache_manager=cache, max_batch_size=4)
    prefetcher = PredictivePrefetcher(cache_manager=cache, scheduler=sched, lookahead=2)

    # Add a running request that's almost done (1 token left)
    running = _req("running", prompt_len=BLOCK_SIZE, max_output=3)
    sched.add_request(running)
    sched.schedule()
    running.generated_tokens = [42, 42]  # 2/3 done, 1 left — within lookahead=2

    # Add a storage-swapped request
    stored = _req("stored", prompt_len=BLOCK_SIZE, max_output=4)
    sched.add_request(stored)
    sched.schedule()
    cache.swap_gpu_to_cpu(stored)
    stored.status = SequenceStatus.SWAPPED_CPU
    sched.running_queue.remove(stored)
    sched.swapped_cpu_queue.append(stored)
    cache.swap_cpu_to_storage(stored)
    stored.status = SequenceStatus.SWAPPED_STORAGE
    sched.swapped_cpu_queue.remove(stored)
    sched.swapped_storage_queue.append(stored)

    moved = prefetcher.tick()
    assert moved >= 1
    assert stored.status == SequenceStatus.SWAPPED_CPU
    assert prefetcher.total_prefetches >= 1

    cache.cleanup()


# ── has_unfinished_requests covers all queues ─────────────────────────────────

def test_has_unfinished_tracks_storage_queue(setup):
    cache, sched = setup
    assert not sched.has_unfinished_requests()
    req = _req("r0")
    sched.swapped_storage_queue.append(req)
    assert sched.has_unfinished_requests()
