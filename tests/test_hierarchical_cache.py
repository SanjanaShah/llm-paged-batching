"""
Tests for 3-tier HierarchicalCacheManager.

Verifies that blocks migrate correctly between GPU, CPU DRAM, and the
NVMe-backed storage tier, and that freed blocks are recycled cleanly
at every tier (zero external fragmentation invariant).
"""
import pytest
from src.models import SequenceRequest, BLOCK_SIZE
from src.cache_manager import HierarchicalCacheManager


def _req(rid: str, prompt_len: int, max_output: int = 10) -> SequenceRequest:
    return SequenceRequest(
        request_id=rid,
        prompt_tokens=list(range(prompt_len)),
        max_output_tokens=max_output,
    )


@pytest.fixture
def mgr():
    m = HierarchicalCacheManager(
        num_gpu_blocks=8,
        num_cpu_blocks=8,
        num_storage_blocks=16,
    )
    yield m
    m.cleanup()


# ── Tier 1 (GPU) ─────────────────────────────────────────────────────────────

def test_gpu_allocate_and_free(mgr):
    req = _req("r0", prompt_len=BLOCK_SIZE)
    assert mgr.allocate(req)
    assert mgr.memory_stats()["gpu_used_blocks"] == 1
    mgr.free("r0")
    assert mgr.memory_stats()["gpu_used_blocks"] == 0
    assert mgr.memory_stats()["gpu_free_blocks"] == 8


def test_gpu_oom_blocks_allocation(mgr):
    req = _req("r0", prompt_len=BLOCK_SIZE * 9)  # needs 9, only 8 available
    assert not mgr.can_allocate(req)
    assert not mgr.allocate(req)


# ── Tier 1 → Tier 2 (GPU → CPU) ──────────────────────────────────────────────

def test_swap_gpu_to_cpu(mgr):
    req = _req("r0", prompt_len=BLOCK_SIZE)
    mgr.allocate(req)
    assert mgr.swap_gpu_to_cpu(req)
    s = mgr.memory_stats()
    assert s["gpu_used_blocks"] == 0
    assert s["cpu_used_blocks"] == 1


def test_swap_cpu_to_gpu_restores(mgr):
    req = _req("r0", prompt_len=BLOCK_SIZE)
    mgr.allocate(req)
    mgr.swap_gpu_to_cpu(req)
    assert mgr.swap_cpu_to_gpu(req)
    s = mgr.memory_stats()
    assert s["gpu_used_blocks"] == 1
    assert s["cpu_used_blocks"] == 0


def test_swap_gpu_to_cpu_fails_when_cpu_full():
    m = HierarchicalCacheManager(num_gpu_blocks=4, num_cpu_blocks=0, num_storage_blocks=0)
    req = _req("r0", prompt_len=BLOCK_SIZE)
    m.allocate(req)
    assert not m.swap_gpu_to_cpu(req)
    # GPU blocks should be restored on failure
    assert m.memory_stats()["gpu_used_blocks"] == 1


# ── Tier 2 → Tier 3 (CPU → Storage) ─────────────────────────────────────────

def test_swap_cpu_to_storage(mgr):
    req = _req("r0", prompt_len=BLOCK_SIZE)
    mgr.allocate(req)
    mgr.swap_gpu_to_cpu(req)
    assert mgr.swap_cpu_to_storage(req)
    s = mgr.memory_stats()
    assert s["cpu_used_blocks"] == 0
    assert s["storage_used_blocks"] == 1


def test_swap_storage_to_cpu_restores(mgr):
    req = _req("r0", prompt_len=BLOCK_SIZE)
    mgr.allocate(req)
    mgr.swap_gpu_to_cpu(req)
    mgr.swap_cpu_to_storage(req)
    assert mgr.swap_storage_to_cpu(req)
    s = mgr.memory_stats()
    assert s["cpu_used_blocks"] == 1
    assert s["storage_used_blocks"] == 0


def test_full_round_trip_tier1_to_tier3_and_back(mgr):
    req = _req("r0", prompt_len=BLOCK_SIZE * 2)
    assert mgr.allocate(req)                   # GPU
    assert mgr.swap_gpu_to_cpu(req)            # GPU → CPU
    assert mgr.swap_cpu_to_storage(req)        # CPU → Storage
    assert mgr.swap_storage_to_cpu(req)        # Storage → CPU
    assert mgr.swap_cpu_to_gpu(req)            # CPU → GPU
    s = mgr.memory_stats()
    assert s["gpu_used_blocks"] == 2
    assert s["cpu_used_blocks"] == 0
    assert s["storage_used_blocks"] == 0


# ── free_all cleans every tier ────────────────────────────────────────────────

def test_free_all_reclaims_gpu(mgr):
    req = _req("r0", prompt_len=BLOCK_SIZE)
    mgr.allocate(req)
    mgr.free_all("r0")
    assert mgr.memory_stats()["gpu_used_blocks"] == 0


def test_free_all_reclaims_storage(mgr):
    req = _req("r0", prompt_len=BLOCK_SIZE)
    mgr.allocate(req)
    mgr.swap_gpu_to_cpu(req)
    mgr.swap_cpu_to_storage(req)
    mgr.free_all("r0")
    s = mgr.memory_stats()
    assert s["gpu_used_blocks"] == 0
    assert s["cpu_used_blocks"] == 0
    assert s["storage_used_blocks"] == 0


# ── No external fragmentation at any tier ────────────────────────────────────

def test_storage_blocks_recycled_after_free(mgr):
    reqs = [_req(f"r{i}", BLOCK_SIZE) for i in range(4)]
    for r in reqs:
        mgr.allocate(r)
        mgr.swap_gpu_to_cpu(r)
        mgr.swap_cpu_to_storage(r)

    assert mgr.memory_stats()["storage_used_blocks"] == 4

    for r in reqs[:2]:
        mgr.free_all(r.request_id)

    assert mgr.memory_stats()["storage_free_blocks"] >= 2

    new_reqs = [_req(f"new{i}", BLOCK_SIZE) for i in range(2)]
    for r in new_reqs:
        mgr.allocate(r)
        mgr.swap_gpu_to_cpu(r)
        assert mgr.swap_cpu_to_storage(r), "freed storage blocks must be reusable"
