import pytest
from src.models import SequenceRequest, BLOCK_SIZE
from src.cache_manager import HierarchicalCacheManager


def _req(rid: str, prompt_len: int, max_output: int = 10) -> SequenceRequest:
    return SequenceRequest(
        request_id=rid,
        prompt_tokens=list(range(prompt_len)),
        max_output_tokens=max_output,
    )


def test_allocate_and_free_reclaims_blocks():
    mgr = HierarchicalCacheManager(num_gpu_blocks=16)
    req = _req("r0", prompt_len=BLOCK_SIZE * 2)

    assert mgr.can_allocate(req)
    assert mgr.allocate(req)

    stats = mgr.memory_stats()
    assert stats["gpu_used_blocks"] == 2
    assert stats["gpu_free_blocks"] == 14

    mgr.free("r0")
    stats = mgr.memory_stats()
    assert stats["gpu_used_blocks"] == 0
    assert stats["gpu_free_blocks"] == 16


def test_oom_blocks_allocation():
    mgr = HierarchicalCacheManager(num_gpu_blocks=4)
    req = _req("r0", prompt_len=BLOCK_SIZE * 5)
    assert not mgr.can_allocate(req)
    assert not mgr.allocate(req)
    assert mgr.memory_stats()["gpu_used_blocks"] == 0


def test_append_slot_grows_on_block_boundary():
    mgr = HierarchicalCacheManager(num_gpu_blocks=16)
    req = _req("r0", prompt_len=BLOCK_SIZE)  # Exactly 1 block needed initially

    assert mgr.allocate(req)
    assert mgr.memory_stats()["gpu_used_blocks"] == 1

    # Fill first block, crossing the boundary triggers a new allocation
    for _ in range(BLOCK_SIZE):
        req.generated_tokens.append(42)

    assert mgr.append_slot(req)
    assert mgr.memory_stats()["gpu_used_blocks"] == 2


def test_append_slot_no_alloc_within_block():
    mgr = HierarchicalCacheManager(num_gpu_blocks=16)
    # Prompt fills BLOCK_SIZE-1 slots, so the first generated token stays in block 0
    req = _req("r0", prompt_len=BLOCK_SIZE - 1)

    assert mgr.allocate(req)
    used_before = mgr.memory_stats()["gpu_used_blocks"]

    req.generated_tokens.append(42)  # total_length = BLOCK_SIZE — still fits in 1 block
    assert mgr.append_slot(req)
    assert mgr.memory_stats()["gpu_used_blocks"] == used_before  # No new block


def test_swap_gpu_to_cpu_and_back():
    mgr = HierarchicalCacheManager(num_gpu_blocks=8, num_cpu_blocks=8)
    req = _req("r0", prompt_len=BLOCK_SIZE)

    mgr.allocate(req)
    assert mgr.memory_stats()["gpu_used_blocks"] == 1

    assert mgr.swap_gpu_to_cpu(req)
    stats = mgr.memory_stats()
    assert stats["gpu_used_blocks"] == 0
    assert stats["cpu_used_blocks"] == 1

    assert mgr.swap_cpu_to_gpu(req)
    stats = mgr.memory_stats()
    assert stats["gpu_used_blocks"] == 1
    assert stats["cpu_used_blocks"] == 0


def test_swap_gpu_to_cpu_fails_when_cpu_full():
    mgr = HierarchicalCacheManager(num_gpu_blocks=8, num_cpu_blocks=0)
    req = _req("r0", prompt_len=BLOCK_SIZE)
    mgr.allocate(req)
    assert not mgr.swap_gpu_to_cpu(req)


def test_no_external_fragmentation():
    """
    Freed physical blocks must be immediately reusable regardless of which
    sequence they previously belonged to. External fragmentation must be zero.
    """
    mgr = HierarchicalCacheManager(num_gpu_blocks=8)
    reqs = [_req(f"r{i}", BLOCK_SIZE) for i in range(8)]
    for r in reqs:
        assert mgr.allocate(r)
    assert mgr.memory_stats()["gpu_free_blocks"] == 0

    # Free 4 requests
    for r in reqs[:4]:
        mgr.free(r.request_id)
    assert mgr.memory_stats()["gpu_free_blocks"] == 4

    # New requests of any shape should be able to use freed blocks
    new_reqs = [_req(f"new{i}", BLOCK_SIZE) for i in range(4)]
    for r in new_reqs:
        assert mgr.can_allocate(r)
        assert mgr.allocate(r)
    assert mgr.memory_stats()["gpu_free_blocks"] == 0


def test_block_table_correctness():
    mgr = HierarchicalCacheManager(num_gpu_blocks=16)
    req = _req("r0", prompt_len=BLOCK_SIZE * 3)
    mgr.allocate(req)
    table = mgr.block_table_for("r0")
    assert len(table) == 3
    assert len(set(table)) == 3  # All distinct physical blocks
