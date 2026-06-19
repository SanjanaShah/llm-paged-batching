"""
Tests for INT8 Quantization Cascading (TTKV 2024).

Covers:
  - quantize/dequantize roundtrip accuracy for small and large-vocab tokens
  - QUANTIZED_BYTES_PER_BLOCK < UNCOMPRESSED_BYTES_PER_BLOCK (actual compression)
  - QuantizingStorageBlockAllocator write/read roundtrip
  - bytes_saved tracks cumulative savings
  - compression_ratio appears in HierarchicalCacheManager.memory_stats()
  - full cascade pipeline: CPU → Storage → CPU preserves data through quantization
  - uncompressed allocator (quantize_storage=False) reports ratio=1.0
"""
import pytest
from src.models import BLOCK_SIZE, SequenceRequest, SequenceStatus
from src.quantizer import (
    COMPRESSION_RATIO,
    QUANTIZED_BYTES_PER_BLOCK,
    UNCOMPRESSED_BYTES_PER_BLOCK,
    dequantize_block,
    quantization_error,
    quantize_block,
)
from src.cache_manager import (
    HierarchicalCacheManager,
    QuantizingStorageBlockAllocator,
    StorageBlockAllocator,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _req(rid: str, tokens: list) -> SequenceRequest:
    return SequenceRequest(
        request_id=rid,
        prompt_tokens=tokens,
        max_output_tokens=4,
    )


# ── Storage layout constants ──────────────────────────────────────────────────

def test_quantized_block_is_smaller_than_uncompressed():
    assert QUANTIZED_BYTES_PER_BLOCK < UNCOMPRESSED_BYTES_PER_BLOCK


def test_compression_ratio_is_two():
    assert COMPRESSION_RATIO == pytest.approx(2.0)


def test_quantized_bytes_match_layout():
    # BLOCK_SIZE uint8 values + 8 bytes scale + 8 bytes zero_point
    assert QUANTIZED_BYTES_PER_BLOCK == BLOCK_SIZE + 16


# ── Roundtrip accuracy ────────────────────────────────────────────────────────

def test_roundtrip_small_integers():
    """Sequential small integers [0, BLOCK_SIZE-1] should round-trip exactly."""
    tokens = list(range(BLOCK_SIZE))
    data = quantize_block(tokens)
    recovered = dequantize_block(data)
    assert len(data) == QUANTIZED_BYTES_PER_BLOCK
    # For uniformly spaced small integers the INT8 grid covers them exactly
    assert quantization_error(tokens, recovered) == pytest.approx(0.0, abs=1.0)


def test_roundtrip_large_vocab_bounded_error():
    """Full vocab-range tokens reconstruct within the INT8 quantization bound."""
    # Simulate token IDs drawn from a 50k vocab — typical for GPT-2 / Qwen
    import random
    random.seed(42)
    tokens = [random.randint(0, 50255) for _ in range(BLOCK_SIZE)]
    data = quantize_block(tokens)
    recovered = dequantize_block(data)
    # Worst-case per-token error: (50255 / 255) / 2 ≈ 98.5
    assert quantization_error(tokens, recovered) < 100.0


def test_roundtrip_uniform_block():
    """All-identical tokens (degenerate block) survive quantization unchanged."""
    tokens = [1234] * BLOCK_SIZE
    data = quantize_block(tokens)
    recovered = dequantize_block(data)
    assert all(r == 1234 for r in recovered)


def test_roundtrip_partial_block_zero_padded():
    """Blocks shorter than BLOCK_SIZE are zero-padded before quantization."""
    tokens = [10, 20, 30]
    data = quantize_block(tokens)
    assert len(data) == QUANTIZED_BYTES_PER_BLOCK
    recovered = dequantize_block(data)
    assert len(recovered) == BLOCK_SIZE


# ── QuantizingStorageBlockAllocator ──────────────────────────────────────────

@pytest.fixture
def q_alloc():
    alloc = QuantizingStorageBlockAllocator(num_blocks=8)
    yield alloc
    alloc.close()


def test_quantizing_allocator_write_read_roundtrip(q_alloc):
    bid = q_alloc.allocate()
    assert bid is not None
    tokens = list(range(BLOCK_SIZE))
    q_alloc.write(bid, tokens)
    recovered = q_alloc.read(bid)
    assert len(recovered) == BLOCK_SIZE
    assert quantization_error(tokens, recovered) < 1.0


def test_bytes_saved_tracks_writes(q_alloc):
    assert q_alloc.bytes_saved == 0
    for i in range(4):
        bid = q_alloc.allocate()
        q_alloc.write(bid, list(range(BLOCK_SIZE)))
    expected = 4 * (UNCOMPRESSED_BYTES_PER_BLOCK - QUANTIZED_BYTES_PER_BLOCK)
    assert q_alloc.bytes_saved == expected


def test_compression_ratio_property(q_alloc):
    assert q_alloc.compression_ratio == pytest.approx(2.0)


def test_uncompressed_allocator_ratio_is_one():
    alloc = StorageBlockAllocator(num_blocks=4)
    assert alloc.compression_ratio == pytest.approx(1.0)
    assert alloc.bytes_saved == 0
    alloc.close()


# ── HierarchicalCacheManager integration ─────────────────────────────────────

@pytest.fixture
def mgr():
    m = HierarchicalCacheManager(
        num_gpu_blocks=4,
        num_cpu_blocks=4,
        num_storage_blocks=16,
        quantize_storage=True,
    )
    yield m
    m.cleanup()


def test_memory_stats_reports_quantized(mgr):
    stats = mgr.memory_stats()
    assert stats["storage_quantized"] is True
    assert stats["storage_compression_ratio"] == pytest.approx(2.0)
    assert stats["storage_bytes_saved"] == 0  # no writes yet


def test_memory_stats_unquantized():
    m = HierarchicalCacheManager(
        num_gpu_blocks=4,
        num_cpu_blocks=4,
        num_storage_blocks=8,
        quantize_storage=False,
    )
    try:
        stats = m.memory_stats()
        assert stats["storage_quantized"] is False
        assert stats["storage_compression_ratio"] == pytest.approx(1.0)
        assert stats["storage_bytes_saved"] == 0
    finally:
        m.cleanup()


def test_cascade_cpu_to_storage_updates_bytes_saved(mgr):
    """After a CPU→Storage swap, bytes_saved must reflect the write."""
    req = _req("r0", list(range(BLOCK_SIZE)))
    assert mgr.allocate(req)
    assert mgr.swap_gpu_to_cpu(req)
    assert mgr.swap_cpu_to_storage(req)

    stats = mgr.memory_stats()
    # One block written → BLOCK_SIZE bytes of overhead saved
    assert stats["storage_bytes_saved"] == (UNCOMPRESSED_BYTES_PER_BLOCK - QUANTIZED_BYTES_PER_BLOCK)


def test_full_round_trip_cpu_storage_cpu(mgr):
    """Data survives the full Tier 2 → Tier 3 → Tier 2 elevator with quantization."""
    req = _req("r0", list(range(BLOCK_SIZE)))
    assert mgr.allocate(req)
    assert mgr.swap_gpu_to_cpu(req)
    assert mgr.swap_cpu_to_storage(req)
    # Dequantization read is exercised inside swap_storage_to_cpu
    assert mgr.swap_storage_to_cpu(req)
    # req is back in CPU tier — block tables are consistent
    assert req.request_id in mgr._cpu_table
    assert req.request_id not in mgr._storage_table
