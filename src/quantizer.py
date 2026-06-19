"""
Per-block INT8 affine quantization for NVMe KV-cache compression.

Implements the block-level min-max quantization scheme described in TTKV
(Xu et al., ArXiv 2024). Each BLOCK_SIZE-token block is quantized
independently with its own (scale, zero_point) pair stored inline —
matching the paper's design rationale: per-block granularity isolates
outlier token IDs to a single block, preventing them from inflating the
quantization error across the whole sequence.

Storage layout per block (for BLOCK_SIZE = 16):
  [16 bytes] unsigned INT8 quantized values
  [ 8 bytes] float64 scale
  [ 8 bytes] float64 zero_point
  ─────────────────────────────────
  [32 bytes] total  vs  [64 bytes] uncompressed (uint32 × BLOCK_SIZE)
  Compression ratio: 2.0×

In a production system the KV-cache stores float16 key/value tensors, not
token IDs. Quantizing float16 to INT8 also yields 2× compression with a
small accuracy penalty (~0.5% perplexity increase per the TTKV paper).
AetherServe simulates this with uint32 token IDs as a stand-in for the
float16 KV vectors, so the quantization math, storage layout, and I/O
semantics are identical — only the data type of the payload differs.
"""

import struct
from typing import List

from src.models import BLOCK_SIZE

_Q_FMT = f">{BLOCK_SIZE}B"            # BLOCK_SIZE unsigned bytes (INT8)
_META_FMT = ">dd"                     # float64 scale + float64 zero_point
_META_BYTES: int = struct.calcsize(_META_FMT)  # always 16

# Byte footprint constants exposed for the rest of the system
QUANTIZED_BYTES_PER_BLOCK: int = BLOCK_SIZE + _META_BYTES   # 32 for BLOCK_SIZE=16
UNCOMPRESSED_BYTES_PER_BLOCK: int = BLOCK_SIZE * 4          # 64 for BLOCK_SIZE=16
COMPRESSION_RATIO: float = UNCOMPRESSED_BYTES_PER_BLOCK / QUANTIZED_BYTES_PER_BLOCK  # 2.0


def quantize_block(tokens: List[int]) -> bytes:
    """
    Affine-quantize a list of token IDs to INT8, return packed bytes.

    The block is zero-padded to BLOCK_SIZE before quantization so callers
    don't need to handle partial trailing blocks specially.

    Returns exactly QUANTIZED_BYTES_PER_BLOCK bytes:
      [BLOCK_SIZE uint8 values] ++ [float64 scale] ++ [float64 zero_point]
    """
    padded = (list(tokens) + [0] * BLOCK_SIZE)[:BLOCK_SIZE]
    min_val = float(min(padded))
    max_val = float(max(padded))

    # Degenerate block (all identical): scale = 1.0 to avoid division by zero
    scale = (max_val - min_val) / 255.0 if max_val > min_val else 1.0

    q_vals = [min(255, max(0, round((t - min_val) / scale))) for t in padded]

    return struct.pack(_Q_FMT, *q_vals) + struct.pack(_META_FMT, scale, min_val)


def dequantize_block(data: bytes) -> List[int]:
    """
    Dequantize QUANTIZED_BYTES_PER_BLOCK bytes back to token IDs.

    Returns exactly BLOCK_SIZE integers. Reconstruction error is bounded by
    ⌈scale / 2⌉ where scale = (max_val - min_val) / 255. For small-integer
    token ID sequences (typical in tests) the error is zero; for full
    vocabulary ranges (0–50 255) the max per-token error is ≈ 98.
    """
    q_vals = struct.unpack(_Q_FMT, data[:BLOCK_SIZE])
    scale, zero_point = struct.unpack(_META_FMT, data[BLOCK_SIZE : BLOCK_SIZE + _META_BYTES])
    return [round(q * scale + zero_point) for q in q_vals]


def quantization_error(original: List[int], reconstructed: List[int]) -> float:
    """Mean absolute error between original and reconstructed token IDs."""
    n = min(len(original), len(reconstructed))
    if n == 0:
        return 0.0
    return sum(abs(o - r) for o, r in zip(original[:n], reconstructed[:n])) / n
