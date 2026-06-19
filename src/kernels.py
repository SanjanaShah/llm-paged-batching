"""
Custom Triton kernels for AetherServe.

Two production-grade kernels, each with a numerically identical PyTorch
fallback for environments without CUDA (MPS, CPU-only, CI).

  _int4_dequant_kernel (Triton)
      Dequantizes INT4-packed weight tensors to FP16 on the GPU using
      per-group affine quantization (GPTQ / AWQ layout). Called once
      per Linear layer per decode step; the dequant FLOPS are far cheaper
      than the subsequent GEMM, so the effective bandwidth is ~4× better
      than loading a full FP16 weight matrix from HBM.

  _paged_decode_attn_kernel (Triton)
      Online-softmax decode attention over non-contiguous KV-cache blocks
      (PagedAttention, Kwon et al. SOSP 2023). Each Triton program handles
      one (batch, head) pair, iterates the block table, and accumulates
      Q·K/V without materialising the full attention-weight vector.

INT4 packing convention (matches GPTQ and AutoAWQ):
  For weight W of shape (out_features, in_features):
    - Grouped along in_features: group_size consecutive K-values share
      one (scale, zero_point) pair.
    - Packed along in_features: two adjacent K-positions share one uint8.
        packed[m, k]  →  low-nibble  = W[m, 2k]
                         high-nibble = W[m, 2k+1]
    - Stored shapes:
        packed : uint8  (out_features, in_features // 2)
        scales : fp16   (out_features, in_features // group_size)
        zeros  : fp16   (out_features, in_features // group_size)
"""

from typing import Tuple

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE: bool = True
except ImportError:
    TRITON_AVAILABLE = False

# torch is a soft dependency — imported lazily inside each function so that
# the rest of the engine (which doesn't use kernels.py) can run without it.


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernels (compiled on first call if TRITON_AVAILABLE)
# ─────────────────────────────────────────────────────────────────────────────

if TRITON_AVAILABLE:

    @triton.jit
    def _int4_dequant_kernel(
        packed_ptr,   # uint8  [out_features, in_features // 2]
        scales_ptr,   # fp16   [out_features, num_groups]
        zeros_ptr,    # fp16   [out_features, num_groups]
        out_ptr,      # fp16   [out_features, in_features]
        out_features,
        in_features,
        num_groups,
        group_size: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Dequantize INT4-packed weight matrix to FP16.

        Grid:  (cdiv(out_features, BLOCK_M),  cdiv(in_features, BLOCK_K))
        Each program loads a (BLOCK_M × BLOCK_K) tile of the output tensor.

        Packed layout: two consecutive in_feature positions share one uint8.
          byte[m, k//2]:  bits 0-3 → W[m, k]   (even k)
                          bits 4-7 → W[m, k+1] (odd  k)
        """
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

        m_mask = m_offs < out_features
        k_mask = k_offs < in_features
        mask = m_mask[:, None] & k_mask[None, :]

        # Each output col k lives in packed col k // 2
        packed = tl.load(
            packed_ptr
            + m_offs[:, None] * (in_features // 2)
            + (k_offs // 2)[None, :],
            mask=mask,
            other=0,
        ).to(tl.int32)

        # Unpack nibbles: even cols → low, odd cols → high
        nibble = tl.where(
            (k_offs % 2 == 1)[None, :],
            (packed >> 4) & 0xF,
            packed & 0xF,
        )

        # Per-group affine dequantization: w = (q - zero) * scale
        g_offs = k_offs // group_size
        scale = tl.load(
            scales_ptr + m_offs[:, None] * num_groups + g_offs[None, :],
            mask=mask,
            other=1.0,
        )
        zero = tl.load(
            zeros_ptr + m_offs[:, None] * num_groups + g_offs[None, :],
            mask=mask,
            other=0.0,
        )

        w = (nibble.to(tl.float16) - zero) * scale
        tl.store(
            out_ptr + m_offs[:, None] * in_features + k_offs[None, :],
            w,
            mask=mask,
        )

    @triton.jit
    def _paged_decode_attn_kernel(
        Q_ptr,             # fp16  [batch, num_heads, head_dim]
        K_cache_ptr,       # fp16  [total_blocks, block_size, num_kv_heads, head_dim]
        V_cache_ptr,       # fp16  [total_blocks, block_size, num_kv_heads, head_dim]
        block_table_ptr,   # i32   [batch, max_blocks_per_seq]
        seq_lens_ptr,      # i32   [batch]
        Out_ptr,           # fp16  [batch, num_heads, head_dim]
        sm_scale,
        # Strides
        stride_qb, stride_qh,
        stride_cb, stride_ct, stride_ch,
        # Compile-time constants
        num_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
        max_blocks_per_seq: tl.constexpr,
    ):
        """
        Decode-phase paged attention (one new query token per sequence).

        Each Triton program handles one (batch_id, head_id) pair.
        Iterates the block table to assemble attention over non-contiguous
        physical KV-cache blocks using FlashAttention 2 online softmax,
        avoiding the O(seq_len) memory allocation for attention weights.

        Grouped-query attention (GQA): num_kv_heads ≤ num_heads, each KV
        head services (num_heads // num_kv_heads) query heads.

        Grid: (batch_size, num_heads)
        """
        batch_id = tl.program_id(0)
        head_id = tl.program_id(1)
        kv_head_id = head_id * num_kv_heads // num_heads

        seq_len = tl.load(seq_lens_ptr + batch_id)

        # Load query vector [head_dim]
        q = tl.load(
            Q_ptr
            + batch_id * stride_qb
            + head_id * stride_qh
            + tl.arange(0, head_dim),
        ).to(tl.float32)

        # Online softmax state (FlashAttention 2 style)
        acc = tl.zeros([head_dim], dtype=tl.float32)
        m_i = tl.full([1], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([1], dtype=tl.float32)

        for blk_idx in range(max_blocks_per_seq):
            blk_start = blk_idx * block_size
            valid_blk = blk_start < seq_len

            phys_blk = tl.load(
                block_table_ptr + batch_id * max_blocks_per_seq + blk_idx,
                mask=valid_blk,
                other=0,
            )

            # KV tile: [block_size, head_dim]
            tok_offs = tl.arange(0, block_size)
            valid_tok = (blk_start + tok_offs < seq_len) & valid_blk
            kv_base = phys_blk * stride_cb + kv_head_id * stride_ch

            k_tile = tl.load(
                K_cache_ptr
                + kv_base
                + tok_offs[:, None] * stride_ct
                + tl.arange(0, head_dim)[None, :],
                mask=valid_tok[:, None],
                other=0.0,
            ).to(tl.float32)

            v_tile = tl.load(
                V_cache_ptr
                + kv_base
                + tok_offs[:, None] * stride_ct
                + tl.arange(0, head_dim)[None, :],
                mask=valid_tok[:, None],
                other=0.0,
            ).to(tl.float32)

            # QK scores and online softmax update
            scores = tl.sum(q[None, :] * k_tile, axis=1) * sm_scale
            scores = tl.where(valid_tok, scores, float("-inf"))

            m_blk = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, m_blk)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)  # [block_size]

            acc = alpha * acc + tl.sum(p[:, None] * v_tile, axis=0)
            l_i = alpha * l_i + tl.sum(p, axis=0)
            m_i = m_new

        out = acc / tl.maximum(l_i, 1e-6)
        tl.store(
            Out_ptr
            + batch_id * stride_qb
            + head_id * stride_qh
            + tl.arange(0, head_dim),
            out.to(tl.float16),
        )


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch fallbacks (CPU / MPS — numerically identical to the Triton paths)
# ─────────────────────────────────────────────────────────────────────────────


def _int4_dequant_pytorch(
    packed: "torch.Tensor",
    scales: "torch.Tensor",
    zeros: "torch.Tensor",
    group_size: int,
) -> "torch.Tensor":
    import torch

    out_features, half_k = packed.shape
    in_features = half_k * 2

    p = packed.to(torch.int32)
    low = p & 0xF
    high = (p >> 4) & 0xF
    q = torch.stack([low, high], dim=2).reshape(out_features, in_features)

    scale_exp = scales.repeat_interleave(group_size, dim=1)[:, :in_features]
    zero_exp = zeros.repeat_interleave(group_size, dim=1)[:, :in_features]

    return (q.to(torch.float16) - zero_exp) * scale_exp


def _paged_decode_attn_pytorch(
    q: "torch.Tensor",
    k_cache: "torch.Tensor",
    v_cache: "torch.Tensor",
    block_table: "torch.Tensor",
    seq_lens: "torch.Tensor",
    sm_scale: float,
) -> "torch.Tensor":
    import torch

    batch, num_heads, head_dim = q.shape
    block_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    out = torch.empty_like(q)

    for b in range(batch):
        seq_len = int(seq_lens[b].item())
        for h in range(num_heads):
            kv_h = h * num_kv_heads // num_heads
            qvec = q[b, h].float()

            num_blocks = (seq_len + block_size - 1) // block_size
            k_list, v_list = [], []
            for blk in range(num_blocks):
                phys = int(block_table[b, blk].item())
                n_tok = min(block_size, seq_len - blk * block_size)
                k_list.append(k_cache[phys, :n_tok, kv_h].float())
                v_list.append(v_cache[phys, :n_tok, kv_h].float())

            k_all = torch.cat(k_list, dim=0)
            v_all = torch.cat(v_list, dim=0)

            scores = (k_all @ qvec) * sm_scale
            weights = torch.softmax(scores, dim=0)
            out[b, h] = (weights.unsqueeze(1) * v_all).sum(0).to(q.dtype)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API — dispatch to Triton or PyTorch depending on availability / device
# ─────────────────────────────────────────────────────────────────────────────


def quantize_weights_int4(
    weight: "torch.Tensor",
    group_size: int = 128,
) -> "Tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """
    Quantize a Linear layer weight matrix to INT4 group quantization.

    Implements symmetric per-group min-max quantization matching GPTQ's
    storage layout (Frantar et al., ICLR 2023). Groups along in_features
    so each row slice of group_size values shares one scale/zero_point.

    Args:
        weight:     fp16/fp32 tensor of shape (out_features, in_features).
        group_size: number of consecutive in_feature positions per group.
                    Common values: 32, 64, 128.  Smaller = less accuracy
                    loss but more metadata overhead.

    Returns:
        packed:  uint8  (out_features, ceil(in_features / 2))
        scales:  fp16   (out_features, ceil(in_features / group_size))
        zeros:   fp16   same shape as scales
    """
    import torch
    import torch.nn.functional as F

    w = weight.detach().float()
    out_features, in_features = w.shape

    eff_gs = min(group_size, in_features)
    pad_target = max(eff_gs, 2)
    pad_k = (-in_features) % pad_target
    if pad_k:
        w = F.pad(w, (0, pad_k))
    k_padded = w.shape[1]

    num_groups = k_padded // eff_gs
    w3 = w.reshape(out_features, num_groups, eff_gs)

    w_min = w3.min(dim=2, keepdim=True).values
    w_max = w3.max(dim=2, keepdim=True).values
    scales = ((w_max - w_min) / 15.0).clamp(min=1e-5)
    zeros = w_min

    q = ((w3 - zeros) / scales).round().clamp(0, 15).to(torch.uint8)
    q = q.reshape(out_features, k_padded)[:, :in_features]

    if q.shape[1] % 2:
        q = F.pad(q.float(), (0, 1)).to(torch.uint8)

    low = q[:, 0::2] & 0xF
    high = (q[:, 1::2] & 0xF) << 4
    packed = low | high

    return (
        packed,
        scales.squeeze(2).to(torch.float16),
        zeros.squeeze(2).to(torch.float16),
    )


def dequantize_int4_weights(
    packed: "torch.Tensor",
    scales: "torch.Tensor",
    zeros: "torch.Tensor",
    group_size: int = 128,
) -> "torch.Tensor":
    """
    Dequantize INT4-packed weights to FP16.

    Dispatches to the Triton kernel on CUDA (hiding HBM bandwidth cost
    inside the compute pipeline), falls back to PyTorch on CPU / MPS.

    Args:
        packed:     uint8  (out_features, in_features // 2)
        scales:     fp16   (out_features, num_groups)
        zeros:      fp16   (out_features, num_groups)
        group_size: must match the value used in quantize_weights_int4.

    Returns:
        fp16 tensor of shape (out_features, in_features)
    """
    if TRITON_AVAILABLE and packed.is_cuda:
        import torch
        out_features = packed.shape[0]
        in_features = packed.shape[1] * 2
        num_groups = scales.shape[1]
        out = torch.empty(
            (out_features, in_features), dtype=torch.float16, device=packed.device
        )
        BLOCK_M, BLOCK_K = 32, 64
        grid = (
            triton.cdiv(out_features, BLOCK_M),
            triton.cdiv(in_features, BLOCK_K),
        )
        _int4_dequant_kernel[grid](
            packed, scales, zeros, out,
            out_features, in_features, num_groups,
            group_size=group_size,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
        )
        return out

    return _int4_dequant_pytorch(packed, scales, zeros, group_size)


def paged_decode_attention(
    q: "torch.Tensor",
    k_cache: "torch.Tensor",
    v_cache: "torch.Tensor",
    block_table: "torch.Tensor",
    seq_lens: "torch.Tensor",
    sm_scale: float,
) -> "torch.Tensor":
    """
    Decode-phase paged attention over non-contiguous KV-cache blocks.

    Args:
        q:           fp16  (batch, num_heads, head_dim)
        k_cache:     fp16  (total_blocks, block_size, num_kv_heads, head_dim)
        v_cache:     fp16  (total_blocks, block_size, num_kv_heads, head_dim)
        block_table: i32   (batch, max_blocks_per_seq)
        seq_lens:    i32   (batch,)
        sm_scale:    float  1.0 / sqrt(head_dim)

    Returns:
        fp16  (batch, num_heads, head_dim)
    """
    if TRITON_AVAILABLE and q.is_cuda:
        import torch
        batch, num_heads, head_dim = q.shape
        _, block_size, num_kv_heads, _ = k_cache.shape
        max_blocks = block_table.shape[1]
        out = torch.empty_like(q)
        _paged_decode_attn_kernel[(batch, num_heads)](
            q, k_cache, v_cache, block_table, seq_lens, out,
            sm_scale,
            q.stride(0), q.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            max_blocks_per_seq=max_blocks,
        )
        return out

    return _paged_decode_attn_pytorch(q, k_cache, v_cache, block_table, seq_lens, sm_scale)
