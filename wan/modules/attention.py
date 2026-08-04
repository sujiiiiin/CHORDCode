# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import torch
# import flash_attn.cute
# try:
#     import flash_attn.cute
#     from flash_attn.cute import flash_attn_varlen_func
#     FLASH_ATTN_4_AVAILABLE = True
# except:
#     FLASH_ATTN_4_AVAILABLE = False
FLASH_ATTN_4_AVAILABLE = False


try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
    'attention_with_qk',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    # print("print FLASH_ATTN_4_AVAILABLE: ", FLASH_ATTN_4_AVAILABLE)
    if FLASH_ATTN_4_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA4 now.
        # print("using fa4")
        x = flash_attn.cute.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
            # Note: dropout_p, window_size are not supported in FA3 now.
            x = flash_attn_interface.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                seqused_q=None,
                seqused_k=None,
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                softmax_scale=softmax_scale,
                causal=causal,
                # deterministic=deterministic)[0].unflatten(0, (b, lq))
                deterministic=deterministic).unflatten(0, (b, lq))
        else:
            assert FLASH_ATTN_2_AVAILABLE
            x = flash_attn.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                    0, dtype=torch.int32).to(q.device, non_blocking=True),
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)




def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if window_size != (-1, -1):
            warnings.warn(
                'Sliding-window attention is unavailable without FlashAttention; using full attention.'
            )

        # Preserve an existing half dtype. In particular, converting FP16 inputs
        # to the default BF16 here makes the fallback unusable on pre-Ampere GPUs.
        half_dtypes = (torch.float16, torch.bfloat16)
        compute_dtype = q.dtype if q.dtype in half_dtypes else dtype
        out_dtype = q.dtype
        q = q.transpose(1, 2).to(compute_dtype)
        k = k.transpose(1, 2).to(compute_dtype)
        v = v.transpose(1, 2).to(compute_dtype)
        if q_scale is not None:
            q = q * q_scale

        attn_mask = None
        if q_lens is not None or k_lens is not None:
            batch, _, lq, _ = q.shape
            lk = k.shape[2]
            if q_lens is None:
                q_lens = torch.full((batch,), lq, device=q.device)
            else:
                q_lens = torch.as_tensor(q_lens, device=q.device)
            if k_lens is None:
                k_lens = torch.full((batch,), lk, device=q.device)
            else:
                k_lens = torch.as_tensor(k_lens, device=q.device)
            valid_q = torch.arange(lq, device=q.device)[None, :] < q_lens[:, None]
            valid_k = torch.arange(lk, device=q.device)[None, :] < k_lens[:, None]
            attn_mask = (valid_q[:, None, :, None] & valid_k[:, None, None, :])

        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=causal and attn_mask is None,
            dropout_p=dropout_p,
            scale=softmax_scale,
        )

        out = out.transpose(1, 2).contiguous()
        return out.to(out_dtype)


def attention_with_qk(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    Compute attention output and return the post-softmax QK matrix.
    Returns:
        out: [B, Lq, Nq, C2]
        qk:  [B, Lq, Nq, Lk]

    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    q_scale:        float. Optional query scale applied before QK^T.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. Unused, for API parity.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    version:        Unused, for API parity.
    """
    del deterministic, version
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes

    out_dtype = q.dtype
    b, lq, lk = q.size(0), q.size(1), k.size(1)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    q = half(q)
    k = half(k)
    v = half(v)

    if q_scale is not None:
        q = q * q_scale

    n_q = q.size(2)
    n_k = k.size(2)
    if n_q % n_k != 0:
        raise ValueError("Nq must be divisible by Nk for grouped attention")
    if n_q != n_k:
        repeat = n_q // n_k
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)

    scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(q.size(-1)))

    q_t = q.permute(0, 2, 1, 3)
    k_t = k.permute(0, 2, 1, 3)
    v_t = v.permute(0, 2, 1, 3)

    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

    if q_lens is not None or k_lens is not None:
        if q_lens is None:
            q_lens = torch.full((b,), lq, device=q.device, dtype=torch.int64)
        if k_lens is None:
            k_lens = torch.full((b,), lk, device=q.device, dtype=torch.int64)
        q_lens = q_lens.to(device=q.device)
        k_lens = k_lens.to(device=q.device)
        q_mask = torch.arange(lq, device=q.device)[None, :] < q_lens[:, None]
        k_mask = torch.arange(lk, device=q.device)[None, :] < k_lens[:, None]
        valid = q_mask[:, None, :, None] & k_mask[:, None, None, :]
        scores = scores.masked_fill(~valid, float('-inf'))

    if causal:
        causal_mask = torch.tril(torch.ones(lq, lk, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask[None, None, :, :], float('-inf'))

    if window_size != (-1, -1):
        left, right = window_size
        left = lq if left < 0 else left
        right = lk if right < 0 else right
        q_idx = torch.arange(lq, device=q.device)[:, None]
        k_idx = torch.arange(lk, device=q.device)[None, :]
        window_mask = (k_idx >= (q_idx - left)) & (k_idx <= (q_idx + right))
        scores = scores.masked_fill(~window_mask[None, None, :, :], float('-inf'))

    scores = scores.float()
    qk = torch.softmax(scores, dim=-1)
    attn = qk

    if dropout_p > 0:
        attn = torch.nn.functional.dropout(attn, p=dropout_p, training=q.requires_grad)

    out = torch.matmul(attn, v_t)
    out = out.permute(0, 2, 1, 3).contiguous().type(out_dtype)
    qk = qk.permute(0, 2, 1, 3).contiguous()
    return out, qk
