import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 允许 GQA：Q/K 头数可不同，但最后一维 head_dim 应一致
    head_dim = xq.shape[-1]

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # 分别对 Q/K 广播以兼容不同头数
    freqs_q = reshape_for_broadcast(freqs_cis, xq_)
    freqs_k = reshape_for_broadcast(freqs_cis, xk_)
    xq_out = torch.view_as_real(xq_ * freqs_q).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_k).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xq)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bsz, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bsz, seqlen, n_kv_heads, n_rep, head_dim)
        .reshape(bsz, seqlen, n_kv_heads * n_rep, head_dim)
    )


if __name__ == "__main__":
    # 准备参数和输入
    batch_size, seq_len, n_heads, n_kv_heads, head_dim = 4, 16, 8, 2, 16
    dim = n_heads * head_dim
    n_rep = n_heads // n_kv_heads

    # --- Test precompute_freqs_cis ---
    print("--- Test precompute_freqs_cis ---")
    freqs_cis = precompute_freqs_cis(dim=head_dim, end=seq_len * 2)
    print("freqs_cis shape:", freqs_cis.shape)

    # --- Test apply_rotary_emb ---
    print("\n--- Test apply_rotary_emb ---")
    xq = torch.randn(batch_size, seq_len, n_heads, head_dim)
    xk = torch.randn(batch_size, seq_len, n_kv_heads, head_dim)
    freqs_cis_slice = freqs_cis[:seq_len]
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis_slice)
    print("xq shape (in/out):", xq.shape, xq_out.shape)
    print("xk shape (in/out):", xk.shape, xk_out.shape)


