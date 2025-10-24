import math
import torch
import torch.nn as nn

from .rope import apply_rotary_emb, repeat_kv, precompute_freqs_cis


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

        n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads


        self.n_local_heads = n_heads
        self.n_local_kv_heads = n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, self.n_local_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_local_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_local_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_local_heads * self.head_dim, dim, bias=False)

        self.register_buffer(
            "cache_k",
            torch.zeros(
                self.max_batch_size,
                self.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            ),
            persistent=False,
        )
        self.register_buffer(
            "cache_v",
            torch.zeros(
                self.max_batch_size,
                self.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq = self.wq(x).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        # 推理向：直接写入缓存
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = torch.softmax(scores.float(), dim=-1).type_as(xq)

        out = torch.matmul(scores, values)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


if __name__ == "__main__":
    # 准备参数和输入
    batch_size, seq_len, dim = 4, 16, 128
    n_heads, n_kv_heads = 8, 2
    head_dim = dim // n_heads

    # 初始化注意力模块
    attention = GroupedQueryAttention(
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_batch_size=batch_size,
        max_seq_len=seq_len,
    )

    # 准备输入
    x = torch.randn(batch_size, seq_len, dim)
    freqs_cis = precompute_freqs_cis(dim=head_dim, end=seq_len * 2)
    freqs_cis_slice = freqs_cis[:seq_len]

    # 执行前向传播
    output = attention(x, start_pos=0, freqs_cis=freqs_cis_slice)

    # 验证输出形状
    print("--- GroupedQueryAttention Test ---")
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)


