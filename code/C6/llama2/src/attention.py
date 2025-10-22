import math
import torch
import torch.nn as nn

from .rope import apply_rotary_emb, repeat_kv


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

        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be multiple of n_kv_heads"

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
        assert 0 <= start_pos <= self.max_seq_len
        assert start_pos + seqlen <= self.max_seq_len

        xq = self.wq(x).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        # 训练时避免跨步累计计算图引用；每个 batch 起始位置清零对应样本 cache
        if self.training and start_pos == 0:
            self.cache_k[:bsz].zero_()
            self.cache_v[:bsz].zero_()

        # 写入缓存时切断梯度，避免下一步反向误用上一步的计算图
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk.detach()
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv.detach()

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


