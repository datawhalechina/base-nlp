import torch
import torch.nn as nn
from typing import Optional

from .norm import RMSNorm
from .rope import precompute_freqs_cis
from .attention import GroupedQueryAttention
from .ffn import FeedForward


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        norm_eps: float,
        max_batch_size: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = GroupedQueryAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class LlamaTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
        norm_eps: float = 1e-6,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        assert vocab_size > 0, "vocab_size must be > 0"
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dim = dim
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(
                i,
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                multiple_of=multiple_of,
                ffn_dim_multiplier=ffn_dim_multiplier,
                norm_eps=norm_eps,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
            )
            for i in range(n_layers)
        ])
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(dim // n_heads, max_seq_len * 2),
            persistent=False,
        )

    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        bsz, seqlen = tokens.shape
        assert tokens.dtype == torch.long
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([torch.zeros((seqlen, start_pos), device=tokens.device), mask]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        logits = self.output(h).float()
        return logits


