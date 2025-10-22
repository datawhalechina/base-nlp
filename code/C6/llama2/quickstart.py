import torch

from src.transformer import LlamaTransformer


def main() -> None:
    model = LlamaTransformer(
        vocab_size=1000,
        dim=256,
        n_layers=2,
        n_heads=8,
        n_kv_heads=2,
        multiple_of=64,
        ffn_dim_multiplier=None,
        norm_eps=1e-6,
        max_batch_size=4,
        max_seq_len=64,
    )

    # 构造随机 token 序列并执行前向
    batch_size, seq_len = 2, 16
    tokens = torch.randint(0, 1000, (batch_size, seq_len))
    logits = model(tokens, start_pos=0)

    # 期望: [batch_size, seq_len, vocab_size]
    print("logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()


