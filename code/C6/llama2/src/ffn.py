import torch
import torch.nn as nn
from typing import Optional


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))



if __name__ == "__main__":
    # 准备参数和输入
    batch_size, seq_len, dim = 4, 16, 128
    
    # 初始化 FFN 模块
    ffn = FeedForward(
        dim=dim,
        hidden_dim=4 * dim,
        multiple_of=256,
        ffn_dim_multiplier=None
    )

    # 准备输入
    x = torch.randn(batch_size, seq_len, dim)

    # 执行前向传播
    output = ffn(x)

    # 验证输出形状
    print("--- FeedForward (SwiGLU) Test ---")
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    assert x.shape == output.shape, "Shape mismatch"
    print("FeedForward test passed!")

