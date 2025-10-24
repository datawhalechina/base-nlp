import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._norm(x.float()).type_as(x)
        return out * self.weight


if __name__ == "__main__":
    # 准备参数和输入
    batch_size, seq_len, dim = 4, 16, 64
    x = torch.randn(batch_size, seq_len, dim)

    # 初始化并应用 RMSNorm
    norm = RMSNorm(dim)
    output = norm(x)

    # 验证输出形状
    print("--- RMSNorm Test ---")
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
