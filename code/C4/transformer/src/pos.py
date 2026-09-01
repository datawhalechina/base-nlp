import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    正弦位置编码
    Transformer 论文中使用固定公式计算位置编码，不涉及可学习参数。
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, dim, max_seq_len=5000):
        super().__init__()
        
        # 创建一个足够长的 PE 矩阵 [max_seq_len, dim]
        pe = torch.zeros(max_seq_len, dim)
        
        # 生成位置索引 [0, 1, ..., max_seq_len-1] -> [max_seq_len, 1]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # 计算分母中的 div_term: 10000^(2i/dim) = exp(2i * -log(10000)/dim)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        # 填充 PE 矩阵
        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        # dim 为奇数时，奇数索引槽位比 div_term 少一个
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        
        # 增加 batch 维度: [1, max_seq_len, dim] 以便广播
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer，不会被视为模型参数（不参与梯度更新），但会随模型 state_dict 保存
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: 输入的词嵌入序列 [batch_size, seq_len, dim]
        Returns:
            加上位置编码后的序列 [batch_size, seq_len, dim]
        """
        # 截取与输入序列长度对应的位置编码并相加
        # x.size(1) 是 seq_len
        x = x + self.pe[:, :x.size(1), :]
        return x

if __name__ == "__main__":
    # 准备参数
    batch_size, seq_len, dim = 2, 10, 512
    max_seq_len = 100
    
    # 初始化模块
    pe = PositionalEncoding(dim, max_seq_len)
    
    # 准备输入
    x = torch.zeros(batch_size, seq_len, dim) # 输入为0，直接观察PE值
    
    # 前向传播
    output = pe(x)
    
    # 验证输出
    print("--- PositionalEncoding Test ---")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
