import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    """
    自注意力模块
    """
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        
        # 定义Q, K, V的线性变换层
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量, 形状为 [batch_size, seq_len, hidden_size]
        
        Returns:
            torch.Tensor: 输出张量, 形状为 [batch_size, seq_len, hidden_size]
        """
        # 1. 线性变换得到 Q, K, V
        # Q, K, V shape: [batch_size, seq_len, hidden_size]
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        # 2. 计算注意力分数, 并进行缩放
        # k.transpose(-2, -1) shape: [batch_size, hidden_size, seq_len]
        # scores shape: [batch_size, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_size)
        
        # 3. 对分数进行 Softmax 归一化
        # attention_weights shape: [batch_size, seq_len, seq_len]
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 4. 注意力权重与 V 相乘，进行加权求和
        # context shape: [batch_size, seq_len, hidden_size]
        context = torch.matmul(attention_weights, v)
        
        return context

class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力模块
    """
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # 定义Q, K, V的线性变换层
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        
        # 输出层的线性变换
        self.wo = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量, 形状为 [batch_size, seq_len, hidden_size]
        
        Returns:
            torch.Tensor: 输出张量, 形状为 [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 线性变换得到 Q, K, V
        # q, k, v shape: [batch_size, seq_len, hidden_size]
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        # 2. 将 Q, K, V 拆分为多个头
        # view: [batch_size, seq_len, num_heads, head_dim]
        # transpose: [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. 计算注意力分数
        # k.transpose(-2, -1) shape: [batch_size, num_heads, head_dim, seq_len]
        # scores shape: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 4. 对分数进行 Softmax 归一化
        # attention_weights shape: [batch_size, num_heads, seq_len, seq_len]
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 5. 注意力权重与 V 相乘
        # context shape: [batch_size, num_heads, seq_len, head_dim]
        context = torch.matmul(attention_weights, v)
        
        # 6. 合并多个头的结果
        # transpose: [batch_size, seq_len, num_heads, head_dim]
        # contiguous().view(): [batch_size, seq_len, hidden_size]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # 7. 通过输出线性层
        output = self.wo(context)
        
        return output

if __name__ == '__main__':
    # --- 全局参数 ---
    batch_size = 8
    seq_len = 10
    hidden_size = 512
    num_heads = 8

    # --- 伪数据 ---
    # 创建一个随机输入张量
    input_tensor = torch.rand(batch_size, seq_len, hidden_size)
    print(f"输入张量形状: {input_tensor.shape}\n")

    # =========================================
    # 1: 单头自注意力
    # =========================================
    print("="*20 + " 1: 测试单头自注意力 " + "="*20)
    self_attention = SelfAttention(hidden_size)
    output_self = self_attention(input_tensor)
    print(f"单头自注意力输出形状: {output_self.shape}\n")

    # =========================================
    # 2: 多头自注意力
    # =========================================
    print("="*20 + " 2: 测试多头自注意力 " + "="*20)
    multi_head_attention = MultiHeadSelfAttention(hidden_size, num_heads)
    output_multi_head = multi_head_attention(input_tensor)
    print(f"多头自注意力输出形状: {output_multi_head.shape}")
