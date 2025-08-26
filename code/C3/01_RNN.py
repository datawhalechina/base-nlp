import torch
import torch.nn as nn
import numpy as np

# 约定: (B, T, E, H) 分别表示 批次/序列长度/输入维度/隐藏维度
B, E, H = 1, 128, 3


def prepare_inputs():
    """
    使用 NumPy 准备输入数据
    使用示例句子: "播放 周杰伦 的 《稻香》"
    构造最小词表和随机(可复现)词向量, 生成形状为 (B, T, E) 的输入张量。
    """
    np.random.seed(42)
    vocab = {"播放": 0, "周杰伦": 1, "的": 2, "《稻香》": 3}
    tokens = ["播放", "周杰伦", "的", "《稻香》"]
    ids = [vocab[t] for t in tokens]

    # 词向量表: (V, E)
    V = len(vocab)
    emb_table = np.random.randn(V, E).astype(np.float32)

    # 取出序列词向量并加上 batch 维度: (B, T, E)
    x_np = emb_table[ids][np.newaxis, :, :]
    return tokens, x_np


def manual_rnn_numpy(x_np, U_np, W_np):
    """
    使用 NumPy 手动实现 RNN(无偏置): h_t = tanh(U x_t + W h_{t-1})
    
    Args:
        x_np: (B, T, E)
        U_np: (E, H)
        W_np: (H, H)
    Returns:
        outputs: (B, T, H)
        final_h: (B, H)
    """
    B_local, T_local, _ = x_np.shape
    h_prev = np.zeros((B_local, H), dtype=np.float32)
    steps = []
    for t in range(T_local):
        x_t = x_np[:, t, :]
        h_t = np.tanh(x_t @ U_np + h_prev @ W_np)
        steps.append(h_t)
        h_prev = h_t
    outputs = np.stack(steps, axis=1)
    return outputs, h_prev


def pytorch_rnn_forward(x, U, W):
    """
    使用api nn.RNN (tanh, bias=False)。
    Returns:
        outputs: (B, T, H)
        final_h: (B, H)
    """
    rnn = nn.RNN(
        input_size=E,
        hidden_size=H,
        num_layers=1,
        nonlinearity='tanh',
        bias=False,
        batch_first=True,
        bidirectional=False,
    )
    with torch.no_grad():
        # PyTorch 内部存放的是转置后的权重
        rnn.weight_ih_l0.copy_(U.T)
        rnn.weight_hh_l0.copy_(W.T)
    y, h_n = rnn(x)
    return y, h_n.squeeze(0)


def main():
    _, x_np = prepare_inputs()

    # PyTorch 张量，用于 nn.RNN 模块
    x = torch.from_numpy(x_np).float()
    
    # 使用可学习参数 U, W（无偏置）
    torch.manual_seed(7)
    U = torch.randn(E, H)
    W = torch.randn(H, H)

    # --- 手写 RNN (使用 NumPy) ---
    U_np = U.detach().numpy()
    W_np = W.detach().numpy()

    print("--- 手写 RNN (NumPy) ---")
    out_manual_np, hT_manual_np = manual_rnn_numpy(x_np, U_np, W_np)
    print("输入形状:", x_np.shape)
    print("手写输出形状:", out_manual_np.shape)
    print("手写最终隐藏形状:", hT_manual_np.shape)

    print("\n--- PyTorch nn.RNN ---")
    out_torch, hT_torch = pytorch_rnn_forward(x, U, W)
    print("模块输出形状:", out_torch.shape)
    print("模块最终隐藏形状:", hT_torch.shape)

    print("\n--- 对齐验证 ---")
    # 将 NumPy 结果转回 PyTorch 张量以进行比较
    out_manual = torch.from_numpy(out_manual_np)
    hT_manual = torch.from_numpy(hT_manual_np)

    print("逐步输出一致:", torch.allclose(out_manual, out_torch, atol=1e-6))
    print("最终隐藏一致:", torch.allclose(hT_manual, hT_torch, atol=1e-6))
    print("最后一步输出等于最终隐藏:", torch.allclose(out_torch[:, -1, :], hT_torch, atol=1e-6))


if __name__ == "__main__":
    main()
