import numpy as np

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
    x_np = emb_table[ids][None]
    return tokens, x_np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def manual_lstm_numpy(x_np, weights):
    """
    使用 NumPy 手动实现 LSTM (无偏置)
    
    Args:
        x_np: (B, T, E)
        weights: 包含 U_f, W_f, U_i, W_i, U_c, W_c, U_o, W_o 的元组
    Returns:
        outputs: (B, T, H)
        final_h: (B, H)
        final_c: (B, H)
    """
    U_f, W_f, U_i, W_i, U_c, W_c, U_o, W_o = weights
    B_local, T_local, _ = x_np.shape
    h_prev = np.zeros((B_local, H), dtype=np.float32)
    c_prev = np.zeros((B_local, H), dtype=np.float32)
    
    steps = []
    # 按时间步循环
    for t in range(T_local):
        x_t = x_np[:, t, :]
        
        # 1. 遗忘门
        f_t = sigmoid(x_t @ U_f + h_prev @ W_f)
        
        # 2. 输入门与候选记忆
        i_t = sigmoid(x_t @ U_i + h_prev @ W_i)
        c_tilde_t = np.tanh(x_t @ U_c + h_prev @ W_c)
        
        # 3. 更新细胞状态
        c_t = f_t * c_prev + i_t * c_tilde_t
        
        # 4. 输出门与隐藏状态
        o_t = sigmoid(x_t @ U_o + h_prev @ W_o)
        h_t = o_t * np.tanh(c_t)
        
        steps.append(h_t)
        h_prev, c_prev = h_t, c_t
        
    outputs = np.stack(steps, axis=1)
    return outputs, h_prev, c_prev


def main():
    _, x_np = prepare_inputs()
    
    # 初始化8个权重矩阵
    np.random.seed(7)
    U_f, W_f = np.random.randn(E, H).astype(np.float32), np.random.randn(H, H).astype(np.float32)
    U_i, W_i = np.random.randn(E, H).astype(np.float32), np.random.randn(H, H).astype(np.float32)
    U_c, W_c = np.random.randn(E, H).astype(np.float32), np.random.randn(H, H).astype(np.float32)
    U_o, W_o = np.random.randn(E, H).astype(np.float32), np.random.randn(H, H).astype(np.float32)
    
    weights_np = (U_f, W_f, U_i, W_i, U_c, W_c, U_o, W_o)

    # --- 手写 LSTM (使用 NumPy) ---
    print("--- 手写 LSTM (NumPy) ---")
    out_manual_np, hT_manual_np, cT_manual_np = manual_lstm_numpy(x_np, weights_np)
    print("输入形状:", x_np.shape)
    print("手写输出形状:", out_manual_np.shape)
    print("手写最终隐藏形状:", hT_manual_np.shape)
    print("手写最终细胞形状:", cT_manual_np.shape)


if __name__ == "__main__":
    main()
