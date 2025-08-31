import torch
import torch.nn as nn
import random

torch.manual_seed(42)

# ==============================================================================
#   1. 全局配置
# ==============================================================================
batch_size = 8
src_len = 10
trg_len = 12
src_vocab_size = 100
trg_vocab_size = 120
hidden_size = 64
num_layers = 2
sos_idx = 1  # Start of Sentence token index
eos_idx = 2  # End of Sentence token index


# ==============================================================================
#   2. 模型定义
# ==============================================================================

class Encoder(nn.Module):
    """编码器: 读取输入序列，输出上下文向量（隐藏状态和细胞状态）。"""
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        self.rnn = nn.LSTM(
            input_size=hidden_size,      # 输入特征维度
            hidden_size=hidden_size,     # 隐藏状态维度
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    """解码器: 接收上一个预测的token和当前状态，单步输出预测和新状态。"""
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        predictions = self.fc(outputs.squeeze(1))
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    """Seq2Seq 包装模块: 管理 Encoder 和 Decoder。"""
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """训练模式下的前向传播，使用 Teacher Forcing。"""
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs

    def greedy_decode(self, src, max_len=trg_len):
        """推理模式下的高效贪心解码。"""
        self.eval()
        with torch.no_grad():
            hidden, cell = self.encoder(src)
            trg_indexes = [sos_idx]
            for _ in range(max_len):
                trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device)
                output, hidden, cell = self.decoder(trg_tensor, hidden, cell)
                pred_token = output.argmax(1).item()
                trg_indexes.append(pred_token)
                if pred_token == eos_idx:
                    break
        return trg_indexes


# ==============================================================================
#   3. 变体模型定义
# ==============================================================================

class DecoderForBadInference(nn.Module):
    """一个特殊的解码器，其forward实现是为了演示“从头计算”的低效推理。"""
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(DecoderForBadInference, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        outputs, _ = self.rnn(embedded, (hidden, cell))
        last_output = outputs[:, -1, :]
        predictions = self.fc(last_output)
        return predictions


class DecoderAlt(nn.Module):
    """解码器变体: 不用上下文初始化状态，而是在每步将其作为输入。"""
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(DecoderAlt, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        # 注意：这里的输入维度是两个 hidden_size 拼接而成
        self.rnn = nn.LSTM(
            input_size=hidden_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, hidden_ctx, hidden, cell):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)
        context = hidden_ctx[-1].unsqueeze(1).repeat(1, embedded.shape[1], 1)
        rnn_input = torch.cat((embedded, context), dim=2)
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        predictions = self.fc(outputs.squeeze(1))
        return predictions, hidden, cell


# ==============================================================================
#   4. 演示主流程
# ==============================================================================

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 统一初始化模型 ---
    encoder = Encoder(src_vocab_size, hidden_size, num_layers).to(device)
    decoder = Decoder(trg_vocab_size, hidden_size, num_layers).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # --- 统一创建伪数据 ---
    src = torch.randint(1, src_vocab_size, (batch_size, src_len)).to(device)
    trg = torch.randint(1, trg_vocab_size, (batch_size, trg_len)).to(device)

    # =========================================
    # 演示1: 低效的推理实现
    # =========================================
    print("\n" + "="*25 + " 演示1: 低效推理 " + "="*25)
    decoder_bad = DecoderForBadInference(trg_vocab_size, hidden_size, num_layers).to(device)

    def inefficient_greedy_decode(encoder, decoder, src, max_len=trg_len):
        with torch.no_grad():
            hidden, cell = encoder(src)
            trg_indexes = [sos_idx]
            for i in range(max_len):
                trg_tensor = torch.LongTensor([trg_indexes]).to(device)
                print(f"  第 {i + 1} 步，解码器输入 shape: {trg_tensor.shape}")
                output = decoder(trg_tensor, hidden, cell)
                pred_token = output.argmax(1).item()
                trg_indexes.append(pred_token)
                if pred_token == eos_idx:
                    break
        return trg_indexes
    
    prediction_bad = inefficient_greedy_decode(encoder, decoder_bad, src[0:1, :])
    print(f"低效推理的预测结果: {prediction_bad}")

    # =========================================
    # 演示2: 标准训练 & 高效推理
    # =========================================
    print("\n" + "="*25 + " 演示2: 标准模式 " + "="*25)
    # 训练过程模拟 (Teacher Forcing)
    model.train()
    outputs = model(src, trg, teacher_forcing_ratio=0.8)
    print(f"训练模式输出张量形状: {outputs.shape}")
    # 推理过程模拟 (高效的自回归)
    prediction = model.greedy_decode(src[0:1, :])
    print(f"高效推理的预测结果: {prediction}")

    # =========================================
    # 演示3: 上下文向量的另一种用法
    # =========================================
    print("\n" + "="*23 + " 演示3: 上下文变体用法 " + "="*23)
    decoder_alt = DecoderAlt(trg_vocab_size, hidden_size, num_layers).to(device)
    
    def alternative_greedy_decode(encoder, decoder, src, max_len=trg_len):
        with torch.no_grad():
            hidden_ctx, cell_ctx = encoder(src)
            trg_indexes = [sos_idx]
            # 初始化解码器的"真实"状态为0
            batch_size = src.shape[0]
            hidden = torch.zeros(num_layers, batch_size, hidden_size).to(device)
            cell = torch.zeros(num_layers, batch_size, hidden_size).to(device)
            
            for _ in range(max_len):
                trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
                output, hidden, cell = decoder(trg_tensor, hidden_ctx, hidden, cell)
                pred_token = output.argmax(1).item()
                trg_indexes.append(pred_token)
                if pred_token == eos_idx:
                    break
        return trg_indexes

    prediction_alt = alternative_greedy_decode(encoder, decoder_alt, src[0:1, :])
    print(f"变体用法预测结果: {prediction_alt}")
