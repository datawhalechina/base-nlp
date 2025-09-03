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
sos_idx = 1
eos_idx = 2


# ==============================================================================
#   2. 模型定义
# ==============================================================================

class Encoder(nn.Module):
    """编码器: 读取输入序列，输出所有时间步的特征以及最终状态。"""
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # 使用双向LSTM
        )
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.rnn(embedded)
        outputs = torch.tanh(self.fc(outputs))
        return outputs, hidden, cell

class DecoderWithAttention(nn.Module):
    """带注意力的通用解码器"""
    def __init__(self, vocab_size, hidden_size, num_layers, attention_module):
        super(DecoderWithAttention, self).__init__()
        self.attention = attention_module
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        self.rnn = nn.LSTM(
            input_size=hidden_size * 2,  # 输入维度是 词嵌入 + 上下文向量
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)
        
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        context = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        predictions = self.fc(outputs.squeeze(1))
        
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    """带注意力的Seq2Seq"""
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)
        
        hidden = hidden.view(self.encoder.rnn.num_layers, 2, batch_size, -1).sum(dim=1)
        cell = cell.view(self.encoder.rnn.num_layers, 2, batch_size, -1).sum(dim=1)

        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
            
        return outputs

    def greedy_decode(self, src, max_len=trg_len):
        self.eval()
        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src)
            hidden = hidden.view(self.encoder.rnn.num_layers, 2, src.shape[0], -1).sum(axis=1)
            cell = cell.view(self.encoder.rnn.num_layers, 2, src.shape[0], -1).sum(axis=1)

            trg_indexes = [sos_idx]
            for _ in range(max_len):
                input_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device)
                output, hidden, cell = self.decoder(input_tensor, hidden, cell, encoder_outputs)
                pred_token = output.argmax(1).item()
                trg_indexes.append(pred_token)
                if pred_token == eos_idx:
                    break
        return trg_indexes

# ==============================================================================
#   3. Attention 定义
# ==============================================================================

class AttentionSimple(nn.Module):
    """1: 无参数的注意力模块"""
    def __init__(self, hidden_size):
        super(AttentionSimple, self).__init__()
        # 确保缩放因子是一个 non-learnable buffer
        self.register_buffer("scale_factor", torch.sqrt(torch.FloatTensor([hidden_size])))

    def forward(self, hidden, encoder_outputs):
        # hidden shape: (num_layers, batch_size, hidden_size)
        # encoder_outputs shape: (batch_size, src_len, hidden_size)
        
        # Q: 解码器最后一层的隐藏状态
        query = hidden[-1].unsqueeze(1)  # -> (batch, 1, hidden)
        # K/V: 编码器的所有输出
        keys = encoder_outputs  # -> (batch, src_len, hidden)

        # energy shape: (batch, 1, src_len)
        energy = torch.bmm(query, keys.transpose(1, 2)) / self.scale_factor
        
        # attention_weights shape: (batch, src_len)
        return torch.softmax(energy, dim=2).squeeze(1)

class AttentionParams(nn.Module):
    """2: 带参数的注意力模块"""
    def __init__(self, hidden_size):
        super(AttentionParams, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden_last_layer = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat((hidden_last_layer, encoder_outputs), dim=2)))
        attention = torch.sum(self.v * energy, dim=2)
        
        return torch.softmax(attention, dim=1)

# ==============================================================================
#   4. 主流程
# ==============================================================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 统一创建编码器和伪数据 ---
    encoder = Encoder(src_vocab_size, hidden_size, num_layers).to(device)
    src = torch.randint(1, src_vocab_size, (batch_size, src_len)).to(device)
    trg = torch.randint(1, trg_vocab_size, (batch_size, trg_len)).to(device)

    # =========================================
    # 1: 无参数的基础 Attention
    # =========================================
    print("\n" + "="*20 + " 1: 无参数的基础 Attention " + "="*20)
    attention_simple = AttentionSimple(hidden_size).to(device)
    decoder_simple = DecoderWithAttention(trg_vocab_size, hidden_size, num_layers, attention_simple).to(device)
    model_simple = Seq2Seq(encoder, decoder_simple, device).to(device)
    
    model_simple.train()
    outputs_simple = model_simple(src, trg)
    print(f"模型结构:\n{model_simple}")
    print(f"\n训练模式输出张量形状: {outputs_simple.shape}")
    
    prediction_simple = model_simple.greedy_decode(src[0:1, :])
    print(f"推理的预测结果: {prediction_simple}")

    # =========================================
    # 2: 带参数的 Attention
    # =========================================
    print("\n" + "="*20 + " 2: 带参数的 Attention " + "="*20)
    attention_params = AttentionParams(hidden_size).to(device)
    decoder_params = DecoderWithAttention(trg_vocab_size, hidden_size, num_layers, attention_params).to(device)
    model_params = Seq2Seq(encoder, decoder_params, device).to(device)

    model_params.train()
    outputs_params = model_params(src, trg)
    print(f"模型结构:\n{model_params}")
    print(f"\n训练模式输出张量形状: {outputs_params.shape}")
    
    prediction_params = model_params.greedy_decode(src[0:1, :])
    print(f"推理的预测结果: {prediction_params}")
