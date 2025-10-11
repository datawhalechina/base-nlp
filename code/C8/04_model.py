import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

class BiGRUNerNetWork(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_tags, num_gru_layers=1):
        super(BiGRUNerNetWork).__init__()
        # 1. Token Embedding 层
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # 2. 使用 ModuleList 构建多层双向 GRU
        self.gru_layers = nn.ModuleList()
        for _ in range(num_gru_layers):
            self.gru_layers.append(
                nn.GRU(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True  # 开启双向
                )
            )

        # 3. 特征融合层
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

        # 4. 分类决策层 (Classifier)
        self.classifier = nn.Linear(hidden_size, num_tags)

    def forward(self, token_ids, attention_mask):
        # 1. 计算真实长度
        lengths = attention_mask.sum(dim=1).cpu()

        # 2. 获取词向量
        embedded_text = self.embedding(token_ids)

        # 3. 打包序列
        current_packed_input = rnn.pack_padded_sequence(
            embedded_text, lengths, batch_first=True, enforce_sorted=False
        )

        # 4. 循环通过 GRU 层
        for gru_layer in self.gru_layers:
            # GRU 输出 (packed)
            packed_output, _ = gru_layer(current_packed_input)

            # 解包以进行后续操作，并指定 total_length
            output, _ = rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=token_ids.shape[1]
            )

            # 特征融合
            features = self.fc(output)

            # 残差连接
            # 同样需要解包上一层的输入
            input_padded, _ = rnn.pad_packed_sequence(
                current_packed_input, batch_first=True, total_length=token_ids.shape[1]
            )
            current_input = features + input_padded

            # 重新打包作为下一层的输入
            current_packed_input = rnn.pack_padded_sequence(
                current_input, lengths, batch_first=True, enforce_sorted=False
            )

        # 5. 解包最终输出用于分类
        final_output, _ = rnn.pad_packed_sequence(
            current_packed_input, batch_first=True, total_length=token_ids.shape[1]
        )

        # 6. 分类
        logits = self.classifier(final_output)

        return logits


if __name__ == '__main__':

    token_ids = torch.tensor([
        [210,   18,  871, 147,   0,   0,   0,   0],
        [922, 2962,  842, 210,  18, 871, 147,   0]
    ], dtype=torch.int64)

    # attention_mask 标记哪些是真实 token (1) 哪些是填充 (0)
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0]
    ], dtype=torch.int64)

    label_ids = torch.tensor([
        [0, 0, 0, 0, -100, -100, -100, -100],
        [0, 0, 0, 0,    0,    0,    0, -100]
    ], dtype=torch.int64)

    # 实例化模型
    model = BiGRUNerNetWork(
        vocab_size=10000,
        hidden_size=128,
        num_tags=37,
        num_gru_layers=2
    )

    # 3. 执行前向传播
    logits = model(token_ids=token_ids, attention_mask=attention_mask)

    # 4. 构造损失函数
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    # 5. 计算损失
    # CrossEntropyLoss 要求类别维度在前，所以需要交换最后两个维度
    # [batch, seq_len, num_tags] -> [batch, num_tags, seq_len]
    permuted_logits = torch.permute(logits, dims=(0, 2, 1))
    loss = loss_fn(permuted_logits, label_ids)

    # 6. 打印结果
    print(f"Logits shape: {logits.shape}")
    print(f"Loss shape: {loss.shape}")
    print("\n每个 Token 的损失:")
    print(loss)