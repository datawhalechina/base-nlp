import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn


class BiGRUForNer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_tags, num_gru_layers=1):
        """
        Args:
            vocab_size (int): 词汇表大小
            embedding_dim (int): 词向量维度, 与 hidden_size 保持一致以进行残差连接
            hidden_size (int): GRU 隐藏层维度
            num_tags (int): 标签数量
            num_gru_layers (int): GRU 层数
        """
        super().__init__()
        # 1. Token Embedding 层
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # 2. 动态特征提取层 (Encoder)
        self.gru_layers = nn.ModuleList([
            nn.GRU(
                input_size=embedding_dim if i == 0 else hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            ) for i in range(num_gru_layers)
        ])

        # 3. 用于融合双向 GRU 输出的线性层
        self.bidirectional_fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )
        
        # 4. 分类决策层 (Classifier)
        self.classifier = nn.Linear(hidden_size, num_tags)

    def forward(self, token_ids, attention_mask):
        """
        Args:
            token_ids (torch.Tensor): a tensor of shape [batch_size, seq_len]
            attention_mask (torch.Tensor): a tensor of shape [batch_size, seq_len]
        
        Returns:
            torch.Tensor: a tensor of shape [batch_size, seq_len, num_tags]
        """
        # [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        gru_input = self.embedding(token_ids)

        # 计算真实长度用于打包
        lengths = attention_mask.sum(dim=1).cpu()

        for gru_layer in self.gru_layers:
            # 打包序列，以处理变长输入
            packed_input = rnn.pack_padded_sequence(
                gru_input,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )
            
            # GRU 前向传播
            packed_output, _ = gru_layer(packed_input)
            
            # 解包序列
            gru_output, _ = rnn.pad_packed_sequence(packed_output, batch_first=True)
            
            # 融合双向输出: [batch_size, seq_len, hidden_size * 2] -> [batch_size, seq_len, hidden_size]
            gru_output = self.bidirectional_fc(gru_output)
            
            # 应用残差连接
            gru_input = gru_output + gru_input

        # 最终分类
        logits = self.classifier(gru_input)
        return logits


if __name__ == '__main__':
    # 定义模型超参数
    VOCAB_SIZE = 2000
    EMBEDDING_DIM = 256 # 与 HIDDEN_SIZE 保持一致
    HIDDEN_SIZE = 256
    NUM_TAGS = 37
    NUM_GRU_LAYERS = 3

    # 实例化模型
    model = BiGRUForNer(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_TAGS, NUM_GRU_LAYERS)

    # 创建伪输入数据
    dummy_token_ids = torch.randint(0, VOCAB_SIZE, (4, 20))
    dummy_attention_mask = torch.ones_like(dummy_token_ids)
    dummy_attention_mask[0, -5:] = 0
    dummy_attention_mask[1, -10:] = 0

    # 执行前向传播
    logits = model(dummy_token_ids, dummy_attention_mask)

    # 打印输出的形状
    print("模型输出的 Logits 形状:", logits.shape)
    # 预期输出: torch.Size([4, 20, NUM_TAGS])
    assert logits.shape == (4, 20, NUM_TAGS)
    print("模型测试通过!")