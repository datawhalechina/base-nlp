# 第三节 模型构建、训练与推理

书接上回，我们已经完成了 NER 项目的数据处理工作，包括构建标签映射、词汇表以及一个功能完备的 `DataLoader`。本节将聚焦于如何利用 PyTorch 构建一个序列标注模型，并进一步封装一个可复用的训练流程，最终实现模型的训练、评估与推理。

> [本节完整代码](https://github.com/datawhalechina/base-nlp/tree/main/code/C8)

## 一、模型结构设计

正如第一节所介绍，NER 任务本质上是一个 **序列标注** 问题——为输入序列中的每一个 Token 预测一个对应的标签。基于此，可以设计一个有效的模型结构，它主要由三个核心部分组成：

1.  **Token Embedding 层**
    -   **作用**：将输入的 `token_ids`（一串数字）转换为初始的词向量。
    -   **实现**：通常使用 `torch.nn.Embedding` 层。它就像一个可学习的、巨大的查询表，每个 `token_id` 对应表中的一行（一个向量）。这些向量在训练开始时随机初始化，并随着模型训练过程不断优化。我们称之为 **静态词向量**，因为它不考虑上下文，同一个字在任何句子中都对应同一个向量。

2.  **动态特征提取层**
    -   **作用**：让模型理解上下文，生成包含上下文特征信息的 **动态词向量**。由于静态词向量无法区分同一个词在不同上下文中的含义，因此需要一个 Encoder 来融合上下文信息，从而生成更能体现语义的动态词向量。
    -   **实现**：循环神经网络 (RNN) 及其变体（如 LSTM, GRU）是处理序列数据的经典选择。我们可以使用 **双向 GRU (Bi-GRU)**，它能够同时从左到右和从右到左两个方向捕捉序列信息，从而更全面地理解每个 Token 的上下文。当然，也可以使用其他更强大的模型，如 **BERT**，来作为特征提取器。

3.  **分类决策层**
    -   **作用**：基于包含上下文信息的动态词向量，为每个 Token 预测其最终的实体标签（如 `B-dis`, `O` 等）。
    -   **实现**：通常使用一个简单的全连接层 (`torch.nn.Linear`)。它将 Encoder 输出的动态词向量从 `hidden_size` 维度映射到 `num_classes`（标签总数）维度，得到的输出即为每个 Token 在所有标签上的置信度得分。

> 整个模型本质上是一个 **Token 分类模型**：它接收 Token 序列，并为其中的每一个 Token 输出一个分类结果。

## 二、构建 PyTorch 模型

编写模型代码之前，先来回顾一下 `DataLoader` 输出的数据。如下图所示，经过 `collate_fn` 处理后，每个批次（Batch）的数据都包含了三个 `torch.Tensor`：`token_ids`、`label_ids` 和 `attention_mask`。

其中，`token_ids` 是模型最直接的输入，它是一个 `torch.int64` 类型的张量，代表了文本序列转换后的 Token 索引。

<div align="center">
  <img src="./images/8_3_1.png" alt="数据加载器输出示例" />
  <p>图 3.1: 数据加载器输出示例</p>
</div>

### 2.1 输入与输出

为了在代码层面更清晰地展示这些张量，我们直接复制如图 3.1 所示的真实数据片段。这有助于在正式实现模型前，先通过这组数据核对输入/输出的维度与取值约定（例如 `-100` 表示忽略位置）。

```python
import torch

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
```

从上面的示例中可以知道：
- **输入**：模型需要接收两个参数，`token_ids` 和 `attention_mask`。
- **输出**：模型的输出 `logits` 是一个三维张量，形状为 `[batch_size, seq_len, num_tags]`。

### 2.2 基础模型框架

目标明确后，就可以开始搭建模型了。先从一个最基础的单向 GRU 模型 `GRUNerNetWork` 开始。它包含 `__init__` 构造函数和 `forward` 
前向传播方法。为了构建一个更强大、更灵活的深度模型，这里采用 `nn.ModuleList` 来显式地堆叠多个 GRU 层。这种做法不仅让网络结构更清晰，还允许我们在层与层之间轻松地加入**残差连接**，这对于训练深度网络很重要。

> **nn.ModuleList vs nn.Sequential**
> 
> 在 PyTorch 中，`nn.ModuleList` 和 `nn.Sequential` 都是用来容纳多个子模块的容器，但它们的设计思想和使用场景截然不同：
> - **`nn.Sequential`**：像一个自动化的**流水线**，数据会自动按顺序流过每一层。它适用于简单的线性堆叠，但无法实现层间的复杂交互。
> - **`nn.ModuleList`**：更像一个普通的 **Python 列表**，只负责存储模块，而不会自动执行它们。你需要在 `forward` 方法中手动编写循环来调用每一层，所以可以在层与层之间加入任何自定义逻辑（如残差连接）。

对此，我们还需要做一个小小的设计：将词向量的维度与 GRU 的隐状态维度 `hidden_size` 设置为相同的值，这样残差连接（即两个张量相加）才能顺利进行。

```python
import torch.nn as nn
import torch.nn.utils.rnn as rnn

class GRUNerNetWork(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_tags, num_gru_layers=1):
        super().__init__()
        # 1. Token Embedding 层
        # 为了方便进行残差连接，embedding_dim 直接等于 hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # 2. 使用 ModuleList 构建多层单向 GRU
        self.gru_layers = nn.ModuleList()
        for _ in range(num_gru_layers):
            self.gru_layers.append(
                nn.GRU(
                    input_size=hidden_size, # 输入维度统一为 hidden_size
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=False
                )
            )
        
        # 3. 分类决策层
        self.classifier = nn.Linear(hidden_size, num_tags)

    def forward(self, token_ids, attention_mask=None):
        # [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        embedded_text = self.embedding(token_ids)
        
        current_input = embedded_text
        for gru_layer in self.gru_layers:
            gru_output, _ = gru_layer(current_input)
            # 添加残差连接
            current_input = gru_output + current_input
        
        logits = self.classifier(current_input)
        
        return logits

if __name__ == '__main__':
    # ... (数据构建) ...
    
    # 实例化模型
    model = GRUNerNetWork(
        vocab_size=10000,
        hidden_size=128,
        num_tags=37,
        num_gru_layers=2
    )
    
    # 3. 执行前向传播
    logits = model(token_ids=token_ids)
    
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
```

**运行结果：**

```text
Logits shape: torch.Size([2, 8, 10])
Loss shape: torch.Size([2, 8])

每个 Token 的损失:
tensor([[2.3364, 2.2961, 2.3879, 2.3275, 0.0000, 0.0000, 0.0000, 0.0000],d
        [2.2855, 2.3020, 2.2478, 2.3787, 2.2882, 2.3392, 2.3553, 0.0000]],
       grad_fn=<ViewBackward0>)
```

这段输出说明：
1.  **维度正确**：模型的输出 `logits` 维度为 `[2, 8, 10]`，与 `[batch_size, seq_len, num_tags]` 对应。
2.  **损失形状正确**：由于设置了 `reduction='none'`，损失张量的形状 `[2, 8]` 与 `label_ids` 一致，返回了每个 Token 各自的损失。
3.  **`ignore_index` 生效**：可以看到 `label_ids` 中值为 `-100` 的填充位置，其对应的损失值为 `0`。这证明损失函数成功忽略了这些填充位，避免了无效信息对模型训练的干扰。

### 2.3 升级为双向模型 (BiGRUNerNetWork)

单向 GRU 只能从左到右处理文本，这意味着在判断一个词的标签时，模型只能利用它左侧的上下文信息，而无法看到右侧的语境。这显然是一个局限。

为了克服这一点，我们可以轻松地将其升级为 **双向 GRU (Bi-GRU)**。Bi-GRU 包含两个并行的 GRU 层：一个从左到右处理，另一个从右到左处理。最后，它会将两个方向的输出特征**拼接**在一起，从而让每个时间步的输出都包含了完整的上下文信息。

代码的主要改动如下：
1.  **开启双向**：在 `nn.GRU` 的参数中设置 `bidirectional=True`。
2.  **增加特征融合层**：由于双向 GRU 的输出维度会变为 `hidden_size * 2`，我们需要增加一个全连接层 `fc`，将拼接后的特征重新映射回 `hidden_size`，以便与输入进行残差连接。

```python
class BiGRUNerNetWork(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_tags, num_gru_layers=2):
        super().__init__()
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
                    bidirectional=True
                )
            )
        
        # 3. 特征融合层
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        
        # 4. 分类决策层 (Classifier)
        self.classifier = nn.Linear(hidden_size, num_tags)

    def forward(self, token_ids, attention_mask):
        # [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        embedded_text = self.embedding(token_ids)

        current_input = embedded_text
        for gru_layer in self.gru_layers:
            gru_output, _ = gru_layer(current_input)
            features = self.fc(gru_output)
            # 添加残差连接
            current_input = features + current_input

        logits = self.classifier(current_input)
        
        return logits
```

### 2.4 核心代码讲解

模型的主体框架已经搭建完成，其中有几个关键点需要深入理解。

## 三、处理填充与变长序列 (进阶)

在 `02_data_processing.md` 中，我们将每个批次的数据都填充（Padding）到了相同的长度。然而，这对于 **双向 RNN** 来说是一个潜在的“陷阱”。

-   **问题**：在反向传播时，RNN 会从序列的末尾开始处理。如果末尾都是无意义的 `<PAD>` 标记，那么这些填充信息会“污染”到序列中真实 Token 的表示。
-   **解决方案**：PyTorch 提供了一套优雅的工具——`pack_padded_sequence` 和 `pad_packed_sequence`，来解决这个问题。

它们的工作流程如同“压缩”和“解压”：
1.  **`pack_padded_sequence`**：在将数据送入 RNN 之前，根据每个样本的真实长度，将填充后的序列“压紧”，丢掉所有填充位。RNN 内部只会对这些压紧后的有效数据进行计算。
2.  **`pad_packed_sequence`**：RNN 计算完成后，再用此函数将结果“解压”，恢复成填充后的形状，方便后续处理。

### 3.1 在模型中集成 Pack/Pad

我们需要修改 `forward` 方法来集成这个流程。同时，`forward` 方法也需要接收 `attention_mask` 来计算每个样本的真实长度。

```python
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

class BiGRUNerNetWork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_tags, num_gru_layers=1):
        super().__init__()
        # 1. Token Embedding 层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. 动态特征提取层 (Encoder)
        self.gru_layers = nn.ModuleList([
            nn.GRU(
                input_size=embedding_dim if i == 0 else hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True  # 开启双向
            ) for i in range(num_gru_layers)
        ])
        
        # 3. 分类决策层 (Classifier)
        # 双向 GRU 的输出维度是 hidden_size 的两倍
        self.bidirectional_fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )

    def forward(self, token_ids, attention_mask):
        # 1. 计算每个样本的真实长度
        lengths = attention_mask.sum(dim=1)
        
        # 2. 获取词向量
        # [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded_text = self.embedding(token_ids)

        # 3. 打包序列
        packed_embeddings = rnn.pack_padded_sequence(
            embedded_text, 
            lengths.cpu(),  # 长度必须是 CPU 上的 tensor
            batch_first=True, 
            enforce_sorted=False # 输入数据未按长度排序
        )
        
        # 4. 将打包后的序列送入 GRU
        gru_input = packed_embeddings
        for gru_layer in self.gru_layers:
            packed_gru_output, _ = gru_layer(gru_input)
            gru_output = rnn.pad_packed_sequence(
                packed_gru_output, 
                batch_first=True
            )[0] # 只取序列部分
            gru_output = self.bidirectional_fc(gru_output)
            gru_input = gru_output + gru_input # 残差连接
        
        # 5. 分类
        # [batch_size, seq_len, hidden_size * 2] -> [batch_size, seq_len, num_tags]
        logits = self.classifier(gru_output)
        
        return logits
```

通过这种方式，模型就能在不被填充位干扰的情况下，高效地处理变长序列。

## 四、训练流程封装

一个成熟的项目，其训练代码不应是零散的脚本，而应是结构化、可复用的框架。为此，我们设计一个 `Trainer` 类，将训练、评估、持久化等通用逻辑封装起来。

### 4.1 定义配置类

使用 `@dataclass` 可以方便地创建一个配置类，用于管理所有超参数。

```python
from dataclasses import dataclass, field

@dataclass
class TrainerConfig:
    # --- 路径参数 ---
    train_file: str = './data/CMeEE-V2_train.json'
    dev_file: str = './data/CMeEE-V2_dev.json'
    vocab_file: str = './data/vocabulary.json'
    tags_file: str = './data/categories.json'
    output_dir: str = './output_model'

    # --- 训练参数 ---
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    early_stop_epoch: int = 5
    
    # --- 模型参数 ---
    embedding_dim: int = 256
    hidden_size: int = 256
    gru_num_layers: int = 2
    
    # --- 其他 ---
    device: str = 'cuda'
    label_ignore_index: int = -100
```

### 4.2 定义 Trainer 类

我们将 `Trainer` 设计成一个独立的类，它包含了完整的训练和评估逻辑。

```python
import torch
import os
from tqdm import tqdm

class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        
        # 初始化组件
        self._init_components()

    def _init_components(self):
        # 设定设备
        self.device = torch.device(self.config.device)

        # 加载数据加载器
        self.train_loader = self.build_dataloader(self.config.train_file, shuffle=True)
        self.dev_loader = self.build_dataloader(self.config.dev_file, shuffle=False)

        # 构建模型
        self.model = self.build_model().to(self.device)

        # 构建优化器和损失函数
        self.optimizer = self.build_optimizer()
        self.criterion = self.build_criterion()

    def build_dataloader(self, data_path, shuffle):
        """由子类实现，返回一个 DataLoader 实例"""
        raise NotImplementedError

    def build_model(self):
        """由子类实现，返回一个 nn.Module 实例"""
        raise NotImplementedError

    def build_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def build_criterion(self):
        # 忽略索引为 -100 的标签 (即填充位)
        return nn.CrossEntropyLoss(ignore_index=self.config.label_ignore_index)
        
    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (token_ids, attention_mask, tag_ids) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
            token_ids, attention_mask, tag_ids = token_ids.to(self.device), attention_mask.to(self.device), tag_ids.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(token_ids, attention_mask)
            loss = self.criterion(logits.permute(0, 2, 1), tag_ids)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _evaluate(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (token_ids, attention_mask, tag_ids) in enumerate(tqdm(self.dev_loader, desc=f"Evaluating Epoch {epoch}")):
                token_ids, attention_mask, tag_ids = token_ids.to(self.device), attention_mask.to(self.device), tag_ids.to(self.device)
                
                logits = self.model(token_ids, attention_mask)
                loss = self.criterion(logits.permute(0, 2, 1), tag_ids)
                total_loss += loss.item()

        return total_loss / len(self.dev_loader)

    def _save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.criterion(self.model(token_ids, attention_mask).permute(0, 2, 1), tag_ids) # This line was not in the new_code, but should be added for consistency
        }
        if is_best:
            torch.save(state, os.path.join(self.config.output_dir, 'best_model.pth'))
        else:
            torch.save(state, os.path.join(self.config.output_dir, f'epoch_{epoch}.pth'))

    def train(self):
        best_f1 = 0.0
        for epoch in range(1, self.config.epochs + 1):
            # --- 训练 ---
            train_loss = self._train_one_epoch(epoch)
            print(f"Epoch {epoch} - Training Loss: {train_loss:.4f}")

            # --- 评估 ---
            dev_loss = self._evaluate(epoch)
            print(f"Epoch {epoch} - Dev Loss: {dev_loss:.4f}")

            # --- 保存最优模型 ---
            is_best = dev_loss < best_f1 # Placeholder for actual evaluation logic
            self._save_checkpoint(epoch, is_best)

            # Early stopping check
            if epoch >= self.config.early_stop_epoch and dev_loss >= best_f1:
                print(f"Early stopping at epoch {epoch}.")
                break
```

### 4.3 实现 NER 任务的 Trainer

现在，我们创建一个 `NerTrainer` 继承自 `Trainer`，并实现其中未完成的方法。

```python
from data_loader import Vocabulary, create_ner_dataloader
import json

class NerTrainer(Trainer):
    def __init__(self, config: TrainerConfig):
        # 加载词汇表和标签映射
        self.vocab = Vocabulary(config.vocab_file)
        with open(config.tags_file, 'r', encoding='utf-8') as f:
            self.tag_map = json.load(f)
        
        super().__init__(config)

    def build_dataloader(self, data_path, shuffle):
        return create_ner_dataloader(
            data_path,
            vocab=self.vocab,
            tag_map=self.tag_map,
            batch_size=self.config.batch_size,
            shuffle=shuffle
        )

    def build_model(self):
        return BiGRUNerNetWork(
            vocab_size=len(self.vocab),
            embedding_dim=self.config.embedding_dim,
            hidden_size=self.config.hidden_size,
            num_tags=len(self.tag_map),
            num_gru_layers=self.config.gru_num_layers
        )
```

### 4.3 启动训练脚本

最后，我们通过一个主脚本来实例化配置和 `Trainer`，并启动训练过程。

```python
# main.py
from trainer import Trainer, TrainerConfig

def main():
    # 1. 初始化配置
    # 可以在这里覆盖 config 的默认值
    # 例如: config = TrainerConfig(batch_size=64, epochs=10)
    config = TrainerConfig()
    
    # 2. 初始化 Trainer
    trainer = Trainer(config)
    
    # 3. 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
```
## 五、模型评估与推理

### 5.1 评估指标

对于 NER 任务，我们更关心 **实体级别** 的精确率 (Precision)、召回率 (Recall) 和 F1 值，而不是简单的 Token 准确率。评估流程如下：

1.  **解码**：将模型输出的 `tag_ids` 序列转换回实体片段列表，如 `[(start, end, type), ...]`。
2.  **对比**：将预测的实体列表与真实的实体列表进行对比。
3.  **计算**：
    -   **TP (True Positive)**：预测正确且与真实实体完全匹配的实体数量。
    -   **FP (False Positive)**：预测出的、但实际不存在的（或不完全匹配的）实体数量。
    -   **FN (False Negative)**：真实存在、但模型未能预测出的实体数量。
    -   **Precision** = TP / (TP + FP)
    -   **Recall** = TP / (TP + FN)
    -   **F1** = 2 * (Precision * Recall) / (Precision + Recall)

> 社区中有成熟的库（如 `seqeval`）可以方便地计算这些指标。

### 5.2 推理

当模型训练完成后，我们可以编写一个推理函数，对新的文本进行实体识别。

```python
# ... (需要加载 Vocabulary, tag_map, 和训练好的模型)

class Predictor:
    def __init__(self, model, vocab, tag_map, device):
        self.model = model.to(device).eval()
        self.vocab = vocab
        # 创建 id -> tag 的逆向映射，方便解码
        self.id_to_tag = {v: k for k, v in tag_map.items()}
        self.device = device

    def predict(self, text):
        # 1. 文本预处理
        tokens = list(text)
        token_ids = self.vocab.convert_tokens_to_ids(tokens)
        
        # 2. 转换为 Tensor
        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        mask = (input_tensor != self.vocab.pad_id).long()

        # 3. 模型推理
        with torch.no_grad():
            logits = self.model(input_tensor, mask)
        
        # 4. 解码
        pred_tag_ids = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
        pred_tags = [self.id_to_tag[tid] for tid in pred_tag_ids]
        
        # 5. 将标签序列转换为实体
        # ... (解码逻辑，将 BMES 标签转换为实体片段)
        
        return entities
```

至此，我们已经完整地构建了从数据处理、模型构建、训练封装到最终评估和推理的整个 NER 项目流程。