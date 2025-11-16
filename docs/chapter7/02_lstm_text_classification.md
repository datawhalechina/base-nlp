# 第二节 基于 LSTM 的文本分类

在上一节，我们实现了一个基于全连接层的文本分类模型。该模型虽然简单有效，但它的核心是将所有词元的特征向量进行平均池化，这本质上是一种“词袋”模型。这种方法的**一个显著局限是它忽略了文本中词语的顺序**，而语序在多数 NLP 任务中是很重要的。

那么，对于文本分类任务，捕捉序列信息是否总能带来性能提升呢？为了验证这一点，我们自然会想到循环神经网络（RNN）及其变体，如LSTM。在**第三章第二节**中我们已经学习了 LSTM 的原理。理论上，它能够通过处理序列信息来捕捉更丰富的语义。本节将进行一次实验，我们将上一节的全连接模型改造为基于LSTM的模型，来**探索**在本新闻分类任务上，序列建模是否会比简单的词袋模型更有效。

## 一、从“词袋”到序列建模

先回顾一下基线模型的主要操作：

1.  **词嵌入**：将输入的 `token_ids` (`[batch_size, seq_len]`) 转换为词向量 `embedded` (`[batch_size, seq_len, embed_dim]`)。
2.  **特征提取**：通过几层全连接网络，将每个词向量独立地映射到更高维的特征空间，得到 `token_features` (`[batch_size, seq_len, hidden_dim]`)。
3.  **掩码平均池化**：为了处理变长序列，将所有 `token_features` 沿 `seq_len` 维度进行求和，再除以真实长度，得到一个代表整句话的向量 `pooled_features` (`[batch_size, hidden_dim]`)。
4.  **分类**：将 `pooled_features` 输入最后的分类层，得到最终预测。

这个流程的瓶颈在第三步。**平均池化**操作将序列信息压缩成一个向量，这可能导致词序信息的丢失。

与之相对，LSTM 网络通过其内部的循环结构和门控机制，能够逐个处理序列中的词元，并持续更新一个内部状态（记忆）。这个状态在每个时间步都会编码从序列开始到当前位置的所有信息。因此，当 LSTM 处理完整个序列后，它最终的隐藏状态理论上包含了对整个句子序列更丰富的语义表示，这**有可能**比简单的词向量平均更能捕捉句子的深层含义。

## 二、代码修改实践

将基线模型改造为 LSTM 模型，主要涉及这三个部分的修改：数据处理、模型结构和推理逻辑。

> [本节完整代码](https://github.com/datawhalechina/base-nlp/blob/main/code/C7/02_lstm_text_classification.ipynb)

### 2.1 改造 `collate_fn` 以提供序列长度

为了让 LSTM 能够高效地处理被填充（Padding）过的变长序列，需要使用 `torch.nn.utils.rnn.pack_padded_sequence` 函数。该函数要求在输入批次中明确提供每个样本在填充前的**真实长度**。所以，我们应该修改 `collate_fn` 函数，让它在返回 `token_ids` 和 `labels` 的同时，也返回一个包含该批次中每个序列真实长度的张量 `lengths`。

```python
def collate_fn(batch):
    max_batch_len = max(len(item["token_ids"]) for item in batch)
    
    batch_token_ids, batch_labels, batch_lengths = [], [], []

    for item in batch:
        token_ids = item["token_ids"]
        # 新增：记录真实长度
        lengths = len(token_ids)
        padding_len = max_batch_len - lengths
        
        padded_ids = token_ids + [0] * padding_len
        batch_token_ids.append(padded_ids)
        batch_labels.append(item["label"])
        # 新增：将长度加入列表
        batch_lengths.append(lengths)
        
    return {
        "token_ids": torch.tensor(batch_token_ids, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
        # 新增：返回长度张量
        "lengths": torch.tensor(batch_lengths, dtype=torch.long),
    }
```

### 2.2 构建 `TextClassifierLSTM` 模型

这是本次优化的主要内容。我们将原来的 `TextClassifier` 替换为一个新的 `TextClassifierLSTM` 模型。

```python
class TextClassifierLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, 
                 n_layers=1, dropout=0.3, bidirectional=False):
        super(TextClassifierLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True  # 关键参数：输入和输出张量的维度为 (batch, seq, feature)
        )
        
        num_directions = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_dim * num_directions, num_classes)
        
    def forward(self, token_ids, lengths):
        embedded = self.embedding(token_ids)
        
        # 1. 打包序列
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, 
            lengths.cpu(),  # 长度必须在CPU上
            batch_first=True, 
            enforce_sorted=False
        )
        
        # 2. LSTM 前向传播
        #    hidden 和 cell 的形状: [n_layers * num_directions, batch_size, hidden_dim]
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # 3. 提取最终隐藏状态用于分类
        if self.lstm.bidirectional:
            # 拼接最后一个时间步的前向和后向的隐藏状态
            # hidden[-2,:,:] 是前向的最后一个隐藏状态
            # hidden[-1,:,:] 是后向的最后一个隐藏状态
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            # 只取最后一层的最后一个隐藏状态
            hidden = hidden[-1,:,:]
            
        # 4. 分类
        logits = self.classifier(hidden)
        return logits
```

**模型解析**:
1.  **`__init__`**:
    *   除了词嵌入层 `nn.Embedding` 和分类层 `nn.Linear`，核心是一个 `nn.LSTM` 层。
    *   增加几个 LSTM 相关的超参数：`n_layers` (LSTM层数), `dropout` (层间丢弃率), `bidirectional` (是否使用双向LSTM)。
    *   `batch_first=True` 是一个重要的设置，它让 LSTM 接受 `[batch_size, seq_len, feature_dim]` 形状的输入，与 `DataLoader` 的输出保持一致，简化了代码。
    *   分类层的输入维度需要根据 `bidirectional` 的值来动态确定。如果是双向的，隐藏层维度会加倍。
    *   在 PyTorch 的 `nn.LSTM` 中，`dropout` 只在 `n_layers > 1` 时于层间生效；当仅 1 层时该参数不会起作用。若使用单层 LSTM，可将 `dropout` 设为 `0.0`（或保留任意值，效果一致），避免造成误解。

2.  **`forward`**:
    *   `forward` 函数现在额外接收 `lengths` 参数。
    *   **打包 (Packing)**：`pack_padded_sequence` 是处理填充序列的关键。它会将一个填充过的批次数据（例如，多个句子被填充到相同长度）压缩成一个更紧凑的表示，LSTM 只需对真实的、非填充部分进行计算，大大提高了效率和准确性。
    *   **最终状态提取**：LSTM 的输出 `hidden` 张量包含了所有层在最后一个时间步的隐藏状态。我们通常取最后一层（对于单向 LSTM 是 `hidden[-1,:,:]`）作为整个序列的语义表示。如果是双向 LSTM，则需要拼接前向和后向的最终隐藏状态。
    *   最后，将这个代表序列的 `hidden` 向量送入分类器。

### 2.3 调整 `Trainer` 和 `Predictor`

由于模型 `forward` 函数的输入签名发生了变化，我们需要对 `Trainer` 和 `Predictor` 进行微调，以确保 `lengths` 张量被正确传递。

**1. `Trainer` 修改**:
在 `_run_epoch` 和 `_evaluate` 方法中，从 `batch` 字典中取出 `lengths`，并将其传递给 `self.model`。

```python
# 在 Trainer._run_epoch 方法中
...
token_ids = batch["token_ids"].to(self.device)
labels = batch["labels"].to(self.device)
lengths = batch["lengths"]

outputs = self.model(token_ids, lengths)
...
```
（`_evaluate` 方法同理）

**2. `Predictor` 修改**:
`Predictor` 在处理单个文本时，也需要模拟批处理的逻辑：对文本分块后，手动计算每个块的长度，并进行填充，然后将 `chunk_tensors` 和 `length_tensors` 一同传入模型。

```python
# 在 Predictor.predict 方法中
...
# (文本分块逻辑不变) ...
chunks = [...] 

# 手动计算长度并进行填充
chunk_lengths = [len(c) for c in chunks]
max_chunk_len = max(chunk_lengths) if chunk_lengths else 0

padded_chunks = []
for chunk in chunks:
    padding_len = max_chunk_len - len(chunk)
    padded_chunks.append(chunk + [0] * padding_len)

if not padded_chunks:
    return "无法预测（文本过短）"

chunk_tensors = torch.tensor(padded_chunks, dtype=torch.long).to(self.device)
length_tensors = torch.tensor(chunk_lengths, dtype=torch.long) # 长度在CPU上

with torch.no_grad():
    outputs = self.model(chunk_tensors, length_tensors)
    preds = torch.argmax(outputs, dim=1)
...
```

### 2.4 更新训练入口代码

最后一步，更新用于启动训练的单元格。我们需要：
1.  为 LSTM 添加新的超参数（`n_layers`, `dropout`, `bidirectional`）。
2.  实例化新的 `TextClassifierLSTM` 模型。
3.  （可选）为新的模型实验设置一个独立的输出目录，如 `"output_lstm"`。

```python
import torch

hparams = {
    "vocab_size": len(tokenizer),
    "embed_dim": 128,
    "hidden_dim": 256,
    "num_classes": len(train_dataset_raw.target_names),
    "n_layers": 2,          # 新增
    "dropout": 0,         # 新增
    "bidirectional": True,  # 新增
    "epochs": 20,
    "learning_rate": 0.001,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "output_lstm" # 修改输出目录
}

# 实例化新模型
model = TextClassifierLSTM(
    vocab_size=hparams["vocab_size"], 
    embed_dim=hparams["embed_dim"], 
    hidden_dim=hparams["hidden_dim"], 
    num_classes=hparams["num_classes"],
    n_layers=hparams["n_layers"],
    dropout=hparams["dropout"],
    bidirectional=hparams["bidirectional"]
).to(hparams["device"])

# (后续代码不变)
```

完成以上修改后，重新运行整个 Notebook，即可训练一个能够处理序列信息的 LSTM 模型。接下来，我们来对比它与基线模型的性能，并分析序列建模在本次任务中的实际效果。

### 2.5 实验结果与分析

在分别运行了基线的全连接模型和我们新构建的LSTM模型后（均未加正则化策略），我们得到了如下的性能数据：

-   **全连接模型 (基线)**：最终验证集最佳准确率约为 **0.8469**。
-   **LSTM 模型**：最终验证集最佳准确率约为 **0.8143**。

<p align="center">
  <img src="./images/7_2_1.png" width="70%" alt="LSTM 模型训练过程指标" />
  <br />
  <em>图 7-4 LSTM 模型训练损失与验证集准确率变化曲线</em>
</p>

**结果分析**:

显然结果并不符合我们的预期，理论上更能捕捉序列信息的 LSTM 模型，在本次新闻分类任务上的表现反而**劣于**简单的全连接模型。这个发现说明**模型的复杂性与任务的实际需求应该匹配**。值得一提的是，本次对比实验并未严格控制固定随机数种子，所以每次运行的结果会存在细微的差别。然而，一个稳定的现象是，引入序列建模的 LSTM 并未带来性能提升，其结果反而总是比简单的全连接模型低 **~2%** 左右，足以让我们得出以下结论。

出现这种结果的可能原因有两点：

1.  **任务对语序不敏感**：新闻分类任务在很大程度上依赖于**关键词**。例如，看到 "Jesus"、"God" 很可能属于宗教类；看到 "Graphics"、"Monitor" 很可能属于计算机图形类。全连接模型本质上是一个高效的“词袋”模型，非常擅长捕捉这类强特征词的存在与否。对于这个特定任务，**“有哪些词”远比“这些词的顺序”更重要**。LSTM 为学习语序付出的额外努力，在这里并没有转化为实际的性能优势。

2.  **模型复杂性与过拟合**：LSTM 模型比简单的全连接网络复杂得多，拥有更多的参数。虽然它理论上能学习到更复杂的模式，但也**更容易在数据量不够大的情况下陷入过拟合**。从训练日志中可以看到，普通 LSTM 的训练损失已经非常低，但验证集准确率却不高，这是过拟合症状。模型过于“记住”了训练集中的特定句子结构，而没有学到普适的规律。

## 三、过拟合解决方案与效果对比

基础 LSTM 模型效果不佳的一个可能原因是 **过拟合**。在第一节的末尾，我们介绍了三种简单有效的正则化方法：**提前停止**、**随机Token遮盖**和**Dropout**。

现在，我们将这三种方法组合应用到新的 LSTM 模型上，观察它们的综合效果。

### 3.1 随机Token遮盖

这是一种数据增强技术。我们在 `TextClassificationDataset` 的基础上创建一个子类，在 `__getitem__` 方法中，对训练样本的 `token_ids` 进行随机替换。具体来说，以一定概率（例如10%）将部分词元替换为 `<UNK>` 对应的 ID。

这迫使模型不能过度依赖个别特征词，而是要从更广的上下文中学习语义，从而增强泛化能力。

```python
import random

class TextClassificationDatasetWithMasking(TextClassificationDataset):
    def __init__(self, texts, labels, tokenizer, max_len=128, is_train=False, mask_prob=0.1):
        super().__init__(texts, labels, tokenizer, max_len)
        self.is_train = is_train
        self.mask_prob = mask_prob
        self.unk_token_id = tokenizer.token_to_id.get("<UNK>", 1)

    def __getitem__(self, idx):
        # 关键：创建副本，避免修改原始数据
        item = super().__getitem__(idx).copy()
        
        if self.is_train:
            token_ids = item['token_ids']
            masked_token_ids = []
            for token_id in token_ids:
                # 不遮盖PAD (ID=0)
                if token_id != 0 and random.random() < self.mask_prob:
                    masked_token_ids.append(self.unk_token_id)
                else:
                    masked_token_ids.append(token_id)
            item['token_ids'] = masked_token_ids
            
        return item
```

> 在 `TextClassificationDatasetWithMasking` 的 `__getitem__` 方法中，有一个非常关键的细节，`item = super().__getitem__(idx).copy()`。必须使用 `.copy()` 方法来创建数据的副本。
> 
> 如果没有 `.copy()`，`__getitem__` 中的修改将会永久地改变原始数据集。这会导致在第二个训练周期（Epoch）时，模型看到的是已经被第一次随机遮盖过的数据，并在此基础上进行二次遮盖，如此循环往复，最终导致有效信息完全丢失。数据增强必须保证每一轮都是在干净的原始数据上进行的独立操作。

### 3.2 提前停止 (Early Stopping)

提前停止是一种简单而高效的正则化策略。其核心思想是在训练过程中持续监控模型在验证集上的性能。如果验证集准确率（或损失）连续 `N` 个轮次（`N` 称为“耐心值” `patience`）没有超过历史最佳水平，就认为模型已经达到了最佳点或开始过拟合，此时应提前终止训练。

我们在 `Trainer` 类的基础上创建一个子类，重写 `train` 方法以实现该逻辑。

```python
import os
import json

class TrainerWithEarlyStopping(Trainer):
    def __init__(self, model, optimizer, criterion, train_loader, valid_loader, device, output_dir=".", patience=3):
        super().__init__(model, optimizer, criterion, train_loader, valid_loader, device, output_dir)
        self.patience = patience
        self.epochs_no_improve = 0

    def train(self, epochs, tokenizer, label_map):
        for epoch in range(epochs):
            avg_loss = self._run_epoch(epoch)
            val_accuracy = self._evaluate(epoch)
            
            print(f"Epoch {epoch+1}/{epochs} | 训练损失: {avg_loss:.4f} | 验证集准确率: {val_accuracy:.4f}")
            
            current_best = self.best_accuracy
            self._save_checkpoint(epoch, val_accuracy)
            
            if self.best_accuracy > current_best:
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            
            if self.epochs_no_improve >= self.patience:
                print(f"\n提前停止于 Epoch {epoch+1}，因为验证集准确率连续 {self.patience} 轮未提升。")
                break
        
        print("\n训练完成！")
        # ... (保存词典和标签映射)
```

### 3.3 实验与对比

最后，我们将所有正则化策略整合起来，实例化相应的数据集、模型和训练器，并启动训练。

```python
# 1. 创建应用了随机遮盖的数据集
train_dataset_reg = TextClassificationDatasetWithMasking(
    ..., is_train=True, mask_prob=0.1
)
train_loader_reg = DataLoader(train_dataset_reg, ...)

# 2. 实例化模型，并启用 Dropout
model_reg = TextClassifierLSTM(
    ...,
    dropout=0.3, # 启用 Dropout
    ...
).to(device)

# 3. 使用带提前停止功能的训练器
trainer_reg = TrainerWithEarlyStopping(
    model_reg,
    ...,
    patience=3
)

# 启动训练
trainer_reg.train(...)
```

完成训练后，可以通过比较两个实验的输出日志，来分析正则化带来的效果：
-   **训练是否提前停止？** 如果是，说明模型可能在更早的阶段就已收敛。
-   **最终验证集准确率**：对比 `output_lstm` 和 `output_lstm_regularized` 中 `best_model.pth` 对应的验证集准确率，正则化版本是否取得了更好的泛化性能？
-   **训练损失与验证准确率曲线**：观察两个实验的日志，正则化版本的验证集准确率曲线是否更平滑，或者与训练损失的差距是否更小？这些都是过拟合得到缓解的迹象。

### 3.4 最终效果分析

在应用了这三种策略后，我们的 LSTM 模型取得了约 **0.8415** 的最佳验证集准确率。从下面的训练日志中可以看到，模型在第16轮达到了最佳性能，并在第19轮成功触发了“提前停止”策略，避免了不必要的训练和潜在的过拟合。

```bash
Epoch 16 [训练中]: 100%|██████████| 224/224 [00:06<00:00, 33.60it/s]
Epoch 16 [评估中]: 100%|██████████| 169/169 [00:01<00:00, 97.13it/s]
Epoch 16/20 | 训练损失: 0.0206 | 验证集准确率: 0.8415
新最佳模型已保存! Epoch: 16, 验证集准确率: 0.8415
Epoch 17 [训练中]: 100%|██████████| 224/224 [00:06<00:00, 32.80it/s]
Epoch 17 [评估中]: 100%|██████████| 169/169 [00:01<00:00, 94.98it/s]
Epoch 17/20 | 训练损失: 0.0176 | 验证集准确率: 0.8243
Epoch 18 [训练中]: 100%|██████████| 224/224 [00:07<00:00, 31.27it/s]
Epoch 18 [评估中]: 100%|██████████| 169/169 [00:01<00:00, 96.61it/s]
Epoch 18/20 | 训练损失: 0.0172 | 验证集准确率: 0.8175
Epoch 19 [训练中]: 100%|██████████| 224/224 [00:06<00:00, 33.10it/s]
Epoch 19 [评估中]: 100%|██████████| 169/169 [00:01<00:00, 95.57it/s]
Epoch 19/20 | 训练损失: 0.0136 | 验证集准确率: 0.8066

提前停止于 Epoch 19，因为验证集准确率连续 3 轮未提升。

训练完成！
词典 (output_lstm_regularized\vocab.json) 和标签映射 (output_lstm_regularized\label_map.json) 已保存。
```

<p align="center">
  <img src="./images/7_2_2.png" width="70%" alt="正则化LSTM模型训练过程指标" />
  <br />
  <em>图 7-5 正则化 LSTM 模型训练损失与验证集准确率变化曲线</em>
</p>

这个结果展示了正则化策略的价值：

-   **相比于无正则化的LSTM（~0.8143）**：性能得到了**明显提升**。这证明我们之前的判断是正确的——基础 LSTM 模型的一个主要问题就是过拟合。通过数据增强、提前停止和层间Dropout的组合，有效地抑制了模型对训练数据的“死记硬背”，使它学习到了更具泛化能力的模式。
    -   **随机Token遮盖**强迫模型不能过度依赖训练集中少数几个“明星”关键词（例如特定作者），而是要学会识别更广泛、更多样化的关键词组合来做出判断，从而提升模型的鲁棒性和泛化能力。
    -   **提前停止**则像一个“安全阀”，在模型性能达到巅峰并即将开始下滑（过拟合）的时刻及时终止了训练，锁定了最佳的模型状态。
    -   **Dropout**是在 LSTM 层之间随机“关闭”一些神经元，破坏了神经元之间可能形成的“共适应”关系，增强了模型的独立特征学习能力。

-   **相比于全连接模型（~0.8469）**：经过正则化后，LSTM模型的性能已经非常接近，但仍然略逊于更简单的基线模型。再次证明了对于这个特定的、以关键词为驱动的新闻分类任务，一个高效的“词袋”模型已经足够强大。试图用更复杂的序列模型来捕捉此处并不关键的语序信息，即使在组合了多种正则化策略后，也难以带来超越性的优势。

这个系列的实验为我们提供了一个宝贵的实践经验，同时也印证了著名的“奥卡姆剃刀原理”——**如无必要，勿增实体**。在模型选择上，我们应该**从一个基线开始，逐步增加复杂性，并通过实验去验证每一步改动是否真的带来了收益**。

