# 第四节 模型的推理与优化

经过前面章节的数据处理、模型构建与训练，我们已经得到了一个可用的 NER 模型。本章将探讨如何实现模型的推理过程，并深入研究如何通过自定义损失函数来应对数据不均衡问题，通过集成可视化日志、提前停止和断点续训等功能，进一步提升训练框架的健壮性和实用性。

## 一、理解模型输出

在上一节构建 `Trainer` 时，已经明确了实体级别的 F1 值是衡量模型性能的核心标准，而非简单的 Token 分类准确率。这里探讨一下 **为什么** 需要这样做，以及这对设计推理流程有何启发。

### 1.1 Token 级准确率的陷阱

最直接的评估方式是计算 **Token 级别的分类准确率**，即模型预测正确的标签数占总标签数的比例。不过，正如在上一节中讨论过的，这个指标具有误导性，尤其是在实体词占比较低的场景中。主要问题在于 **数据不均衡**。在大部分文本中，绝大多数的 Token 标签都是 `'O'`（非实体）。一个“聪明”但完全没用的模型，如果它将所有 Token 都预测为 `'O'`，也能轻松达到一个非常高的 Token 准确率。但是，这样的模型没有识别出任何一个实体，对于当前的任务来说毫无价值。

> 当模型训练到一定阶段后，其预测结果可能会出现大量甚至全部为 `'O'`（ID 为 0）的情况。尽管此时的 Token 准确率看上去很高，但模型实际上已经陷入了通过预测多数类来最小化损失的“捷径”中，这是一种典型的过拟合现象，说明模型并没有真正学会识别实体。

### 1.2 对推理流程的启发

**模型的原始输出（Token 标签序列）本身不是最终交付物**。我们需要一个“后处理”或“解码”步骤，将这个标签序列转换成用户真正关心的结构化的实体列表。这不仅是正确评估模型的需要，也是模型能否在实际应用中创造价值的关键。

所以，当前的主要任务就是实现这个从标签序列到实体列表的解码过程。

## 二、从标签到实体：解码预测序列

模型的前向传播最终输出的是一个 `logits` 张量，形状为 `[batch_size, seq_len, num_tags]`。经过 `argmax` 操作后，会得到一个标签 ID 序列，例如 `[0, 9, 10, 11, 0, ...]`。

这个序列本身并不直观。为了进行实体级评估，或者将预测结果呈现给用户，必须实现一个 **解码 (Decode)** 函数，将这个数字序列转换成一个包含具体实体信息的列表，例如：`[{"text": "高血压", "type": "dis", "start": 3, "end": 6}]`。这个解码过程的核心，就是根据 `BMES` 标注体系的规则，从标签序列中解析出实体的边界和类型。

### 2.1 解码逻辑详解

解码函数需要遍历标签序列，并像一个“状态机”一样，根据当前遇到的标签（`B`, `M`, `E`, `S`, `O`）来维护一个 `current_entity` 对象。其解码逻辑如下：

1.  **遇到 `B-` (实体开始)**:
    -   如果此时还有一个未结束的 `current_entity`（说明上一个实体没有被 `E-` 正常闭合），则将其视为一个无效片段并**放弃**。
    -   创建一个新的 `current_entity` 对象，记录下它的类型、起始位置和起始字符。

2.  **遇到 `M-` (实体中间)**:
    -   检查当前是否存在一个 `current_entity`，并且其类型与 `M-` 标签的类型是否一致。
    -   如果一致，将当前字符追加到 `current_entity` 的 `text` 中。
    -   如果不一致（例如 `B-dis` 后面跟了一个 `M-sym`），则说明这是一个非法的标签序列。我们将 `current_entity` 重置为 `None`，放弃这个不完整的片段。

3.  **遇到 `E-` (实体结束)**:
    -   与 `M-` 标签的检查逻辑类似，首先确保存在一个类型匹配的 `current_entity`。
    -   如果匹配，将当前字符追加进去，并记录下结束位置 `end = i + 1`。
    -   此时，一个完整的实体已经被识别出来，将其添加到最终的 `entities` 列表中。
    -   最后，**必须** 将 `current_entity` 重置为 `None`，表示当前实体已处理完毕。

4.  **遇到 `S-` (单字实体)**:
    -   同样地，先放弃任何未闭合的 `current_entity`。
    -   直接创建一个包含类型、文本、起始和结束位置的完整实体，并将其添加到 `entities` 列表中。

5.  **遇到 `O` (非实体)**:
    -   `O` 标签的出现意味着当前位置没有实体，或者一个实体刚刚结束。
    -   如果此时还有一个未闭合的 `current_entity`，放弃它，并将 `current_entity` 重置为 `None`。

这个过程确保了只有符合 `BMES` 规范、被正确“闭合”的实体才会被最终提取出来，继而保证了解码结果的健壮性。

> **解码策略：**
> 
> 当前采用的是一种 **“严格”模式**。任何不符合规范的序列（例如只有 `B-` 没有 `E-` 的实体）都会被直接放弃。这是最常见的做法，因为它能保证输出实体的规范性。
>
> 在某些特定的业务场景下，也可以采用更 **“宽松”的策略**。例如，如果模型预测出一个 `B-M-O` 的序列，可以选择将 `B-M` 这部分作为一个实体输出，而不是完全丢弃它。这种策略的选择，取决于具体应用对“召回率”和“精确率”的不同侧重，需要根据实际需求来决定。

### 2.2 代码实现

这个解码逻辑在 `06_predict.py` 中实现为一个名为 `_extract_entities` 的方法。它接收分词后的 `tokens` 列表和模型预测的 `tags` 列表作为输入，输出结构化的实体字典列表。

```python
# code/C8/06_predict.py

def _extract_entities(self, tokens, tags):
    entities = []
    current_entity = None
    for i, tag in enumerate(tags):
        if tag.startswith('B-'):
            # 如果前一个实体未正确结束，则放弃
            if current_entity:
                pass # 或者可以根据业务逻辑决定是否保存不完整的实体
            current_entity = {"text": tokens[i], "type": tag[2:], "start": i}
        elif tag.startswith('M-'):
            # M 标签必须跟在 B- 或 M- 之后
            if current_entity and current_entity["type"] == tag[2:]:
                current_entity["text"] += tokens[i]
            else:
                # 非法 M 标签，重置当前实体
                current_entity = None
        elif tag.startswith('E-'):
            # E 标签必须跟在 B- 或 M- 之后
            if current_entity and current_entity["type"] == tag[2:]:
                current_entity["text"] += tokens[i]
                current_entity["end"] = i + 1
                entities.append(current_entity)
            # 实体已结束，重置
            current_entity = None
        elif tag.startswith('S-'):
            # S 标签表示单个字符的实体
            # 如果有未结束的实体，则放弃
            current_entity = None
            entities.append({"text": tokens[i], "type": tag[2:], "start": i, "end": i + 1})
        else: # 'O' 标签
            # O 标签意味着没有实体，或者实体已经结束
            # 如果有未结束的实体，则放弃
            current_entity = None
    
    # 循环结束后，不再处理任何未闭合的实体
    return entities
```

## 三、封装推理器

最后将所有推理相关的逻辑（加载模型、文本预处理、模型预测、结果解码）封装到一个 `NerPredictor` 类中，使其成为一个开箱即用的独立组件。

### 3.1 推理器的设计

一个好的推理器应该具备以下特点：
-   **易于初始化**: 只需提供训练好的模型目录，就能自动加载所有必要的资源（模型权重、配置文件、词汇表等）。
-   **接口简洁**: 提供一个简单的 `predict(text)` 方法，接收原始文本字符串，返回结构化的实体列表。
-   **与训练解耦**: 推理过程不应依赖任何训练时的代码或对象。


### 3.2 `NerPredictor` 核心流程

#### 3.2.1 初始化 `__init__`

`__init__` 方法的目标是加载并准备好所有推理所需的组件。

1.  **加载配置**: 从模型目录加载 `config.json`，获取模型超参数和相关文件路径。
    > **[开发插曲] 确保训练与推理的配置同步**
    >
    > 在编写 `NerPredictor` 时，可能会遇到了一个问题：推理脚本需要知道训练时使用的模型配置（如 `hidden_size` 等）才能正确地重建模型，但之前的训练脚本 `05_train.py` 并没有将这些配置信息保存下来。
    >
    > 这会导致在运行 `06_predict.py` 时出现 `FileNotFoundError: [Errno 2] No such file or directory: 'output/config.json'` 的错误。
    >
    > 为了解决这个问题，回到 `05_train.py`，**增加一步：在训练开始前，将当前的配置对象保存到输出目录中**。这样，训练和推理阶段就能共享同一份配置，确保信息同步。
    >
    > ```python
    > # code/C8/05_train.py
    >
    > from dataclasses import asdict
    > from src.utils.file_io import save_json
    > 
    > def main():
    >     # ... (组件初始化)
    > 
    >     trainer = Trainer(...)
    > 
    >     # 在训练开始前，保存配置文件
    >     os.makedirs(config.output_dir, exist_ok=True)
    >     save_json(asdict(config), os.path.join(config.output_dir, "config.json"))
    >     print(f"Configuration saved to {os.path.join(config.output_dir, 'config.json')}")
    > 
    >     trainer.fit(epochs=config.epochs)
    > ```

2.  **加载词汇表和标签映射**: 根据配置文件中的路径，加载 `vocabulary.json` 和 `tags.json`，并构建 `id2tag` 映射。
3.  **加载分词器**: 初始化 `CharTokenizer`。
4.  **初始化模型并加载权重**:
    -   根据配置实例化 `BiGRUNerNetWork` 模型。
    -   从模型目录加载 `best_model.pth` 模型权重。这里需要使用 `map_location=self.device` 来确保模型可以被加载到指定的设备上（无论是 CPU 还是 GPU）。
    -   调用 `model.to(self.device)` 将模型移至指定设备。
    -   调用 `model.eval()` 将模型切换到**评估模式**，关闭 Dropout 和 BatchNorm 等只在训练时使用的层，确保预测结果的确定性。

#### 3.2.2 预测 `predict`

`predict` 方法负责执行从原始文本到实体列表的完整端到端流程。

1.  **预处理**:
    -   调用 `tokenizer` 将输入文本转换为 `token_ids`。
    -   将 `token_ids` 转换为 `torch.Tensor`，并添加一个 batch 维度（因为模型期望的输入是 `[batch_size, seq_len]`）。
    -   创建 `attention_mask`。
    -   将所有张量移动到 `self.device`。
2.  **模型预测**:
    -   使用 `with torch.no_grad():` 临时禁用梯度计算，减少内存消耗并加速推理过程。
    -   将 `token_ids` 和 `attention_mask` 送入模型，得到 `logits`。
3.  **后处理**:
    -   对 `logits` 在最后一个维度上执行 `argmax`，得到预测的 `label_ids` 序列。
    -   使用 `id2tag` 映射，将 `label_ids` 转换为 `tags` 字符串列表。
    -   调用 `_extract_entities` 方法，完成最终的解码，返回实体列表。

### 3.3 完整代码实现

在清晰地理解了设计思路和流程后，下面是 `06_predict.py` 的完整代码。

```python
# code/C8/06_predict.py
import torch
import json
import os
import argparse
from src.models.ner_model import BiGRUNerNetWork
from src.tokenizer.vocabulary import Vocabulary
from src.tokenizer.char_tokenizer import CharTokenizer
from src.utils.file_io import load_json

class NerPredictor:
    def __init__(self, model_dir, device='cpu'):
        self.device = torch.device(device)
        
        # --- 1. 加载配置文件以获取模型参数 ---
        config_path = os.path.join(model_dir, 'config.json')
        self.config = load_json(config_path)

        # --- 2. 加载词汇表和标签映射 ---
        vocab_path = os.path.join(self.config["data_dir"], self.config["vocab_file"])
        tags_path = os.path.join(self.config["data_dir"], self.config["tags_file"])

        self.vocab = Vocabulary.load_from_file(vocab_path)
        self.tokenizer = CharTokenizer(self.vocab)
        tag_map = load_json(tags_path)
        self.id2tag = {v: k for k, v in tag_map.items()}

        # --- 3. 初始化模型并加载权重 ---
        self.model = BiGRUNerNetWork(
            vocab_size=len(self.vocab),
            hidden_size=self.config["hidden_size"],
            num_tags=len(tag_map),
            num_gru_layers=self.config["num_gru_layers"]
        )
        model_path = os.path.join(model_dir, 'best_model.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        tokens = self.tokenizer.text_to_tokens(text)
        token_ids = self.tokenizer.tokens_to_ids(tokens)
        
        # --- 预处理 ---
        token_ids_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.ones_like(token_ids_tensor)

        # --- 模型预测 ---
        with torch.no_grad():
            logits = self.model(token_ids_tensor, attention_mask)
        
        # --- 后处理 ---
        predictions = torch.argmax(logits, dim=-1).squeeze(0)
        tags = [self.id2tag[id_.item()] for id_ in predictions]

        return self._extract_entities(tokens, tags)

    def _extract_entities(self, tokens, tags):
        entities = []
        current_entity = None
        for i, tag in enumerate(tags):
            if tag.startswith('B-'):
                if current_entity:
                    pass
                current_entity = {"text": tokens[i], "type": tag[2:], "start": i}
            elif tag.startswith('M-'):
                if current_entity and current_entity["type"] == tag[2:]:
                    current_entity["text"] += tokens[i]
                else:
                    current_entity = None
            elif tag.startswith('E-'):
                if current_entity and current_entity["type"] == tag[2:]:
                    current_entity["text"] += tokens[i]
                    current_entity["end"] = i + 1
                    entities.append(current_entity)
                current_entity = None
            elif tag.startswith('S-'):
                current_entity = None
                entities.append({"text": tokens[i], "type": tag[2:], "start": i, "end": i + 1})
            else: # 'O' 标签
                current_entity = None
        
        return entities

def main():
    parser = argparse.ArgumentParser(description="NER Prediction")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of the saved model and config.")
    parser.add_argument("--text", type=str, required=True, help="Text to predict.")
    args = parser.parse_args()

    predictor = NerPredictor(model_dir=args.model_dir)
    entities = predictor.predict(args.text)
    print(f"Text: {args.text}")
    print(f"Entities: {json.dumps(entities, ensure_ascii=False, indent=2)}")

if __name__ == "__main__":
    main()
```

### 3.4 使用示例

`06_predict.py` 的 `main` 函数提供了一个标准的命令行使用接口。在训练完成后，可以通过以下命令来调用训练好的模型进行预测：

```bash
python 06_predict.py --model_dir "output" --text "患者自述发热、咳嗽，伴有轻微头痛。"
```

-   `--model_dir`: 指向我们第三节中训练结果的输出目录（包含了 `best_model.pth` 和 `config.json`）。
-   `--text`: 需要进行实体识别的文本。

**预期输出**:

> 由于我们仅进行了简单的训练，并未进行调优，所以当前模型的预测结果可能并不完美（例如可能只识别出部分实体或单字实体）。这里展示的输出主要是为了说明整个推理流程的格式和工作方式。

```text
Text: 患者自述发热、咳嗽，伴有轻微头痛。
Entities: [
  {
    "text": "发",
    "type": "sym",
    "start": 4,
    "end": 5
  },
  {
    "text": "咳",
    "type": "sym",
    "start": 7,
    "end": 8
  }
]
```

## 四、自定义损失函数

在当前使用的 CMeEE 数据集中，数据不均衡是一个显著的特点：大部分 Token 都是非实体的 'O' 标签。虽然导致模型性能不佳的原因可能多种多样，但这种数据不均衡无疑是影响模型学习效果的关键因素之一。仅仅依赖实体级评估指标是在“下游”进行补救，我们也可以尝试从“上游”——即损失函数的设计入手，主动引导模型去关注实体样本。

标准的交叉熵损失函数对所有 Token 一视同仁，当 `'O'` 标签占据绝大多数时，损失值自然会被这些“多数派”主导。下面介绍两种策略，来尝试缓解这个问题。

### 4.1 核心策略

#### 4.1.1 加权交叉熵损失

最简单的方法就是“加权”。给数量稀少的实体标签（B, M, E, S）一个更高的权重，给数量庞大的非实体标签（O）一个较低的权重。例如，我们可以设置实体损失的权重为 10，非实体损失的权重为 1。这样，模型在反向传播时，如果弄错了一个实体 Token，会受到比弄错一个非实体 Token 大 10 倍的“惩罚”，从而迫使模型更加关注对实体的识别。

#### 4.1.2 硬负样本挖掘

另一种思路是“采样”。在大量的非实体样本中，大部分是模型可以轻易正确预测的“简单样本”，它们对损失的贡献很小，反复学习意义不大。真正有价值的是那些模型容易搞错的“硬负样本”，例如一个模型倾向于预测为实体的非实体 Token。

硬负样本挖掘的做法是：在计算非实体部分的损失时，不计算所有非实体 Token 的平均损失，而是只选择其中损失值最大（Top-K）的一部分进行计算和反向传播。这样就相当于从海量的“多数派”中，筛选出了最有价值的“疑难样本”进行学习，提升了训练的效率和效果。

### 4.2 代码实现

为了将上述策略集成到训练框架中，来创建一个新的 `NerLoss` 类，并修改项目的相关部分来调用它。

#### 4.2.1 创建 `NerLoss`

首先，在 `src` 目录下创建一个新的 `loss` 文件夹，并在其中新建 `ner_loss.py` 文件。

```python
# code/C8/src/loss/ner_loss.py

import torch
import torch.nn as nn

class NerLoss(nn.Module):
    """
    自定义 NER 损失函数，集成两种策略来对抗数据不均衡问题：
    1. 加权交叉熵
    2. 硬负样本挖掘
    """
    def __init__(self, loss_type='cross_entropy', entity_weight=10.0, hard_negative_ratio=0.5, ignore_index=-100):
        super().__init__()
        # --- 参数定义 ---
        self.loss_type = loss_type                # 损失类型: 'cross_entropy', 'weighted_ce', 'hard_negative_mining'
        self.entity_weight = entity_weight        # 实体损失的权重
        self.hard_negative_ratio = hard_negative_ratio  # 硬负样本与正样本的比例
        
        # 基础损失函数，设置为 'none' 模式以获取每个 token 的单独损失
        self.base_loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, logits, labels):
        """
        根据初始化时选择的 loss_type 计算损失。
        """
        if self.loss_type == 'weighted_ce':
            return self._weighted_cross_entropy(logits, labels)
        elif self.loss_type == 'hard_negative_mining':
            return self._hard_negative_mining(logits, labels)
        else: 
            # 默认使用 PyTorch 原生的交叉熵损失
            return self.base_loss_fn(logits, labels).mean()

    def _weighted_cross_entropy(self, logits, labels):
        """
        加权交叉熵损失的实现。
        """
        # 计算每个 token 的基础损失, shape: [batch_size, seq_len]
        loss_per_token = self.base_loss_fn(logits, labels)

        # 创建掩码来区分实体和非实体 token
        entity_mask = (labels > 0).float()      # 实体 (B, M, E, S)
        non_entity_mask = (labels == 0).float() # 非实体 (O)

        # 分别计算实体和非实体部分的平均损失
        entity_loss = torch.sum(loss_per_token * entity_mask) / (torch.sum(entity_mask) + 1e-8)
        non_entity_loss = torch.sum(loss_per_token * non_entity_mask) / (torch.sum(non_entity_mask) + 1e-8)

        # 根据预设权重，组合两部分损失
        total_loss = self.entity_weight * entity_loss + 1.0 * non_entity_loss
        return total_loss, entity_loss.detach(), non_entity_loss.detach()

    def _hard_negative_mining(self, logits, labels):
        """
        硬负样本挖掘损失的实现。
        """
        # 计算每个 token 的基础损失
        loss_per_token = self.base_loss_fn(logits, labels)

        # 实体部分的损失计算与加权交叉熵方法相同
        entity_mask = (labels > 0).float()
        entity_loss = torch.sum(loss_per_token * entity_mask) / (torch.sum(entity_mask) + 1e-8)

        # 筛选出所有非实体 token 的损失
        non_entity_mask = (labels == 0).float()
        non_entity_loss = loss_per_token * non_entity_mask

        # 确定要挖掘的硬负样本数量
        num_entities = torch.sum(entity_mask).item()
        num_hard_negatives = int(num_entities * self.hard_negative_ratio)

        # 如果当前批次没有实体，则按固定比例选择负样本，避免数量为0
        if num_hard_negatives == 0:
            num_non_entities = torch.sum(non_entity_mask).item()
            num_hard_negatives = int(num_non_entities * 0.1)

        # 从非实体损失中选出最大的 top-k 个作为硬负样本
        topk_losses, _ = torch.topk(non_entity_loss.view(-1), k=num_hard_negatives)
        
        # 计算硬负样本的平均损失
        hard_negative_loss = torch.mean(topk_losses)

        # 结合实体损失和硬负样本损失
        total_loss = self.entity_weight * entity_loss + 1.0 * hard_negative_loss

        return total_loss, entity_loss.detach(), hard_negative_loss.detach()
```

这个类封装了所有与损失计算相关的逻辑。它会返回一个元组 `(总损失, 实体损失, 非实体损失)`，便于我们在训练日志中观察不同部分损失的变化情况。

#### 4.2.2 硬负样本挖掘实现细节

在 `_hard_negative_mining` 的实现中，有一个需要特别注意的细节：`torch.topk` 函数要求 `k` 的值不能超过输入张量的维度大小。在此场景中，如果计算出的 `num_hard_negatives` 超过了当前批次中非实体 `O` 的总数，就会引发运行时错误。

同时，需要将二维的 `non_entity_loss` 展平（`view(-1)`）成一维，以确保 `topk` 是在所有非实体样本中寻找损失最大的 `k` 个。下面是修正后的关键代码片段：

```python
# code/C8/src/loss/ner_loss.py

def _hard_negative_mining(self, logits, labels):
    # ... (省略实体损失计算)

    non_entity_mask = (labels == 0).float()
    non_entity_loss = loss_per_token * non_entity_mask
    
    num_hard_negatives = int(torch.sum(entity_mask).item() * self.hard_negative_ratio)
    if num_hard_negatives == 0:
         num_hard_negatives = int(torch.sum(non_entity_mask).item() * 0.1)

    # 关键修改：将损失展平为一维
    non_entity_loss_flat = non_entity_loss.view(-1)
    
    # 关键修改：确保 k 不超过非实体 token 的总数
    num_non_entities = torch.sum(non_entity_mask).item()
    k = min(num_hard_negatives, num_non_entities)
    
    if k == 0: # 如果没有负样本可选，则损失为 0
        non_ner_loss_mean = torch.tensor(0.0, device=logits.device)
    else:
        topk_losses, _ = torch.topk(non_entity_loss_flat, k=k)
        non_ner_loss_mean = torch.mean(topk_losses)

    total_loss = self.entity_weight * ner_loss_mean + 1.0 * non_ner_loss_mean
    return total_loss, ner_loss_mean.detach(), non_ner_loss_mean.detach()
```

#### 4.2.3 更新配置文件

接着，需要在 `src/configs/configs.py` 中添加几个参数，以便能够灵活地选择和配置损失函数。

```python
# code/C8/src/configs/configs.py

# ...
    learning_rate: float = 1e-3
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 损失函数参数 ---
    loss_type: str = "weighted_ce"  # 可选: "cross_entropy", "weighted_ce", "hard_negative_mining"
    entity_loss_weight: float = 10.0 # 在 weighted_ce 和 hard_negative_mining 中, 给实体部分损失的权重
    hard_negative_ratio: float = 0.5 # 在 hard_negative_mining 中, 负样本数量与正样本数量的比例

    # --- 模型参数 ---
# ...
```

#### 4.2.4 修改训练器

为了处理 `NerLoss` 返回的多个损失值，并优化训练日志，需要对 `src/trainer/trainer.py` 进行升级。

主要的修改点包括：
- 仅用“主损”反向传播（若为元组损失，取 `loss[0]`）。
- 训练阶段累计并返回三元组（总损/实体/非实体）。
- 评估阶段用“主损”统计验证集 `loss`。
- 保存最优模型以 `{'model_state_dict': ...}` 方式，便于 `06_predict.py` 直接加载。

```python
# code/C8/src/trainer/trainer.py

# ... (省略未修改部分)
from tqdm import tqdm
import os
import torch

class Trainer:
    # ... (省略 __init__ 等)

    def fit(self, epochs):
        os.makedirs(self.output_dir, exist_ok=True)
        best_metric = float('-inf')  # 优先最大化 F1
        for epoch in range(1, epochs + 1):
            print(f"--- Epoch {epoch}/{epochs} ---")
            train_losses = self._train_one_epoch()
            # 支持元组损失的日志打印（总损/实体/非实体）
            if isinstance(train_losses, tuple):
                train_loss_str = (
                    f"Train Total Loss: {train_losses[0]:.4f}, "
                    f"NER Loss: {train_losses[1]:.4f}, "
                    f"Non-NER Loss: {train_losses[2]:.4f}"
                )
            else:
                train_loss_str = f"Train Total Loss: {train_losses:.4f}"
            print(train_loss_str)

            eval_metrics = self._evaluate()
            eval_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in eval_metrics.items()])
            print(f"Validation Metrics: {eval_metrics_str}")

            # 以验证集 F1 作为保存准则；无 F1 时回退用 loss
            is_best = False
            if 'f1' in eval_metrics:
                if eval_metrics['f1'] > best_metric:
                    best_metric = eval_metrics['f1']
                    is_best = True
            else:
                if best_metric == float('-inf'):
                    best_metric = float('inf')
                if eval_metrics['loss'] < best_metric:
                    best_metric = eval_metrics['loss']
                    is_best = True

            if is_best:
                print(f"New best model found! Saving to {self.output_dir}")
                # 以字典方式保存，键为 'model_state_dict'，便于 06_predict.py 加载
                torch.save({'model_state_dict': self.model.state_dict()},
                           os.path.join(self.output_dir, "best_model.pth"))

    def _train_one_epoch(self):
        self.model.train()
        total_loss_sum = 0
        total_ner_loss = 0
        total_non_ner_loss = 0
        custom_loss_used = False

        for batch in tqdm(self.train_loader, desc=f"Training Epoch"):
            outputs = self._train_step(batch)
            loss = outputs['loss']
            if isinstance(loss, tuple):
                # 支持元组损失（总损/实体/非实体）并分别累计
                custom_loss_used = True
                total_loss_sum += loss[0].item()
                total_ner_loss += loss[1].item()
                total_non_ner_loss += loss[2].item()
            else:
                total_loss_sum += loss.item()

        if custom_loss_used:
            # 返回三元组 (avg_total, avg_ner, avg_non_ner)
            avg_loss = total_loss_sum / len(self.train_loader)
            avg_ner_loss = total_ner_loss / len(self.train_loader)
            avg_non_ner_loss = total_non_ner_loss / len(self.train_loader)
            return avg_loss, avg_ner_loss, avg_non_ner_loss
        else:
            return total_loss_sum / len(self.train_loader)

    def _train_step(self, batch):
        # ... (省略前向部分)
        logits = self.model(token_ids=batch['token_ids'], attention_mask=batch['attention_mask'])
        loss = self.loss_fn(logits.permute(0, 2, 1), batch['label_ids'])
        # 仅用主损进行反向传播（元组时取 loss[0]）
        main_loss = loss[0] if isinstance(loss, tuple) else loss
        self.optimizer.zero_grad()
        main_loss.backward()
        self.optimizer.step()
        return {'loss': loss, 'logits': logits}

    def _evaluate(self):
        if self.dev_loader is None:
            return None
        self.model.eval()
        total_loss = 0
        all_logits, all_labels, all_attention_mask = [], [], []
        with torch.no_grad():
            for batch in tqdm(self.dev_loader, desc="Evaluating"):
                outputs = self._evaluation_step(batch)
                loss = outputs['loss']
                # 验证 loss 也使用主损统计
                main_loss = loss[0] if isinstance(loss, tuple) else loss
                total_loss += main_loss.item()
                all_logits.append(outputs['logits'].cpu())
                all_labels.append(batch['label_ids'].cpu())
                all_attention_mask.append(batch['attention_mask'].cpu())
        metrics = {}
        if self.eval_metric_fn:
            metrics = self.eval_metric_fn(all_logits, all_labels, all_attention_mask)
        metrics['loss'] = total_loss / len(self.dev_loader)
        return metrics

    # ... (其余方法保持不变)
```

#### 4.2.5 集成到主函数

最后一步，在 `05_train.py` 中根据配置来实例化对应的损失函数。

```python
# code/C8/05_train.py

# ...
from src.loss.ner_loss import NerLoss # 导入新模块

# ... (在main函数中)
    # --- 3. 初始化模型、优化器、损失函数 ---
    model = BiGRUNerNetWork(...)
    optimizer = torch.optim.AdamW(...)
    
    # 根据配置选择损失函数
    if config.loss_type == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    else:
        loss_fn = NerLoss(
            loss_type=config.loss_type,
            entity_weight=config.entity_loss_weight,
            hard_negative_ratio=config.hard_negative_ratio
        )
# ...
```

完成以上步骤后，就可以通过简单地修改 `configs.py` 中的 `loss_type` 参数，来切换不同的损失函数策略，并观察它们对模型训练效果的影响。例如，将 `loss_type` 设置为 `"weighted_ce"`，然后重新运行 `05_train.py`，会看到训练日志中包含了实体和非实体各自的损失值。

#### 4.2.6 解读验证集损失

在使用自定义损失函数（尤其是 `weighted_ce` 和 `hard_negative_mining`）时，你可能会观察到一个现象：**验证集上的 F1 分数在稳步提升，但 `loss` 值却停滞不前甚至上升**。这是一个正常且符合预期的现象。

这是因为 `Trainer` 在评估阶段同样使用了这个自定义的、加权的损失函数来计算验证集 `loss`。这个 `loss` 主要反映的是**训练目标**的优化情况，而不是一个标准的评估指标。

-   **权重影响**: 由于实体部分的损失被赋予了很高的权重（例如 `entity_loss_weight=10.0`），少数几个实体相关的错误就会导致 `loss` 值大幅波动或居高不下。
-   **硬负样本挖掘影响**: `hard_negative_mining` 策略会动态地聚焦于模型最容易搞错的那些非实体 `O` 标签。随着训练的进行，简单的负样本损失会降低，但模型会转而面对更“棘手”的硬样本，导致计算出的 `non_ner_loss` 可能不会持续下降。

因此，当使用这些高级损失策略时，**验证集 `loss` 不再是衡量模型好坏的主要标准**。应将注意力更多地放在能够直接反映任务最终目标的指标上，对于 NER 任务而言，这个指标就是**实体级别的 F1 分数**。这也是 `Trainer` 将 F1 作为保存最佳模型依据的原因。

## 五、优化训练工作流

在我们实现了核心的训练、评估与推理流程之后，一个健robustness的训练框架还需要更多辅助功能来应对真实场景中的各种挑战。本节将介绍如何为 Trainer 集成三项关键的实用功能：**可视化日志**、**提前停止**和**断点续训**，让训练过程更加可控、高效和可靠。

### 5.1 训练过程可视化

纯文本的训练日志虽然直接，但难以洞察模型训练的全局动态。为了更直观地监控训练过程，例如观察损失是否平稳下降、验证集 F1 是否持续提升，以及模型是否出现过拟合迹象，可以集成 TensorBoard 来实现可视化；同时，为提高结果的可复现性，建议在训练开始前固定随机数种子。

为了将日志记录功能模块化，可以创建一个专门的 `TensorBoardLogger` 类来封装所有与 `SummaryWriter` 相关的操作。

1.  **创建 `TensorBoardLogger` 类**:

    在 `src/utils/` 目录下创建 `logger.py` 文件。这个类将负责 `SummaryWriter` 的初始化、指标记录和关闭。

    ```python
    # code/C8/src/utils/logger.py
    from torch.utils.tensorboard import SummaryWriter

    class TensorBoardLogger:
        def __init__(self, log_dir):
            # 如果提供了日志目录，则初始化 SummaryWriter
            self.writer = SummaryWriter(log_dir) if log_dir else None

        def log_metrics(self, metrics, step, prefix):
            # 如果 writer 未初始化，则不执行任何操作
            if self.writer is None: return
            
            # 根据 metrics 类型（元组或字典）以不同方式记录
            if isinstance(metrics, tuple):
                self.writer.add_scalar(f"{prefix}/Total_Loss", metrics[0], step)
                if len(metrics) > 1:
                    self.writer.add_scalar(f"{prefix}/NER_Loss", metrics[1], step)
                    self.writer.add_scalar(f"{prefix}/Non-NER_Loss", metrics[2], step)
            elif isinstance(metrics, dict):
                for k, v in metrics.items():
                    self.writer.add_scalar(f"{prefix}/{k.capitalize()}", v, step)
        
        def close(self):
            # 确保在训练结束时关闭 writer，将所有挂起的事件写入磁盘
            if self.writer:
                self.writer.close()
    ```

2.  **在 `configs.py` 中添加配置**:

    ```python
    # code/C8/src/configs/configs.py
    
    # ... (省略)
    class NerConfig:
        # ... (省略)
        # --- 增强功能参数 ---
        output_summary_dir: str = "output/logs" # TensorBoard 日志输出路径
        seed: int = 42  # 随机数种子（用于可复现性）
    # ... (省略)
    ```

3.  **在 `Trainer` 中使用 `TensorBoardLogger`**:

    ```python
    # code/C8/src/trainer/trainer.py
    from src.utils.logger import TensorBoardLogger

    class Trainer:
        def __init__(self, ..., summary_writer_dir=None, ...):
            # ... (省略其他初始化)
            # 初始化日志记录器
            self.logger = TensorBoardLogger(summary_writer_dir)
            
        def fit(self, epochs):
            for epoch in range(self.start_epoch, epochs + 1):
                # ... (训练与评估)
                
                # 在每个 epoch 结束后调用 logger 记录训练和验证指标
                self.logger.log_metrics(train_losses, epoch, "Train")
                self.logger.log_metrics(eval_metrics, epoch, "Validation")
            
            # 训练结束后关闭 logger
            self.logger.close()
    ```

4.  **添加随机数种子**

    为了使可视化对比与调参更稳定可复现，建议在训练启动时固定随机数种子，读取 `configs.py` 中新增的 `seed` 配置。

    在 `05_train.py` 中添加工具函数并调用：

    ```python
    # code/C8/05_train.py
    # ... 省略导入
    
    def seed_everything(seed: int = 42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def main():
        # 训练前设置随机种子（读取 configs.py 的 seed）
        seed_everything(getattr(config, 'seed', 42))
        # ... 后续组件初始化与训练
    ```

### 5.2 早停实现

为了让这个逻辑更清晰且可复用，可将其封装到一个独立的 `EarlyStopping` 类中，这个类就像一个“回调”一样，在每个 epoch 结束时被 `Trainer` 调用来检查是否需要停止。

1.  **创建 `EarlyStopping` 工具类**:

    在 `src/utils/` 目录下创建一个新文件 `early_stop.py`。

    ```python
    # code/C8/src/utils/early_stop.py
    import numpy as np

    class EarlyStopping:
        def __init__(self, patience=5, verbose=False, delta=0, monitor='f1', mode='max'):
            self.patience = patience          # 耐心值：连续多少轮性能没有提升则停止
            self.verbose = verbose            # 是否打印日志
            self.counter = 0                  # 计数器
            self.best_score = None            # 历史最佳分数
            self.early_stop = False           # 提前停止标志
            self.val_metric_best = np.inf if mode == 'min' else -np.inf # 根据模式初始化最佳指标
            self.delta = delta                # 容忍的性能下降范围
            self.monitor = monitor            # 监控的指标
            self.mode = mode                  # 'max' 或 'min'

        def __call__(self, val_metric):
            # 根据 'mode' 调整分数计算方式
            score = -val_metric if self.mode == 'min' else val_metric

            if self.best_score is None:
                self.best_score = score
            # 如果当前分数没有超过（最佳分数 + delta），则增加计数器
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            # 如果分数有提升，则更新最佳分数并重置计数器
            else:
                self.best_score = score
                self.counter = 0
            return self.early_stop
    ```

2.  **在 `configs.py` 中添加配置**:

    ```python
    # code/C8/src/configs/configs.py
    
    # ... (省略)
    early_stopping_patience: int = 5 # 提前停止的耐心轮数
    ```

3.  **在 `Trainer` 中集成 `EarlyStopping` 实例**:

    ```python
    # code/C8/src/trainer/trainer.py
    from src.utils.early_stop import EarlyStopping

    class Trainer:
        def __init__(self, ..., early_stopping_patience=5, ...):
            # ... (省略其他初始化)
            # 初始化 EarlyStopping 回调
            self.early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                verbose=True,
                monitor='f1'
            )

        def fit(self, epochs):
            for epoch in range(self.start_epoch, epochs + 1):
                # ... (训练与评估)

                current_metric = eval_metrics.get('f1', -eval_metrics.get('loss', float('inf')))
                
                # ... (保存最佳模型逻辑)

                # 调用 early_stopping 实例判断是否需要停止
                if self.early_stopping(current_metric):
                    print("Early stopping triggered.")
                    break # 跳出训练循环
    ```
### 5.3 实现断点续训

对于需要数小时甚至数天的长时间训练任务，意外中断（如断电、程序崩溃）是常见风险。从头开始训练会造成巨大的时间浪费。**断点续训 (Checkpointing & Resuming)** 机制允许我们保存训练过程中的完整状态（包括模型权重、优化器状态和当前轮数），并在需要时从中恢复，继续训练。

实现此功能主要分为三步：首先添加配置项，然后在 `Trainer` 中构建核心的保存与恢复逻辑，最后在主训练脚本中启用它。

1.  **在 `configs.py` 中添加配置**:

    首先，在 `NerConfig` 中增加一个 `resume_checkpoint` 字段，用于指定需要恢复的检查点文件路径。如果它为 `None`，则从头开始训练。

    ```python
    # code/C8/src/configs/configs.py
    # ... (其他配置)
    # 用于恢复训练的检查点路径, e.g., "output/last_model.pth"
    resume_checkpoint: str = None 
    ```

2.  **为 `Trainer` 新增保存与恢复能力**:

    接下来，为 `Trainer` 类赋予保存和恢复检查点的能力。这包括新增两个核心方法 `_save_checkpoint` 和 `_resume_checkpoint`，并修改 `__init__` 和 `fit` 方法来调用它们。

    ```python
    # code/C8/src/trainer/trainer.py

    class Trainer:
        def __init__(self, ..., resume_checkpoint=None, ...):
            # ...
            self.start_epoch = 1 # 默认从第一轮开始
            # ...
            # 如果指定了检查点路径，则调用恢复方法
            if resume_checkpoint:
                self._resume_checkpoint(resume_checkpoint)

        def fit(self, epochs):
            # 使用 self.start_epoch 替换固定的 `1`，以支持从指定轮数开始
            for epoch in range(self.start_epoch, epochs + 1):
                # ... (训练和评估)

                # --- 保存逻辑 ---
                is_best = False
                current_metric = eval_metrics.get('f1', -eval_metrics.get('loss', float('inf')))
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    is_best = True
                
                # 在每轮结束后都保存检查点
                self._save_checkpoint(epoch, is_best)

                # ... (早停逻辑)

                # 调用 early_stopping 实例判断是否需要停止
                if self.early_stopping(current_metric):
                    print("Early stopping triggered.")
                    break # 跳出训练循环
    ```
    这里有几个关键点：
    - `__init__` 中会检查 `resume_checkpoint`，如果提供了路径，就调用恢复方法。
    - `fit` 方法的循环 `for epoch in range(1, epochs + 1)` 需要修改为 `for epoch in range(self.start_epoch, epochs + 1)`，以便从恢复的轮数继续训练。
    - `fit` 方法在每轮结束时调用 `_save_checkpoint` 来保存当前状态。

3.  **在 `05_train.py` 中启用并校验**:

    最后，在主训练脚本中，我们需要在初始化 `Trainer` 之前，先检查配置文件中 `resume_checkpoint` 指定的路径是否有效。如果路径无效，就将其置为 `None`，以确保 `Trainer` 能够安全地从头开始训练，而不是因找不到文件而报错。

    ```python
    # code/C8/05_train.py

    # ... (省略前半部分)

    # 在初始化 Trainer 前，检查检查点文件是否存在
    if config.resume_checkpoint and not os.path.exists(config.resume_checkpoint):
        print(f"Checkpoint file not found: {config.resume_checkpoint}. Starting training from scratch.")
        config.resume_checkpoint = None # 设为 None, 避免 Trainer 报错

    trainer = Trainer(
        # ...
        resume_checkpoint=config.resume_checkpoint
    )
    ```

### 5.4 更新主训练脚本

完成了对 `Trainer` 的升级并将日志、早停等功能模块化后，最后一步是在主训练脚本 `05_train.py` 中，将相应的配置参数传递给 `Trainer` 实例，从而正式启用这些新功能。

```python
# code/C8/05_train.py

# ... (省略前半部分代码)

# --- 5. 初始化并启动训练器 ---
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_loader=train_loader,
    dev_loader=dev_loader,
    eval_metric_fn=eval_metric_fn,
    output_dir=config.output_dir,
    device=config.device,
    # 传入新增的配置参数，以启用对应的功能
    summary_writer_dir=config.output_summary_dir,      # TensorBoard 日志目录
    early_stopping_patience=config.early_stopping_patience, # 早停耐心轮数
    resume_checkpoint=config.resume_checkpoint        # 断点续训的检查点路径
)

# ... (省略后半部分代码)
```

## 六、小结

回顾整个流程，一个完整的命名实体识别项目已经从零开始被系统性地构建出来。整个过程贯穿了从数据处理、模型构建到训练优化与最终推理的全流程：

*   **数据处理与准备 (第二节)**：首先解析原始的 CMeEE 数据集，构建了全局统一的 `BMES` 标签映射 (`categories.json`) 和字符级词汇表 (`vocabulary.json`)，并最终封装成一个高效、可复用的 `DataLoader`，为模型训练提供了标准化的数据输入。

*   **模型构建与训练框架 (第三节)**：设计并实现了一个基于 `Bi-GRU` 的序列标注模型，并围绕它打造了一个结构清晰、组件化的训练框架。通过将模型、数据加载器、分词器、评估指标等核心功能解耦，构建了一个易于维护和扩展的 `Trainer` 类。

*   **推理与工作流优化 (第四节)**：实现从模型输出到结构化实体的解码逻辑，并将其封装成一个开箱即用的 `NerPredictor` 推理器。同时，为了提升训练框架的健壮性和实用性，还集成了自定义损失函数来应对数据不均衡问题，并引入了 TensorBoard 可视化日志、提前停止（Early Stopping）和断点续训（Checkpointing）等高级功能。

通过以上步骤，不仅实现了一个能跑通的 NER 模型，更重要的是搭建起了一套模块化、功能完备的 NER 项目脚手架。尽管当前基线模型的性能可能还有提升空间，但这个框架为后续探索更先进的模型（如 BERT）、尝试更复杂的策略提供了不错的起点。
