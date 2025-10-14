# 第四节 模型推理

经过前面章节的数据处理、模型构建与训练，我们已经得到了一个可用的 NER 模型。本节将探讨如何实现最后的模型推理的过程。接下来我们从理解模型的原始输出开始，探讨其局限性，然后详细拆解如何将这些输出解码为有意义的、结构化的实体信息，并最终将所有逻辑封装成一个简洁、可复用的 `NerPredictor` 类。

## 一、理解模型输出的挑战

在上一节构建 `Trainer` 时，已经明确了实体级别的 F1 值是衡量模型性能的核心标准，而非简单的 Token 分类准确率。这里探讨一下 **为什么** 需要这样做，以及这对我们设计推理流程有何启发。

### 1.1 Token 级准确率的陷阱

最直接的评估方式是计算 **Token 级别的分类准确率**，即模型预测正确的标签数占总标签数的比例。不过，正如在上一节中讨论过的，这个指标具有误导性，尤其是在实体词占比较低的场景中。主要问题在于 **数据不均衡**。在大部分文本中，绝大多数的 Token 标签都是 `'O'`（非实体）。一个“聪明”但完全没用的模型，如果它将所有 Token 都预测为 `'O'`，也能轻松达到一个非常高的 Token 准确率。但是，这样的模型没有识别出任何一个实体，对于当前的任务来说毫无价值。

> 当模型训练到一定阶段后，其预测结果可能会出现大量甚至全部为 `'O'`（ID 为 0）的情况。尽管此时的 Token 准确率看上去很高，但模型实际上已经陷入了通过预测多数类来最小化损失的“捷径”中，这是一种典型的过拟合现象，说明模型并没有真正学会识别实体。

### 1.2 对推理流程的启发

上述分析告诉我们一个关键点：**模型的原始输出（Token 标签序列）本身不是最终交付物**。我们需要一个“后处理”或“解码”步骤，将这个标签序列转换成用户真正关心的、结构化的实体列表。这不仅是正确评估模型的需要，也是模型能否在实际应用中创造价值的关键。

所以，本节后续的主要任务就是实现这个从标签序列到实体列表的解码过程。

## 二、从标签到实体：解码预测序列

模型的前向传播最终输出的是一个 `logits` 张量，形状为 `[batch_size, seq_len, num_tags]`。经过 `argmax` 操作后，我们会得到一个标签 ID 序列，例如 `[0, 9, 10, 11, 0, ...]`。

这个序列本身并不直观。为了进行实体级评估，或者将预测结果呈现给用户，我们必须实现一个 **解码 (Decode)** 函数，将这个数字序列转换成一个包含具体实体信息的列表，例如：`[{"text": "高血压", "type": "dis", "start": 3, "end": 6}]`。这个解码过程的核心，就是根据 `BMES` 标注体系的规则，从标签序列中解析出实体的边界和类型。

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

> **解码策略**
> 
> 当前采用的是一种 **“严格”模式**。任何不符合规范的序列（例如只有 `B-` 没有 `E-` 的实体）都会被直接放弃。这是最常见的做法，因为它能保证输出实体的规范性。
>
> 在某些特定的业务场景下，也可以采用更 **“宽松”的策略**。例如，如果模型预测出一个 `B-M-O` 的序列，我们可以选择将 `B-M` 这部分作为一个实体输出，而不是完全丢弃它。这种策略的选择，取决于具体应用对“召回率”和“精确率”的不同侧重，需要根据实际需求来决定。

### 2.2 代码实现

我们将这个解码逻辑在 `06_predict.py` 中实现为一个名为 `_extract_entities` 的方法。它接收分词后的 `tokens` 列表和模型预测的 `tags` 列表作为输入，输出结构化的实体字典列表。

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

标准的交叉熵损失函数对所有 Token 一视同仁，当 `'O'` 标签占据绝大多数时，损失值自然会被这些“多数派”主导。下面介绍两种自定义损失函数策略，来尝试缓解这个问题。

### 5.1 加权交叉熵损失 (Weighted Cross-Entropy)

最直观的方法就是“加权”。我们给数量稀少的实体标签（B, M, E, S）一个更高的权重，给数量庞大的非实体标签（O）一个较低的权重。例如，我们可以设置实体损失的权重为 10，非实体损失的权重为 1。这样，模型在反向传播时，如果弄错了一个实体 Token，会受到比弄错一个非实体 Token 大 10 倍的“惩罚”，从而迫使模型更加关注对实体的识别。

### 5.2 硬负样本挖掘 (Hard Negative Mining)

另一种思路是“采样”。在大量的非实体样本中，大部分是模型可以轻易正确预测的“简单样本”，它们对损失的贡献很小，反复学习意义不大。真正有价值的是那些模型容易搞错的“硬负样本”（Hard Negatives），例如一个模型倾向于预测为实体的非实体 Token。

硬负样本挖掘的做法是：在计算非实体部分的损失时，我们不计算所有非实体 Token 的平均损失，而是只选择其中损失值最大（Top-K）的一部分进行计算和反向传播。这样就相当于从海量的“多数派”中，筛选出了最有价值的“疑难样本”进行学习，提升了训练的效率和效果。
