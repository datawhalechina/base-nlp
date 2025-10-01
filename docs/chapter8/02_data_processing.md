# 第二节 NER 流程化代码实践

在上一节中，我们简单介绍了命名实体识别的任务定义、应用场景及主流实现方法。本节将正式进入编码阶段，从数据处理开始，逐步构建一个完整的 NER 项目。为了清晰地构建 NER 的处理流程，我们采用流程化的代码组织思路，将整个流程拆分为多个独立的脚本。

## 一、数据处理流程总览

在 NLP 中，原始的文本和标注数据是无法直接被神经网络模型利用的。需要将这些原始数据转换成模型能够理解的、标准化的数字张量。

### 1.1 明确数据处理的目标

在设计之前，我们首先要明确最终的目标。对于一个命名实体识别任务，数据处理流水线需要产出什么？

1.  **模型的输入 (X) 是什么？**
    -   它应该是一个整数张量，形状为 `[batch_size, seq_len]`。
    -   其中 `batch_size` 是批次大小，`seq_len` 是序列长度（通常是批次内最长句子的长度）。
    -   张量中的每一个数字，都代表原始句子中一个字符（Token）在词汇表里对应的唯一 ID。

2.  **模型的标签 (Y) 是什么？**
    -   它也应该是一个整数张量，形状与输入 X 完全相同，即 `[batch_size, seq_len]`。
    -   其中的每一个数字，代表着对应位置字符的实体标签 ID（例如，`B-bod` 对应的 ID）。

3.  **如何实现从“文本”到“ID”的转换？**
    -   **文本 -> Token ID**：需要构建一个 “字符-ID” 的映射表，也就是**词汇表 (Vocabulary)**。
    -   **实体 -> 标签 ID**：需要构建一个 “标签-ID” 的映射表。

### 1.2 数据格式解析

我们使用的是 `CMeEE-V2`（中文医学实体抽取）数据集，采用 JSON Lines 格式存储。

#### 1.2.1 原始数据示例

打开 `CMeEE-V2_train.json`，每一行是一个独立的 JSON 对象：

```json
{
  "text": "胃癌癌前病变和胃癌根治术后患者，...",
  "entities": [
    {"start_idx": 7, "end_idx": 9, "type": "dis", "entity": "胃癌"},
    {"start_idx": 29, "end_idx": 34, "type": "pro", "entity": "胃癌根治术"}
  ]
}
```

#### 1.2.2 字段说明

-   **`text`**：原始文本字符串
-   **`entities`**：实体标注列表，每个实体包含：
    -   `start_idx`：实体起始位置（**包含**）
    -   `end_idx`：实体结束位置（**不包含**）
    -   `type`：实体类型（如 `dis` 疾病、`pro` 诊疗程序）
    -   `entity`：实体文本（用于验证）

> **关键细节：索引的包含性**
>
> 通过实际测试可以验证：`start_idx` **包含**在实体范围内，`end_idx` **不包含**。这与 Python 的切片操作 `text[start:end]` 行为一致。例如：
> - 文本："胃癌癌前病变和胃癌根治术后患者"
> - 实体 "胃癌"：`start_idx=7, end_idx=9`
> - 实际字符：`text[7:9]` = "胃癌"（索引7和8）
>
> 因此，实体长度 = `end_idx - start_idx`

## 二、步骤一：构建标签映射体系

> **目标**：从原始数据中提取所有实体类型，然后基于 `BMES` 标注方案构建一个全局统一的“标签-ID”映射表。
>
> **对应脚本**：`code/C8/01_build_category.py`

### 2.1 设计思路

在开始编码前，我们先梳理一下思路：

1.  **输入**：一个或多个原始数据文件（`.json` 格式）。
2.  **核心逻辑**：
    *   我们需要一个函数，能够读取单个文件，并抽取出其中 `entities` 字段下所有的 `type` 值。
    *   为了防止重复，使用 `set` 数据结构来收集这些实体类型。
    *   遍历所有输入文件（训练集、验证集等），将所有实体类型汇总。
    *   为了保证每次生成的 ID 映射都是固定的（这对于模型复现至关重要），需要对汇总后的实体类型进行**排序**。
    *   创建一个字典，首先放入 `'O': 0` 作为非实体标签。然后遍历排序后的实体类型列表，为每一种类型生成 `B-`、`M-`、`E-`、`S-` 四种标签，并依次赋予递增的整数 ID。
3.  **输出**：一个 `categories.json` 文件，以 JSON 格式存储最终的“标签-ID”映射字典。

### 2.2 核心代码实现

根据以上思路，我们编写 `01_build_category.py` 脚本：

```python:code/C8/01_build_category.py
import json
import os
from collections import Counter


def save_json_pretty(data, file_path):
    """
    将 Python 对象以格式化的 JSON 形式保存到文件。
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def collect_entity_types_from_file(file_path):
    """
    从单个数据文件中提取所有唯一的实体类型。
    """
    types = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # 遍历实体列表，提取 'type' 字段
                for entity in data.get('entities', []):
                    types.add(entity['type'])
            except json.JSONDecodeError:
                print(f"警告: 无法解析行: {line} in {file_path}")
    return types


def generate_tag_map(data_files, output_file):
    """
    从多个数据文件构建并保存一个完整的标签到ID的映射表。
    """
    # 1. 从所有提供的数据文件中提取所有唯一的实体类型
    all_entity_types = set()
    for file_path in data_files:
        all_entity_types.update(collect_entity_types_from_file(file_path))

    # 排序以确保每次生成的映射表顺序一致
    sorted_types = sorted(list(all_entity_types))
    print(f"发现的实体类型: {sorted_types}")

    # 2. 基于BMES模式构建 label2id 映射字典
    tag_to_id = {'O': 0}  # 'O' 代表非实体 (Outside)
    for entity_type in sorted_types:
        for prefix in ['B', 'M', 'E', 'S']:
            tag_name = f"{prefix}-{entity_type}"
            tag_to_id[tag_name] = len(tag_to_id)

    print(f"\n已生成 {len(tag_to_id)} 个标签映射。")

    # 3. 将映射表保存到指定的输出文件
    save_json_pretty(tag_to_id, output_file)
    print(f"标签映射已保存至: {output_file}")


if __name__ == '__main__':
    # 定义输入的数据文件和期望的输出路径
    train_file = './data/CMeEE-V2_train.json'
    dev_file = './data/CMeEE-V2_dev.json'
    output_path = './data/categories.json'

    generate_tag_map(data_files=[train_file, dev_file], output_file=output_path)
```

### 2.3 运行结果

执行此脚本后，会生成 `data/categories.json` 文件，内容如下（部分展示），完美符合我们的预期输出：

```json
{
    "O": 0,
    "B-bod": 1,
    "M-bod": 2,
    "E-bod": 3,
    "S-bod": 4,
    "B-dep": 5,
    "M-dep": 6,
    "E-dep": 7,
    "S-dep": 8,
    "B-dis": 9,
    "M-dis": 10,
    "E-dis": 11,
    "S-dis": 12
}
```

## 三、步骤二：构建词汇表

> **目标**：创建一个“字符-ID”的映射表（即词汇表），为后续将文本转换为数字序列做准备。
>
> **对应脚本**：`code/C8/02_build_vocabulary.py`

### 3.1 设计思路

1.  **问题分析**：
    *   模型无法直接处理文本，需要将字符转换为数字 ID。
    *   原始文本中可能包含语义相同但编码不同的字符，如全角字符（`Ａ`）和半角字符（`A`）。如果不统一，它们会被视为两个不同的 token，这会不必要地增加词汇表大小，并可能影响模型学习。
2.  **核心逻辑**：
    *   首先，需要一个**文本规范化**函数，将所有全角字符统一转换为半角。
    *   遍历所有数据文件，读取每一行文本。
    *   对每一行文本进行规范化处理。
    *   使用 `collections.Counter` 来高效统计所有出现过的字符及其频率。
    *   （可选）可以设置一个频率阈值 `min_freq` 来过滤掉低频或罕见的字符，以减小词汇表规模。
    *   在最终的词汇表前，添加两个特殊的 token：`<PAD>`（用于后续填充）和 `<UNK>`（用于表示未登录词）。
3.  **输出**：一个 `vocabulary.json` 文件，它是一个列表，存储了所有词汇（包括特殊 token）。

### 3.2 核心代码实现

```python:code/C8/02_build_vocabulary.py
import json
import os
from collections import Counter


def save_json_pretty(data, file_path):
    """
    将数据以易于阅读的格式保存为 JSON 文件。
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def normalize_text(text):
    """
    规范化文本，例如将全角字符转换为半角字符。
    """
    full_width = "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ！＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～＂"
    half_width = r"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&'" + r'()*+,-./:;<=>?@[\]^_`{|}~".'
    mapping = str.maketrans(full_width, half_width)
    return text.translate(mapping)


def create_char_vocab(data_files, output_file, min_freq=1):
    """
    从数据文件创建字符级词汇表。
    """
    char_counts = Counter()
    for file_path in data_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = normalize_text(data['text'])
                    char_counts.update(list(text))
                except (json.JSONDecodeError, KeyError):
                    print(f"警告: 无法处理行: {line} in {file_path}")

    # 过滤低频词
    frequent_chars = [char for char, count in char_counts.items() if count >= min_freq]
    
    # 保证每次生成结果一致
    frequent_chars.sort()

    # 添加特殊标记
    special_tokens = ["<PAD>", "<UNK>"]
    final_vocab_list = special_tokens + frequent_chars
    
    print(f"词汇表大小 (min_freq={min_freq}): {len(final_vocab_list)}")

    # 保存词汇表
    save_json_pretty(final_vocab_list, output_file)
    print(f"词汇表已保存至: {output_file}")


if __name__ == '__main__':
    train_file = './data/CMeEE-V2_train.json'
    dev_file = './data/CMeEE-V2_dev.json'
    output_path = './data/vocabulary.json'

    # 设置字符最低频率，1表示包含所有出现过的字符
    create_char_vocab(data_files=[train_file, dev_file], output_file=output_path, min_freq=1)
```

### 3.3 运行结果

执行后，会生成 `data/vocabulary.json` 文件，它是一个列表，索引 `0` 和 `1` 分别是 `<PAD>` 和 `<UNK>`：

```json
[
    "<PAD>",
    "<UNK>",
    " ",
    "!",
    ...
]
```

## 四、步骤三：封装数据加载器

> **目标**：利用前两步生成的映射文件，将原始数据彻底转换为模型可用的、批次化的 PyTorch Tensor。
>
> **对应脚本**：`code/C8/03_data_loader.py`

### 4.1 设计思路

1.  **问题分析**：
    *   我们现在有了原始数据、词汇表和标签映射，如何将它们高效地整合起来？
    *   模型训练时需要以“批次”（batch）为单位输入数据，而不是单条数据。
    *   同一批次内的文本长度往往不同，但输入模型的 Tensor 必须是规整的矩形，如何处理不等长序列？
2.  **核心逻辑（采用 PyTorch 标准实践）**：
    *   **`Vocabulary` 类**：创建一个类来封装词汇表加载和“token-ID”转换的逻辑，使其清晰、可复用。
    *   **`NerDataProcessor` (Dataset) 类**：这是数据处理的核心。它继承 PyTorch 的 `Dataset`，负责：
        *   在 `__init__` 中加载所有原始数据记录。
        *   在 `__getitem__` 中处理**单条**数据：将文本转换为 `token_ids`，并根据实体标注生成 `tag_ids`。
    *   **`create_ner_dataloader` 工厂函数**：这个函数封装了创建 `DataLoader` 的全部逻辑，包括一个非常关键的内部函数 `collate_batch`。
    *   **`collate_batch` 函数**：它负责解决不等长序列的问题。其工作原理是：
        *   接收一个批次的数据（一个由 `__getitem__` 返回的字典组成的列表）。
        *   找到当前批次中最长的序列长度。
        *   使用 `pad_sequence` 函数，将所有序列都填充（pad）到这个最大长度。对于 `token_ids` 使用 `pad_id` (通常是0)，对于 `tag_ids` 使用 `-100`（PyTorch 损失函数会忽略这个值）。
        *   生成一个 `attention_mask`，标记出哪些是真实 token（值为1），哪些是填充 token（值为0），以便模型在计算时忽略填充部分。

### 4.2 核心代码实现

```python:code/C8/03_data_loader.py
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def normalize_text(text):
    """
    规范化文本，例如将全角字符转换为半角字符。
    """
    full_width = "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ！＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～＂"
    half_width = r"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&'" + r'()*+,-./:;<=>?@[\]^_`{|}~".'
    mapping = str.maketrans(full_width, half_width)
    return text.translate(mapping)


class Vocabulary:
    """
    负责管理词汇表和 token 到 id 的映射。
    """
    def __init__(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.tokens = json.load(f)
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.pad_id = self.token_to_id['<PAD>']
        self.unk_id = self.token_to_id['<UNK>']

    def __len__(self):
        return len(self.tokens)

    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]


class NerDataProcessor(Dataset):
    """
    处理 NER 数据，并将其转换为适用于 PyTorch 模型的格式。
    """
    def __init__(self, data_path, vocab: Vocabulary, tag_map: dict):
        self.vocab = vocab
        self.tag_to_id = tag_map
        self.records = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"警告: 无法解析行: {line}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        text = normalize_text(record['text'])
        tokens = list(text)
        
        # 将文本 tokens 转换为 ids
        token_ids = self.vocab.convert_tokens_to_ids(tokens)

        # 初始化标签序列为 'O'
        tags = ['O'] * len(tokens)
        for entity in record.get('entities', []):
            entity_type = entity['type']
            start = entity['start_idx']
            end = entity['end_idx'] - 1  # 转换为包含模式

            if end >= len(tokens): continue

            if start == end:
                tags[start] = f'S-{entity_type}'
            else:
                tags[start] = f'B-{entity_type}'
                tags[end] = f'E-{entity_type}'
                for i in range(start + 1, end):
                    tags[i] = f'M-{entity_type}'
        
        # 将标签转换为 ids
        tag_ids = [self.tag_to_id.get(tag, self.tag_to_id['O']) for tag in tags]

        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "tag_ids": torch.tensor(tag_ids, dtype=torch.long)
        }


def create_ner_dataloader(data_path, vocab, tag_map, batch_size, shuffle=False):
    """
    创建 NER 任务的 DataLoader。
    """
    dataset = NerDataProcessor(data_path, vocab, tag_map)
    
    def collate_batch(batch):
        token_ids_list = [item['token_ids'] for item in batch]
        tag_ids_list = [item['tag_ids'] for item in batch]

        padded_token_ids = pad_sequence(token_ids_list, batch_first=True, padding_value=vocab.pad_id)
        padded_tag_ids = pad_sequence(tag_ids_list, batch_first=True, padding_value=-100) # -100 用于在计算损失时忽略填充部分

        attention_mask = (padded_token_ids != vocab.pad_id).long()

        return padded_token_ids, padded_tag_ids, attention_mask

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)


if __name__ == '__main__':
    # 文件路径
    train_file = './data/CMeEE-V2_train.json'
    vocab_file = './data/vocabulary.json'
    categories_file = './data/categories.json'

    # 1. 加载词汇表和标签映射
    vocabulary = Vocabulary(vocab_path=vocab_file)
    with open(categories_file, 'r', encoding='utf-8') as f:
        tag_map = json.load(f)
    print("词汇表和标签映射加载完成。")

    # 2. 创建 DataLoader
    train_loader = create_ner_dataloader(
        data_path=train_file,
        vocab=vocabulary,
        tag_map=tag_map,
        batch_size=4,
        shuffle=True
    )
    print("DataLoader 创建完成。")

    # 3. 验证一个批次的数据
    print("\n--- 验证一个批次的数据 ---")
    tokens, labels, mask = next(iter(train_loader))
    
    print(f"  Token IDs (shape): {tokens.shape}")
    print(f"  Label IDs (shape): {labels.shape}")
    print(f"  Attention Mask (shape): {mask.shape}")
    print(f"  Token IDs (sample): {tokens[0][:20]}...")
    print(f"  Label IDs (sample): {labels[0][:20]}...")
    print(f"  Attention Mask (sample): {mask[0][:20]}...")
```

### 4.3 运行验证

执行 `03_data_loader.py` 脚本会完整地加载所有数据，并输出一个批次数据的形状和示例，验证整个数据加载流程的正确性。

```
词汇表和标签映射加载完成。
DataLoader 创建完成。

--- 验证一个批次的数据 ---
  Token IDs (shape): torch.Size([4, 152])
  Label IDs (shape): torch.Size([4, 152])
  Attention Mask (shape): torch.Size([4, 152])
  ...
```

至此，我们已经通过三个独立的、流程化的脚本，完成了从原始 JSON 数据到模型可用的、批次化的 PyTorch Tensor 的全部转换工作。

---

> 完整代码请参考：[GitHub 仓库](https://github.com/FutureUnreal/base-nlp/tree/main/code/C8)