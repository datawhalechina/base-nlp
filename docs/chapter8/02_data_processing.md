# 第二节 NER 流程化代码实践

在上一节中，我们简单介绍了命名实体识别的任务定义、应用场景及主流实现方法。本节将正式进入编码阶段，从数据处理开始，逐步构建一个完整的 NER 项目。为了清晰地构建 NER 的处理流程，我们采用流程化的代码组织思路，将整个流程拆分为多个独立的脚本。

## 一、数据处理流程总览

在写代码之前，需要先明确整个数据处理流程的全貌并对数据结构进行拆解。

### 1.1 数据处理的主要任务

针对当前任务数据处理可以分解为以下几个关键问题：

1. **模型的输入是什么？**
   - **X**：`[batch_size, seq_len]` - 每个样本的 Token ID 序列
   - **Y**：`[batch_size, seq_len]` - 每个 Token 对应的标签 ID

2. **如何从原始数据到模型输入？**
   - 分词策略：按**字**切分（每个字就是一个 Token）
   - Token 到 ID：需要构建词汇表
   - 标签到 ID：需要构建标签映射表

3. **输入输出的取值范围？**
   - X 的取值：`[0, vocab_size)` - 词汇表大小
   - Y 的取值：`[0, 1 + 4 × 实体数)` - 标签总数（`O` + `BMES` × 实体类型数）

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

> [完整代码](https://github.com/FutureUnreal/base-nlp/blob/main/code/C8/01_build_category.py)

这是数据处理的第一步。我们需要从原始数据中提取所有实体类型，然后基于 `BMES` 标注方案构建完整的标签映射表。

### 2.1 设计思路

在写代码之前，先明确**输入**和**输出**：

-   **输入**：原始数据文件路径列表（如 `['train.json', 'dev.json']`）
-   **核心逻辑**：遍历所有文件 → 提取实体类型 → 结合 `BMES` 生成映射
-   **输出**：`categories.json` 文件，存储标签到 ID 的映射关系

### 2.2 实现：从文件中提取实体类型

第一步是编写一个函数，从单个数据文件中提取所有唯一的实体类型。

```python
import json

def get_entity_types_from_file(file_path):
    """
    从单个数据文件中提取所有唯一的实体类型。
    """
    entity_types = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # 遍历实体列表，提取 'type' 字段
                for entity in data.get('entities', []):
                    entity_types.add(entity['type'])
            except json.JSONDecodeError:
                print(f"警告: 无法解析行: {line} in {file_path}")
    return entity_types
```

**代码逻辑**：
1. 使用 `set()` 自动去重
2. 逐行读取文件（JSON Lines 格式）
3. 解析每行的 JSON，提取 `entities` 中的 `type` 字段
4. 用 `try-except` 处理可能的解析错误

### 2.3 实现：构建标签映射字典

接下来，处理**多个文件**，合并所有实体类型，并生成完整的标签映射。

```python
import os

def save_json(data, file_path):
    """
    将 Python 对象以格式化的 JSON 形式保存到文件。
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def build_label_map(data_files, output_file):
    """
    从多个数据文件构建并保存一个完整的标签到ID的映射表。
    """
    # 1. 从所有提供的数据文件中提取所有唯一的实体类型
    all_types = set()
    for file_path in data_files:
        all_types.update(get_entity_types_from_file(file_path))

    # 排序以确保每次生成的映射表顺序一致
    sorted_types = sorted(list(all_types))
    print(f"从数据中发现的实体类型: {sorted_types}")

    # 2. 基于BMES模式构建 label2id 映射字典
    label2id = {'O': 0}  # 'O' 代表非实体 (Outside)
    for entity_type in sorted_types:
        for prefix in ['B', 'M', 'E', 'S']:
            label_name = f"{prefix}-{entity_type}"
            label2id[label_name] = len(label2id)

    print(f"\n已生成 {len(label2id)} 个标签的映射关系。")

    # 3. 将映射表保存到指定的输出文件
    save_json(label2id, output_file)
    print(f"标签映射表已保存至: {output_file}")
```

**代码要点**：

1. **合并多个文件**：
   - 使用 `set.update()` 合并所有文件的实体类型
   - 确保训练集和验证集的标签都被包含

2. **排序的重要性**：
   - `sorted(list(all_types))` 保证每次运行的顺序一致
   - 这对于模型的可复现性至关重要

3. **构建映射**：
   - 先初始化 `{'O': 0}`，为非实体标签预留 ID `0`
   - 嵌套循环：外层遍历实体类型，内层遍历 `BMES` 前缀
   - 巧妙利用 `len(label2id)` 实现 ID 自增

4. **保存 JSON**：
   - `ensure_ascii=False` 保证中文正常显示
   - `indent=4` 格式化输出，便于阅读

### 2.4 执行脚本

```python
if __name__ == '__main__':
    # 定义输入的数据文件和期望的输出路径
    train_file = './data/CMeEE-V2_train.json'
    dev_file = './data/CMeEE-V2_dev.json'
    output_path = './data/categories.json'

    build_label_map(data_files=[train_file, dev_file], output_file=output_path)
```

**运行输出**：

```
从数据中发现的实体类型: ['bod', 'dep', 'dis', 'dru', 'equ', 'ite', 'mic', 'pro', 'sym']

已生成 37 个标签的映射关系。
标签映射表已保存至: ./data/categories.json
```

**生成的 `categories.json`（部分展示）**：

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

> **标签数量计算**
>
> 如果数据集中有 $N$ 种实体类型，则总标签数为：
>
> $$\text{Total} = 1 + N \times 4$$
>
> 其中 `1` 是非实体标签 `O`，`N × 4` 是每种实体类型的 `B/M/E/S` 标签。本例中 $N=9$，故总数为 $1 + 9 \times 4 = 37$。

## 三、步骤二：生成 Token 与标签序列

> [完整代码](https://github.com/FutureUnreal/base-nlp/blob/main/code/C8/02_data_loader.py)

有了标签映射表后，就可以开始处理原始文本了。这一步的目标是将文本按字切分，并为每个字生成对应的 `BMES` 标签。

### 3.1 分词策略：按字切分

在中文 NER 任务中，最直接有效的分词方式就是**按字切分**。

```python
text = "胃癌癌前病变和胃癌根治术后患者"
tokens = list(text)
```

执行后，`tokens` 变成：

```python
['胃', '癌', '癌', '前', '病', '变', '和', '胃', '癌', '根', '治', '术', '后', '患', '者']
```

**为什么这样做？**
- 避免传统分词工具可能将实体错误切分
- Token 与原文字符位置完全对应，便于标签对齐
- 与主流预训练模型（如 BERT）的中文处理方式一致

### 3.2 核心逻辑：从实体标注生成 BMES 标签

这是数据处理的核心步骤。根据实体的起止位置，为对应的 Token 分配正确的标签。

```python
def process_ner_data(file_path):
    """
    加载NER数据，处理第一行以生成BMES标签，
    并打印结果用于验证。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # 作为演示，我们只处理文件的第一行
        first_line = f.readline()
        if not first_line:
            print("文件为空。")
            return

        data = json.loads(first_line)

        # 1. 按字切分文本 (Tokenization)
        text = data['text']
        tokens = list(text)

        # 2. 初始化标签列表，全部标记为 "O" (Outside)
        labels = ['O'] * len(tokens)

        # 3. 根据实体标注信息，应用 BMES 标签
        for entity in data.get('entities', []):
            entity_type = entity['type']
            start_idx = entity['start_idx']
            # 注意：原始end_idx不包含，我们将其-1转换为包含模式，便于处理
            end_idx = entity['end_idx'] - 1
            entity_len = end_idx - start_idx + 1

            if entity_len == 1:
                # 单字实体
                labels[start_idx] = f'S-{entity_type}'
            else:
                # 多字实体
                labels[start_idx] = f'B-{entity_type}'
                labels[end_idx] = f'E-{entity_type}'
                for i in range(start_idx + 1, end_idx):
                    labels[i] = f'M-{entity_type}'

        # 4. 打印Token和对应的标签，用于检查逻辑正确性
        print("文本Tokens及其生成的BMES标签:")
        for token, label in zip(tokens, labels):
            print(f"{token}\t{label}")
```

**代码逻辑详解**：

1. **初始化标签序列**：
   ```python
   labels = ['O'] * len(tokens)
   ```
   创建与 Token 序列等长的列表，默认全部为 `'O'`（非实体）

2. **遍历实体标注**：
   ```python
   for entity in data.get('entities', []):
   ```
   处理每个标注的实体

3. **计算实体长度**：
   ```python
   # 注意：原始end_idx不包含，我们将其-1转换为包含模式，便于处理
   end_idx = entity['end_idx'] - 1
   entity_len = end_idx - start_idx + 1
   ```
   我们将不包含的 `end_idx` 减 1，使其变为实体最后一个字符的索引（即包含模式），这样实体长度就可以统一用 `end - start + 1` 计算。

4. **应用 BMES 标签**：
   - **单字实体**（`entity_len == 1`）：
     ```python
     labels[start_idx] = f'S-{entity_type}'
     ```
   - **多字实体**（`entity_len > 1`）：
     ```python
     labels[start_idx] = f'B-{entity_type}'  # Begin
     labels[end_idx] = f'E-{entity_type}'      # End (使用转换后的end_idx)
     for i in range(start_idx + 1, end_idx):
         labels[i] = f'M-{entity_type}'      # Middle
     ```

### 3.3 执行效果

```python
if __name__ == '__main__':
    train_file = './data/CMeEE-V2_train.json'
    print(f"--- 正在处理文件的第一行: {train_file} ---")
    process_ner_data(train_file)
```

**运行输出示例**（部分）：

```
--- 正在处理文件的第一行: ./data/CMeEE-V2_train.json ---
文本Tokens及其生成的BMES标签:
胃	B-dis
癌	M-dis
癌	M-dis
前	M-dis
病	M-dis
变	E-dis
和	O
胃	B-pro
癌	M-pro
根	M-pro
治	M-pro
术	E-pro
后	O
患	O
者	O
```

**结果验证**：
- "胃癌癌前病变"（`dis` 疾病）：`B-dis`, `M-dis`, `M-dis`, `M-dis`, `M-dis`, `E-dis`
- "胃癌根治术"（`pro` 诊疗程序）：`B-pro`, `M-pro`, `M-pro`, `M-pro`, `E-pro`
- 其他字符：`O`

至此，我们成功地将原始的实体标注转换成了与 Token 一一对应的 `BMES` 标签序列！

## 四、后续步骤预告

当前我们已经完成了数据处理的两个核心步骤：

1. ✅ 构建标签映射体系（`categories.json`）
2. ✅ 生成 Token 与标签序列

接下来的工作包括：

### 4.1 步骤三：转换为 ID 序列

将 Token 序列和标签序列转换为模型可用的整数 ID：

-   **Token → ID**：需要构建词汇表（或使用 BERT tokenizer）
-   **Label → ID**：加载 `categories.json` 映射表

### 4.2 步骤四：构建 DataLoader

封装成 PyTorch 的 `Dataset` 和 `DataLoader`：
- 实现批处理（Batch Processing）
- 实现填充（Padding）
- 实现数据增强等

### 4.3 步骤五：模型构建与训练

基于处理好的数据，构建 NER 模型并进行训练。

## 五、关键要点总结

本节通过流程化的方式，完成了 NER 数据处理的前两个关键步骤：

1. **标签映射构建**（`01_build_category.py`）：
   - 扫描所有数据文件，提取唯一的实体类型
   - 通过排序确保映射关系的固定性和可复现性
   - 结合 `BMES` 方案生成完整的标签到 ID 映射表

2. **Token 与标签生成**（`02_data_loader.py`）：
   - 采用按字切分策略，确保边界精确
   - 根据实体起止位置，精确分配 `BMES` 标签
   - 正确处理单字实体和多字实体的不同情况

3. **流程化代码组织**：
   - 每个脚本专注于一个明确的任务
   - 先想清楚输入输出，再动手写代码
   - 通过打印验证每一步的正确性

> **核心思想**
>
> 写代码的本质是：明确**输入** → 设计**转换逻辑** → 得到**输出**。将复杂任务拆解成独立的小步骤，逐一实现并验证，最终组合成完整的流程。

---

> 完整代码请参考：[GitHub 仓库](https://github.com/FutureUnreal/base-nlp/tree/main/code/C8)