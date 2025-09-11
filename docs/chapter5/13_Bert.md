# 第一节 BERT 架构及应用

在前面的章节中，我们探讨了 Transformer 架构，它的结构是由一个编码器和一个解码器组成，而这两部分内部又分别由 N 个相同的层堆叠而成。Transformer 的提出催生了许多强大的预训练模型。有趣的是，这些后续模型往往只采用了 Transformer 架构的一部分。其中，以 GPT（Generative Pre-trained Transformer）为代表的模型主要 **采用解码器结构**，主要任务是预测下一个词元，这种特性使其天然适用于文本生成任务。而本章的主角 **BERT（Bidirectional Encoder Representations from Transformers）**，则完全基于**编码器结构**构建[^1]。

正如其名，BERT 的核心优势在于其 **双向性**。通过 Transformer 编码器中自注意力机制能够同时关注上下文的特性，BERT 在理解语言的深层语义方面取得了突破性进展。

## 一、BERT 的设计原理与预训练策略

在 BERT 出现之前，像 Word2Vec 这样的模型能够为词语生成一个固定的向量表示（静态词向量），但无法解决一词多义的问题。例如，“破防”在“我出了一件破防装备”和“NLP算法给我学破防了”中的含义完全不同，但在 Word2Vec 中它们的向量是相同的。

BERT 的设计初衷就是为了解决这个问题，它的目标是生成 **动态的、与上下文相关的词向量**。它不仅仅是一个词向量生成工具，更是一个强大的 **预训练语言模型**。其工作范式可以分为两个主要阶段：

1.  **预训练 (Pre-training)**
    -   在一个庞大的、通用的文本语料库（如维基百科、书籍）上，通过特定的无监督任务来训练一个深度神经网络模型。
    -   这个阶段的目标不是为了完成某个具体的 NLP 任务，而是让模型学习语言本身的规律，比如语法结构、词语间的语义关系、上下文依赖等。
    -   训练完成后，就得到了一个包含了丰富语言知识的、参数已经训练好的 **预训练模型**。

2.  **微调 (Fine-tuning)**
    -   当面临一个具体的下游任务时（如文本分类、命名实体识别），我们不再从头开始训练模型。而是把预训练好的 BERT 模型作为任务模型的 **基础结构**。加载它已经学习到的所有参数作为 **初始值**。 
    -   接着根据具体任务，在 BERT 模型之上增加一个小的、任务相关的输出层（例如，一个用于分类的全连接层）。
    -   最后在自己的任务数据集上对整个模型（或仅仅是顶部的输出层）进行训练。由于模型已经具备了强大的语言理解能力，这个微调过程通常非常快速，并且只需要相对较少的数据就能达到很好的效果。

这种 **“预训练 + 微调”** 的训练范式，属于 **迁移学习** 的一种实现，也是 BERT 的训练框架。它使得我们能够将从海量数据中学到的通用语言知识，迁移到数据量有限的特定任务中。

> **与 RNN/LSTM 的区别**
>
> 你可能会有这样的疑问：像 Bi-LSTM 这样的循环神经网络不也能捕捉上下文信息，生成动态词向量吗，为什么还需要 BERT？
>
> 虽然目标相似，但实现方式和效果却有很大的差别。Bi-LSTM 的“双向”本质上是两个独立的单向 RNN（一个正向，一个反向）的 **浅层拼接**。在计算过程中，正向 RNN 并不知道未来的信息，反向 RNN 也不知道过去的信息。
>
> 而 BERT 基于 Transformer 的**自注意力机制**，实现了**真正的“深度双向”**。也就是上一节中所学到的，自注意力机制使得在模型的每一层，计算任何一个词的表示时，都能**同时与序列中的所有其他词直接交互**。这种全局视野使得 BERT 能够捕捉到比 RNN 更复杂、更长距离的依赖关系，并且其并行计算的特性也远比 RNN 的串行结构高效。可以说，BERT 实现了 RNN 想要达到但未能完美实现的目标。

## 二、BERT 架构详解

BERT 的整体架构非常简洁，它就是将 Transformer 的编码器部分进行堆叠。我们在上节学到的所有关于编码器的知识，都适用于 BERT。

一个标准的 Transformer 编码器层包含两个核心子层：
1.  **多头自注意力层**：负责捕捉输入序列中不同词元之间的依赖关系，是实现双向理解的核心。
2.  **位置前馈网络**：在自注意力层之后对每个位置的表示进行非线性变换，以提取更深层次的特征。

此外，每个子层都包裹在残差连接和层归一化（Add & Norm）结构中。BERT 模型就是将这个编码器层重复堆叠 N 次。

### 2.1 BERT 的模型规模

与 Transformer 论文中 N=6 不同，BERT 提供了几种不同规模的预训练模型，以适应不同的计算资源和性能需求。其中最常见的两个是：

| 模型          | 层数 (L) | 隐藏层大小 (H) | 注意力头数 (A) | 总参数量 |
| :------------ | :------: | :------------: | :------------: | :------: |
| `BERT-Base`   |    12    |      768       |       12       | ~1.1 亿  |
| `BERT-Large`  |    24    |      1024      |       16       | ~3.4 亿  |

通过加深、加宽网络，BERT 拥有了比原始 Transformer 更强大的特征提取和表示能力。

### 2.2 BERT 的输入表示

为了让模型能够处理各种复杂的输入，BERT 的输入表示由三个部分的嵌入向量 **逐元素相加** 而成：

$$
Input_{embedding} = Token_{embedding} + Position_{embedding} + Segment_{embedding}
$$

对应的结构示意见图 13.1。

<div align="center">
  <img src="images/13_2_2.svg" alt="BERT 输入表示" />
  <p>图 13.1: BERT 输入表示</p>
</div>

1.  **词元嵌入**:
    -   与之前模型类似，这是每个词元（Token）自身的向量表示。
    -   BERT 使用一种称为 **WordPiece** 的分词方法[^2]。对于英文，它能将不常见的词拆分成更小的子词单元（如 "studying" -> "study", "##ing"），有效处理了未登录词问题。对于中文，可以简单地将其理解为 **字向量**，即每个汉字是一个 Token。

2.  **片段嵌入**:
    -   这是 BERT 为了处理句子对任务（如判断两个句子是否是连续的）而引入的，用于区分输入中的不同句子。
    -   例如，对于一个由句子 A 和句子 B 拼接而成的输入，所有属于句子 A 的词元都会加上一个相同的“句子A嵌入”，而所有属于句子 B 的词元则会加上另一个“句子B嵌入”。

3.  **位置嵌入**:
    -   由于 Transformer 的自注意力机制本身不包含位置信息，必须额外引入位置编码来告诉模型每个词元在序列中的位置。
    -   与原始 Transformer 使用固定的正余弦函数不同，BERT 采用的是 **可学习的位置嵌入**。即创建一个大小为 `[max_position_embeddings, hidden_size]` 的嵌入表，让模型在预训练过程中自己学习每个位置的最佳向量表示。
    -   **BERT 的最大长度限制**：正是这个可学习的位置嵌入表，决定了 BERT 的最大输入长度。例如，在 `BERT-Base` 模型中，这个嵌入表的大小通常是 `[512, 768]`，这意味着模型最多只能处理 512 个词元的序列。这并非 Transformer 架构本身的限制，而是 BERT 预训练时设定的一个参数。

最终，每个输入词元的表示向量，是上述三种嵌入向量相加的结果。

### 2.3 特殊词元

BERT 在输入序列中引入了几个特殊的词元，它们在预训练和微调阶段扮演着重要的角色：

-   **`[CLS]` (Classification)**:
    -   这个词元被添加到 **每个输入序列的开头**。
    -   BERT 的一个特殊设计是，这个 `[CLS]` 词元在经过整个编码器网络后的最终输出向量，被视为整个输入序列的 **聚合表示**。
    -   因此，在进行文本分类、情感分析等句子级别的任务时，通常只取用 `[CLS]` 对应的输出向量，然后将其送入一个分类器。

-   **`[SEP]` (Separator)**:
    -   这个词元用于 **分隔不同的句子**。
    -   例如，在输入是单个句子时，它被添加在句子的末尾。
    -   在输入是句子对时，它被用在两个句子之间，以及整个序列的末尾。

-   **`[MASK]`**:
    -   这个词元仅在 **预训练阶段** 使用，用于“掩盖”掉输入序列中的某些词元，是“掩码语言模型”任务的核心。

## 三、BERT 的预训练任务

BERT 的训练过程引入了两项全新的预训练任务，这也是它成功的关键。

### 3.1 任务一：掩码语言模型 (Masked Language Model, MLM)

MLM 的思路是在输入文本中随机遮盖掉一部分词元（Token），然后训练模型去**根据上下文预测这些被遮盖的词元**。这就像在做“完形填空”一样，迫使模型学习词元之间深层次的语义关系和句法结构。

MLM 的执行策略如下：

1.  **随机选择**：在每一个训练序列中，随机挑选 **15%** 的词元作为预测目标。
2.  **特殊替换策略**：对于这 15% 被选中的词元，并非简单地全部替换为 `[MASK]` 标记，同时还采用“80/10/10”的划分方法，以缓解**预训练**与**微调**阶段的数据差异：
    *   **80% 的情况**：将选中的词元替换为 `[MASK]` 标记。
        *   **示例**：`我 爱 吃 [MASK] 果`
        *   **目的**：这是最核心的操作，强制模型利用上下文信息来预测被“挖空”的词。
    *   **10% 的情况**：将选中的词元替换成一个**随机**的其他词元。
        *   **示例**：`我 爱 吃 [闲] 果`
        *   **目的**：这相当于引入了噪声，要求模型不仅要理解上下文，还要具备识别并纠正错误词元的能力，从而增强模型的鲁棒性。
    *   **10% 的情况**：**保持词元不变**。
        *   **示例**：`我 爱 吃 [苹] 果`
        *   **目的**：这是为了解决**预训练-微调不一致**的问题。因为在微调阶段，输入中并没有 `[MASK]` 标记。通过让模型在看到真实词元时也去预测它自己，可以使得模型更好地处理和学习每一个真实词元的上下文表示，从而更好地泛化到下游任务。

通过这种方式，MLM 任务促使 BERT 学习到一种**动态的、与上下文深度融合的词向量表示**。

### 3.2 任务二：下一句预测 (Next Sentence Prediction, NSP)

NSP 任务的目标是让模型理解句子与句子之间的逻辑关系。在训练时，模型会接收一对句子 A 和 B，并判断句子 B 是否是句子 A 在原文中的下一句。

1.  **构造句子对**: 在预训练时，为模型准备句子对 (A, B)。其中，50% 的情况下，B 是 A 的真实下一句；另外 50% 的情况下，B 是从语料库中随机选择的一个句子。
2.  **二分类任务**: 模型需要预测 B 是否是 A 的下一句。

这个任务的预测，正是通过之前提到的 `[CLS]` 词元的输出来完成的。将 `[CLS]` 的最终隐藏状态送入一个二分类器，来判断 `IsNext` 还是 `NotNext`。

通过 NSP 任务的训练，模型被“强迫”将判断两个句子关系所需的全局信息都汇聚到 `[CLS]` 这个词元的表示中。因此，`[CLS]` 的输出向量就成了一个优秀的、代表整个输入序列的句子级别特征。

> 说明：后续研究对 NSP 的有效性提出了质疑。例如，RoBERTa[^3] 移除了 NSP 并通过更大的批量、更长的训练和更好的数据处理取得了更优结果；而 ALBERT[^4] 则以 **SOP（Sentence Order Prediction）** 任务替代 NSP，强调句子顺序的一致性。在实际应用中，是否包含 NSP 取决于所选预训练模型及任务需求。

## 四、BERT 的应用与实践

预训练完成后，我们就得到了一个强大的 BERT 模型。接下来，可以根据具体的任务，通过微调来释放它的能力。

### 4.1 微调下游任务

#### 4.1.1 文本分类任务

对于文本分类任务（如情感分析、意图识别），利用 `[CLS]` 词元的聚合表示能力。

1.  **输入构造**: 将单个句子或句子对按照 BERT 的要求格式化（添加 `[CLS]` 和 `[SEP]`）。
2.  **获取表示**: 将输入送入 BERT 模型，提取出 `[CLS]` 词元对应的最终输出向量（大小为 `hidden_size`）。
3.  **添加分类头**: 在这个向量之上，添加一个简单的全连接层作为分类器，其输出维度等于任务的类别数量。
4.  **微调**: 在任务数据上，训练这个新添加的分类器，同时对 BERT 模型的参数进行微调（通常使用比预训练时小得多的学习率）。

#### 4.1.2 词元分类任务

对于词元级别的任务（如命名实体识别、分词、词性标注），需要为输入序列中的每一个词元进行分类。

1.  **输入构造**: 与文本分类类似，将序列格式化。
2.  **获取表示**: 将输入送入 BERT 模型，但这次提取 **所有词元** 对应的最终输出向量序列。输出的形状为 `(batch_size, sequence_length, hidden_size)`。
3.  **添加分类头**: 在这个序列之上，添加一个全连接层（可以看作是在时间维度上共享权重），将每个词元的 `hidden_size` 维向量映射到类别数量的维度。
4.  **微调**: 在任务数据上进行端到端的训练和微调。

#### 4.1.3 其他任务

BERT 的应用非常广泛，几乎可以适配所有的 NLP 任务。例如，在问答任务中，可以将问题和段落作为两个句子输入 BERT，然后训练模型去预测答案在段落中的起始和结束位置。

### 4.2 实践技巧与生态

#### 4.2.1 特征提取策略

虽然我们通常使用最后一层的输出作为词元或句子的特征表示，但这并非唯一选择。研究和实践表明，BERT 的不同层级学习到的特征有所侧重：

-   **底层** 更偏向于捕捉 **词法、语法** 等表层信息。
-   **高层** 更偏向于捕捉 **语义、语境** 等深层信息。

在某些任务（如命名实体识别）中，将最后几层（例如，最后四层）的向量进行拼接（Concatenate）或相加（Sum），有时能获得比单独使用最后一层更好的效果。这提供了一种简单有效的性能提升技巧。

#### 4.2.2 Hugging Face Transformers 生态

如今，从头实现或手动管理 BERT 模型已无必要。**Hugging Face** 公司开源的 `transformers` 库[^5]已经成为 NLP 领域的标准工具。

-   **模型中心**: 它提供了一个庞大的模型仓库，包含了几乎所有主流的预训练模型（包括各种语言、各种规模的 BERT）。用户可以在 [Hugging Face 官网](https://huggingface.co/models) 或其国内镜像上查找、下载和试用这些模型。
-   **统一的 API**: 开发者可以使用简洁、统一的接口，轻松地加载、使用和微调这些模型。
-   **社区支持**: 拥有活跃的社区和丰富的文档，是学习和应用 BERT 等预训练模型的最佳起点。

## 五、BERT 代码实战

> [本节完整代码](https://github.com/datawhalechina/base-nlp/blob/main/code/C5/01_bert_usage.py)

下面来通过一个完整的示例，展示如何使用 `transformers` 库加载预训练的 BERT 模型，并从中提取文本特征向量。

### 5.1 环境准备

首先，确保你已经安装了 `transformers`：

```bash
pip install transformers
```

如果在国内下载模型遇到网络问题，可以设置环境变量来使用 Hugging Face 的国内镜像：

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### 5.2 代码示例

以下代码涵盖了从加载模型到提取特征的完整流程：

```python
import torch
import os
from transformers import AutoTokenizer, AutoModel

# 1. 环境和模型配置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # 可选：设置镜像
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-chinese"
texts = ["我来自中国", "我喜欢自然语言处理"]

# 2. 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

print("\\n--- BERT 模型结构 ---")
print(model)

# 3. 文本预处理
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

# 打印 Tokenizer 的完整输出，以理解其内部结构
print("--- Tokenizer 输出 ---")
for key, value in inputs.items():
    print(f"{key}: \n{value}\n")

# 4. 模型推理
with torch.no_grad():
    outputs = model(**inputs)

# 5. 提取特征
last_hidden_state = outputs.last_hidden_state
sentence_features_pooler = getattr(outputs, "pooler_output", None)

# (1) 提取句子级别的特征向量 ([CLS] token)
sentence_features = last_hidden_state[:, 0, :]

# (2) 提取第一个句子的词元级别特征
first_sentence_tokens = last_hidden_state[0, 1:6, :]


print("\n--- 特征提取结果 ---")
print(f"句子特征 shape: {sentence_features.shape}")
if sentence_features_pooler is not None:
    print(f"pooler_output shape: {sentence_features_pooler.shape}")
print(f"第一个句子的词元特征 shape: {first_sentence_tokens.shape}")
```

**代码解析：**

1.  **加载模型和分词器**:
    -   使用 `AutoTokenizer.from_pretrained(model_name)` 和 `AutoModel.from_pretrained(model_name)` 来自动下载并加载指定的预训练模型。
2.  **文本预处理**:
    -   `tokenizer(...)` 函数的输出是一个字典，其中包含了模型所需的全部输入信息：
        -   `input_ids`: 这是最重要的部分，是文本被转换为的 token ID 序列。Tokenizer 会在合适的位置自动添加 `101` (`[CLS]`) 和 `102` (`[SEP]`) 等特殊标记；批次中较短的句子会被 `0` (`[PAD]`) 填充到与最长序列等长（或到指定的 `max_length`）。
        -   `token_type_ids`: 这就是 **片段嵌入** 的体现，用于区分不同的句子。在这里因为每个样本都只有一个句子，所以值全为 0。
        -   `attention_mask`: 这是一个非常重要的参数，它告诉模型在进行自注意力计算时，哪些 token 是真实的（值为 1），哪些是填充的（值为 0），模型会忽略填充 token。
3.  **模型推理**:
    -   将预处理好的 `inputs` 字典通过 `**inputs` 解包后送入模型。
    -   `with torch.no_grad():` 上下文管理器会禁用梯度计算，用于在推理阶段减少内存消耗并加速计算。
4.  **解析与提取**:
    -   `outputs.last_hidden_state`: 它的形状是 `(batch_size, sequence_length, hidden_size)`，包含了序列中**每一个词元**在最高层的向量表示。
    -   **句子特征提取**: 通过索引 `[:, 0, :]`，可以轻松获得整个批次的 `[CLS]` 位置的隐藏状态；此外，`outputs.pooler_output`（若存在）是对该隐藏状态再经过一层全连接+Tanh 的结果（BERT 原用于 NSP 任务）。具体使用哪一种，建议以验证集效果为准，平均池化或拼接最后几层在实践中也常见。
    -   **词元特征提取**: 通过对特定范围进行切片可以获得具体某个句子的所有**非特殊词元**的特征向量。例如，对于第一个句子 "我来自中国"，Tokenizer 会将其转换为 `['[CLS]', '我', '来', '自', '中', '国', '[SEP]']`。因此，我们使用 `[0, 1:6, :]` 来提取索引从 1 到 5 的词元向量，这对应了 "我" 到 "国" 这五个汉字。这些特征可以用于命名实体识别等词元级任务。

### 5.3 模型结构分解

当运行 `print(model)` 时，会得到一个详细的、树状的 BERT 模型结构图，如下所示。

```bash
BertModel(
  (embeddings): BertEmbeddings(
    (word_embeddings): Embedding(21128, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (encoder): BertEncoder(
    (layer): ModuleList(
      (0-11): 12 x BertLayer(
        (attention): BertAttention(
          (self): BertSdpaSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
  (pooler): BertPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
)
```

这为我们提供了一个直观的方式来理解其内部组件，并将其与理论知识对应起来。下面来结合这个结构，对 `bert--base-chinese` 模型的核心部分进行分解：

1.  **`embeddings` (嵌入层)**: 此模块是上文 **BERT 输入表示** 理论的具体实现。它负责将输入的 token ID 序列，通过组合 **词元、位置和片段** 三种嵌入向量，转换为模型真正的输入。
    *   `(word_embeddings): Embedding(21128, 768)`: **词元嵌入**。这里的 `21128` 是 `bert-base-chinese` 模型的词汇表大小，`768` 则是 BERT-Base 模型的隐藏层维度 H。
    *   `(position_embeddings): Embedding(512, 768)`: **位置嵌入**。这正是在理论部分提到的**可学习的位置嵌入**，其 `[512, 768]` 的大小也直接解释了为什么 BERT-Base 模型的最大输入长度是 512 个词元。
    *   `(token_type_embeddings): Embedding(2, 768)`: **片段嵌入**。它用于区分输入的两个不同句子（句子 A 和 B），这对于 NSP 这样的预训练任务至关重要。
    *   `(LayerNorm)` 和 `(dropout)`: 在将上述三种嵌入向量相加后，会进行层归一化和 Dropout 操作，以稳定训练过程并增强模型的泛化能力。

    > 可以打开 [bert-base-chinese 的 `vocab.txt` 词汇表文件](https://huggingface.co/google-bert/bert-base-chinese/blob/main/vocab.txt) 验证一下。该文件共有 21128 行，每一行代表一个词元，词汇表大小正好是 `21128`。

    > 直接浏览器查看可能会导致浏览器卡死，别问笔者怎么知道的🫠

2.  **`encoder` (编码器)**: 这是 BERT 的核心主体，正是由前面提到的 **12 层 Transformer 编码器** 堆叠而成。
    *   `(layer): ModuleList((0-11): 12 x BertLayer)`: `ModuleList` 中包含了 12 个完全相同的 `BertLayer`。模型的“深度”就体现在这里，每一层的输出都会作为下一层的输入，逐层提取更深层次的特征。
    *   在每一个 `BertLayer` 内部，都包含了理论中所述的两个核心子层：
        *   `(attention)`: **多头自注意力模块**。
            *   其内部的 `query`, `key`, `value` 线性层负责将输入序列映射成 Q, K, V 矩阵。
            *   `(output)` 中的 `dense` 层则对应注意力机制中的 $W^O$ 矩阵，负责将多头注意力的输出结果重新组合并进行线性变换。
        *   `(intermediate)` 和 `(output)`: **位置前馈网络模块**。
            *   `(intermediate)` 中的 `dense` 层将维度从 768 **升维** 到 3072。
            *   `(output)` 中的 `dense` 层再将维度从 3072 **降维** 回 768。这个“先升维再降维”的结构，就是为了提取更丰富的特征，并让模型学习保留最重要的信息，与老师在课程中所讲完全一致。

3.  **`pooler` (池化层)**:
    *   该层功能与前面介绍的特殊词元 **`[CLS]`** 紧密相关。
    *   它的作用是处理 `[CLS]` 词元在经过 12 层 Encoder 后的最终输出向量（即 `last_hidden_state[:, 0]`），通过一个全连接层和 `Tanh` 激活函数，将其转换为一个代表整个序列的“池化”后的特征向量。
    *   这个经过特殊处理的 `[CLS]` 向量，在预训练阶段专门用于完成 **NSP** 任务，从而使其学习到了整个输入序列的聚合信息。

通过这个结构，可以清晰地看到 BERT 是如何将 Transformer 编码器的思想付诸实践的，从输入处理到多层特征提取，再到最终的输出，每一步都清晰可见。

## 练习

- 总结 BERT 和 Transformer(Encoder) 区别

---

## 参考文献

[^1]: [Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.]
(https://arxiv.org/abs/1810.04805)

[^2]: [Wu, Y., Schuster, M., Chen, Z., et al. (2016). *Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation*.]
(https://arxiv.org/abs/1609.08144)

[^3]: [Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*.]
(https://arxiv.org/abs/1907.11692)

[^4]: [Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., Soricut, R. (2019). *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*.]
(https://arxiv.org/abs/1909.11942)

[^5]: [Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Brew, J. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP 2020 System Demonstrations.]
(https://arxiv.org/abs/1910.03771)
