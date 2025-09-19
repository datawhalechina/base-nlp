# 第二节 GPT 架构及应用

在 Transformer 催生的众多预训练模型中，除了应用广泛的 BERT 还有另一条技术路线。那就是完全基于 **Transformer 解码器** 构建的 **GPT (Generative Pre-trained Transformer)** 系列模型[^1]。与 BERT 致力于通过双向上下文来“理解”语言不同，GPT 的重心在“生成”语言。它的设计初衷就是为了根据给定的上文，以自回归的方式预测下一个最可能的词元，这种特性使其在文本生成、对话系统、内容续写等任务上展现出无与伦比的能力。从早期的 GPT-1，到引爆全球AI热潮的 ChatGPT (其背后是 GPT-3.5)，OpenAI 沿着解码器这条路，不断刷新着我们对语言模型能力的认知。

## 一、GPT 的设计思想与技术发展

与 BERT 类似，GPT 也遵循 **“预训练 + 微调”** 的思路。不过，随着模型规模的指数级增长，其应用方式发生了变化，从传统的微调逐渐演变为更为灵活的 **提示 (Prompt)**。

### 1.1 因果语言模型

GPT 的成功根植于一个非常简洁而强大的预训练任务——**因果语言模型 (Causal Language Model, CLM)**。通俗地讲，就是 **预测下一个词**。给定一段文本序列 $x_1, x_2, ..., x_t$，模型的目标是最大化预测出下一个词元 $x_{t+1}$ 的概率。它在预训练过程中，不断地在海量文本上重复这个“续写”任务。

$$
P(x_1, ..., x_T) = \prod_{t=1}^{T} P(x_t | x_1, ..., x_{t-1})
$$

这个看似简单的目标，却迫使模型必须学习到语言的深层规律，包括语法结构、事实知识、上下文逻辑关系等，因为只有真正“理解”了前面的内容，才能准确地预测后面会说什么。由于在预测第 `t` 个词时，模型只能看到它前面的 `t-1` 个词，这种单向的、自回归的特性是 GPT 与 BERT 最根本的区别。

### 1.2 从微调到提示

1.  **GPT-1 的微调方法**

    早期的 GPT-1 采用与 BERT 类似的微调模式：在预训练模型的基础上，针对不同的下游任务，添加一个任务相关的输出层，然后在特定任务的数据集上进行训练。依然是“预训练-微调”的传统思路。

2.  **GPT-2 的零样本学习探索**[^2]

    GPT-2 带来了突破性的发现。研究者发现，当模型规模足够大（GPT-2 最大有 15 亿参数）并且在一个更多样化的海量数据集上进行训练后，模型无需任何微调，就能在许多任务上展现出不错的性能。也就是 **零样本学习 (Zero-shot Learning)**。例如，可以直接给模型输入一段指令，如：“将‘今天天气真好’翻译成英文：”，模型就能在没有任何额外训练的情况下，自动续写出 “Today is a nice day.”。GPT-2 证明了，模型能够从海量文本中学习到任务的“模式”，并通过提示词来激活这种能力。

3.  **GPT-3 对少样本学习的完善**[^3]

    GPT-3 将这一思想推向了极致。其模型规模和数据量的爆炸式增长（参数量达 1750 亿），极大地激发了模型的 **上下文学习** 能力。此时，我们不再需要为每个任务都去修改模型结构和重新训练（微调），而是可以将所有任务都统一重构成 **文本生成任务**。特别是其强大的 **少样本学习 (Few-shot Learning)** 能力——在提示词中加入几个示例——使其在很多任务上的表现足以媲美甚至超越经过微调的专用模型。

这种新的方法就是 **提示**。我们通过设计特定的输入文本（即“提示词”），来“引导”或“激发”模型完成我们想要的任务。

-   **文本分类**：
    -   **旧方法 (微调)**: 输入句子 -> `[CLS]`向量 -> 分类头 -> 类别标签
    -   **新方法 (提示)**: 输入一段文本，如：`请判断以下评论的情感类别（正面/负面）："这家餐厅的菜品味道惊艳，服务也很周到。"`，然后让模型续写出 `正面`。
-   **文本相似度**：
    -   **旧方法 (微调)**: 输入句子对 -> `[CLS]`向量 -> 分类头 -> 相似/不相似
    -   **新方法 (提示)**: 输入：`请判断下面两个句子的意思是否相似。句子1："今天天气真好" 句子2："天气晴朗的一天"`，然后让模型续写出 `相似`。

通过这种方式，几乎所有的自然语言处理任务都可以被转换成一个统一的“问答”或“续写”模式，极大地提升了模型的通用性和易用性。

## 二、GPT 架构解析

GPT 的架构本质就是将 Transformer 的解码器模块进行堆叠。我们在 transformer 那节学习的关于解码器的知识，大部分都适用于 GPT。

一个标准的 Transformer 解码器层包含两个核心子层：

1.  **掩码多头自注意力**: 这是实现单向、自回归生成的关键。通过一个“掩码”机制，确保在计算任何一个位置的表示时，只能关注到它左侧（即已经生成）的词元，而不能“看到”未来的信息。
2.  **位置前馈网络**: 与编码器中的结构完全相同，负责对每个位置的表示进行非线性变换。

需要注意的是，原始的 Transformer 解码器还有一个用于与编码器交互的“交叉注意力”层，但由于 GPT 模型完全没有编码器部分，所以 **这一层被移除了**。

### 2.1 输入表示

与 BERT 相似，GPT 的输入也是由两种嵌入向量 **逐元素相加** 而成：

$$
Input_{embedding} = Token_{embedding} + Position_{embedding}
$$

1.  **词元嵌入**:
    -   这是每个词元自身的向量表示。GPT 系列采用 **字节级 BPE (Byte-level BPE)** 的分词方法[^4]。这种方法能有效处理未登录词，但对于像中文这类非核心语言，可能会降低编码效率。详见 4.3 节分词器工作原理。
2.  **位置嵌入**:
    -   与 BERT 类似，GPT 也采用 **可学习的位置嵌入**，让模型在训练过程中自主学习每个位置的向量表示。这也同样决定了模型所能处理的最大序列长度。

> **为什么 GPT 没有片段嵌入？**
>
> BERT 的预训练任务之一是“下一句预测 (NSP)”，需要输入成对的句子，因此引入了片段嵌入来区分句子 A 和句子 B。而 GPT 的预训练任务始终是预测下一个 token，其输入总是一个连续的文本流，不需要区分不同的句子片段。

## 三、GPT 与 BERT 的核心差异总结

尽管都基于 Transformer，但由于选择了编码器和解码器两条不同的路线，BERT 和 GPT 在设计和应用上有着明显区别。

| 特性 | BERT (Encoder) | GPT (Decoder) |
| :--- | :--- | :--- |
| **核心结构** | Transformer 编码器 | Transformer 解码器 |
| **注意力机制** | 标准自注意力 (双向) | 掩码自注意力 (单向) |
| **预训练任务** | 掩码语言模型 (MLM), 下一句预测 (NSP) | 因果语言模型 (CLM) / 预测下一个词元 |
| **输入表示** | 词元 + 位置 + **片段** | 词元 + 位置 |
| **应用模式** | 主要通过微调，添加任务头 | 从微调演变为 **Prompting**，将所有任务转为生成 |
| **分词器** | WordPiece (常用于英文), 字 (常用于中文) | 字节级 BPE (Byte-level BPE) |
| **适用场景** | 语言理解任务 (分类、NER、问答) | 文本生成、对话、续写、摘要 |

## 四、GPT 代码实战

> [本节完整代码](https://github.com/datawhalechina/base-nlp/blob/main/code/C5/02_gpt_usage.py)

为了更好地理解 GPT 模型自回归生成的内部原理，以及 `gpt2` 这类英文预训练模型处理不同语言时的差异，下面的示例将并列展示它处理英文和中文时的不同表现。

### 4.1 代码示例

以下代码将分别对英文和中文进行手动地逐字生成，以观察其内部过程和差异。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. 环境配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

# --- 2. 英文示例 ---
prompt_en = "I like eating fried"
input_ids_en = tokenizer(prompt_en, return_tensors="pt")['input_ids'].to(device)

print(f"英文输入: '{prompt_en}'")
print(f"被编码为 {input_ids_en.shape[1]} 个 token: {tokenizer.convert_ids_to_tokens(input_ids_en[0])}")
print("开始为英文逐个 token 生成...")

generated_ids_en = input_ids_en
with torch.no_grad():
    for i in range(5): # 只生成 5 个 token 作为示例
        outputs = model(generated_ids_en)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        new_token = tokenizer.decode(next_token_id[0])
        print(f"第 {i+1} 步, 生成 token: '{new_token.strip()}'")
        generated_ids_en = torch.cat([generated_ids_en, next_token_id], dim=1)

full_decoded_text_en = tokenizer.decode(generated_ids_en[0], skip_special_tokens=True)
print(f"\n英文最终生成结果: \n'{full_decoded_text_en}'\n")


# --- 3. 中文示例 ---
prompt_zh = "我喜欢吃炸"
input_ids_zh = tokenizer(prompt_zh, return_tensors="pt")['input_ids'].to(device)

print(f"中文输入: '{prompt_zh}'")
print(f"被编码为 {input_ids_zh.shape[1]} 个 token: {tokenizer.convert_ids_to_tokens(input_ids_zh[0])}")
print("开始为中文逐个 token 生成...")

generated_ids_zh = input_ids_zh
with torch.no_grad():
    for i in range(5): # 只生成 5 个 token 作为示例
        outputs = model(generated_ids_zh)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        new_token = tokenizer.decode(next_token_id[0])
        print(f"第 {i+1} 步, 生成 token: '{new_token.strip()}'")
        generated_ids_zh = torch.cat([generated_ids_zh, next_token_id], dim=1)

full_decoded_text_zh = tokenizer.decode(generated_ids_zh[0], skip_special_tokens=True)
print(f"\n中文最终生成结果 (出现乱码是正常现象): \n'{full_decoded_text_zh}'")

```

上面的代码通过并列对比 `gpt2` 模型处理英文和中文时的不同表现，可以得到两个关于原始 GPT2 预训练模型的结论：

1.  **分词效率**：
    -   **英文**: 输入 `"I like eating fried"` 包含 4 个单词，被分词器高效地编码成了 4 个有意义的 token: `['I', 'Ġlike', 'Ġeating', 'Ġfried']` (带 `Ġ` 前缀表示一个词的开始)。
    -   **中文**: 输入 `"我喜欢吃炸"` 同样是 4 个字，但通常会被编码成更多的字节级 token（数量会随 transformers 版本和分词细节略有差异）。这说明 `gpt2` 的分词机制导致了中文编码效率的下降，具体原因详见 4.3 节的分析。

2.  **生成质量**：
    -   **英文**: 对于英文输入，模型能够理解其语义，并生成流畅、有逻辑的续写，例如 `I like eating fried chicken...`。这证明了模型在其“母语”环境下的强大能力。
    -   **中文**: 对于中文输入，由于模型完全不理解这些字节组合的含义，它只能根据在英文世界里学到的统计规律来预测下一个字节，最终呈现为一片乱码。其深层原因同样与分词机制和训练语料有关。

### 4.2 模型结构分解

通过 `print(model)` 可以得到 GPT-2 模型的结构图。

```bash
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D(nf=2304, nx=768)
          (c_proj): Conv1D(nf=768, nx=768)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D(nf=3072, nx=768)
          (c_proj): Conv1D(nf=768, nx=3072)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
```

整个 `GPT2LMHeadModel` 由两大部分组成：核心的 `transformer` 主体 (即 `GPT2Model`)，和顶层的 `lm_head` (语言模型头)。

1.  **`transformer` (GPT2Model)**: 这是模型的主体，负责从输入 token ID 序列中提取深层特征。
    *   **嵌入层 (Embeddings)**:
        *   `(wte): Embedding(50257, 768)`: **词元嵌入 (Word Token Embedding)**。这里的 `50257` 是 GPT-2 的词汇表大小，`768` 是模型的隐藏层维度 H。
        *   `(wpe): Embedding(1024, 768)`: **位置嵌入 (Word Position Embedding)**。这正是在理论部分提到的**可学习的位置嵌入**，其 `[1024, 768]` 的大小也直接解释了为什么标准 GPT-2 模型的最大输入长度是 1024 个词元。
        *   与 BERT 不同，这里没有片段嵌入（`token_type_embeddings`），具体原因已在 2.1 节中阐述。

    *   **解码器堆栈 (Decoder Stack)**:
        *   `(h): ModuleList((0-11): 12 x GPT2Block)`: `ModuleList` 中包含了 12 个完全相同的 `GPT2Block`，即 12 层的 Transformer 解码器。每一层的输出都会作为下一层的输入。
        *   在每一个 `GPT2Block` 内部，都包含了理论中所述的两个核心子层（被 `LayerNorm` 包裹）：
            *   `(attn): GPT2Attention`: **掩码多头自注意力模块**。其内部的 `c_attn` 和 `c_proj` 使用 `Conv1D` 来实现。`Conv1D` 在这里被巧妙地用作全连接层。`c_attn` 一次性将 768 维的输入映射到 2304 维 (768 * 3)，正好对应 Q, K, V 三个矩阵的拼接。`c_proj` 则是注意力机制中的输出映射矩阵 $W^O$。
            *   `(mlp): GPT2MLP`: **位置前馈网络模块**。它同样使用 `Conv1D` 实现。`c_fc` 将维度从 768 **升维** 到 3072，经过激活函数后，`c_proj` 再将维度从 3072 **降维** 回 768。这个“先升维再降维”的结构与 BERT 中的前馈网络完全一致。

2.  **`lm_head` (语言模型头)**:
    *   这是 `GPT2LMHeadModel` 与基础的 `GPT2Model` 的唯一区别，也是它能够生成文本的关键。它是一个简单的线性层（全连接层）。
    *   **功能对比**: 它与 BERT 的 `pooler` 层形成了鲜明对比。BERT 的 `pooler` 层只接收 `[CLS]` 词元的输出，用于 NSP 任务的**句子级别**二分类。而 GPT 的 `lm_head` 接收**序列中每一个位置**的输出向量，并将其从 `768` 维映射到词汇表大小 `50257` 维，得到每个位置上所有词元的 logits 分布，专门用于**词元级别**的“预测下一个词”任务。
    *   **与手动生成的关联**: 在我们手动实现的贪心搜索循环中，`next_token_logits = outputs.logits[:, -1, :]` 这一步，实际上就是 `transformer` 主体输出最后一个位置的 768 维向量后，再通过这个 `lm_head` 转换成 50257 维 logits 的结果。`lm_head` 正是连接模型特征与最终词元预测的桥梁。

### 4.3 分词器工作原理

理解 GPT 与 BERT 在分词器设计上的差异，是理解 GPT 模型特性的一个关键。GPT 使用的 **字节对编码 (Byte-Pair Encoding, BPE)**，特别是**字节级 (Byte-level)** 的实现，使它能够从根本上杜绝 `[UNK]` (未知词) 。

其工作原理如下：

1.  **基础词汇表**：BPE 的基础词汇表是所有的单个字节，即 0-255。这意味着任何文本，无论是什么语言、什么符号，都可以被无损地表示为一个字节序列。
2.  **贪心合并**：在训练分词器时，算法会贪心地、迭代地合并最高频出现的相邻字节对，形成新的、更长的子词（subword），并将其加入词汇表。例如，`t` 和 `h` 经常一起出现，就会被合并成 `th`；之后 `th` 和 `e` 经常一起出现，就可能被合并成 `the`。
3.  **编码过程**：当对新文本进行编码时，会先将其转换为字节序列，然后在这个序列上，贪心地用词汇表里最长的子词来替换对应的字节序列。

这个机制带来了两大特点：

-   **优点：无未知词**
    -   因为最坏情况下，任何在词汇表中找不到的词，都可以被拆分成最基础的单个字节，所以模型永远不会遇到它完全不认识的 token。
    -   通过代码可以直观地看到这一点。对于一个非常生僻的汉字 `龘`，BERT 的分词器可能会直接判为 `[UNK]`，但 GPT-2 的分词器会将其分解为对应的 UTF-8 字节表示。

    ```python
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    text = "一个生僻字：龘"
    tokens = tokenizer.tokenize(text)
    
    # 这类不在词表的字符会被拆成若干字节级 token
    print(tokens)
    ```

-   **缺点：对非核心语言效率低**
    -   `gpt2` 的 BPE 词汇表是为英文构建的，**不包含任何独立的中文词语作为 token**。虽然其字节级机制能表示任意汉字，但它会将这些汉字拆分成多个基础的字节 token。
    -   这导致在处理中文时，一个汉字通常会被拆分成 2-3 个字节级别的 token。这不仅增加了序列的长度，也让模型学习中文语义变得更加困难。如代码示例所示，英文提示词被编码为 10 个 token，而几乎同样长度的中文提示词，则被编码为了 19 个 token，极大地消耗了模型的上下文长度预算。

> **为什么 `gpt2` 模型生成中文会是乱码？**
>
> 这就是由上述的分词机制和模型的英文训练背景共同导致的。
>
> 1.  **分词层面**：模型看到的不是“汉字”，而是一堆无意义的字节组合。
> 2.  **模型层面**：模型是在英文语料上训练的，它只懂得按英文的规律来预测下一个字节。
>
> 最终，模型会续写出一串符合**英文统计规律**但完全不符合**中文编码规则**的字节序列，解码时自然就成了一片乱码。要解决此问题，必须使用在中文语料上预训练的模型，例如 `uer/gpt2-chinese-cluecorpussmall`，这类模型的分词器和模型本身都为中文进行了适配。

### 4.4 使用 pipeline

手动实现循环有助于理解原理，但在实际应用中，`transformers` 库提供了更高阶、更便捷的工具——`pipeline`。它将所有步骤（分词、模型调用、解码）封装在一起，一行代码即可完成文本生成[^5]。下面的示例将用 `pipeline` 快速复现前面的英文生成任务。

```python
# pipeline 应用
from transformers import pipeline

print("\n\n--- Pipeline 快速演示 (英文) ---")
generator = pipeline("text-generation", model=model_name, device=device)
pipeline_outputs = generator("I like eating fried", max_new_tokens=5, num_return_sequences=1)
print(pipeline_outputs[0]['generated_text'])
```

-   `pipeline("text-generation", ...)`: 创建一个专门用于文本生成的 `pipeline` 对象，它自动处理了分词、模型推理和解码的所有细节。
-   `generator(...)`: 仅用一行代码就完成了我们之前手动循环实现的全部功能。其输出结果与我们手动实现的英文生成结果是一致的，这证明了 `pipeline` 是在底层执行了相似的逻辑，但为开发者提供了极其便利的接口。

---

## 参考文献

[^1]: [Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). *Improving Language Understanding by Generative Pre-Training*.]
(https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

[^2]: [Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners*.]
(https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

[^3]: [Brown, T. B., Mann, B., Ryder, N., et al. (2020). *Language Models are Few-Shot Learners*.]
(https://arxiv.org/abs/2005.14165)

[^4]: [Sennrich, R., Haddow, B., & Birch, A. (2015). *Neural Machine Translation of Rare Words with Subword Units*.]
(https://arxiv.org/abs/1508.07909)

[^5]: [Hugging Face. GPT-2 tokenizer and model documentation.]
(https://huggingface.co/docs/transformers/model_doc/gpt2)
