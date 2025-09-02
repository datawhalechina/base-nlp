# 第二节 注意力机制

在上一节的结尾，讨论了标准 Seq2Seq 架构存在的一个核心缺陷：**信息瓶颈**。编码器需要将源序列的所有信息，不论长短，全部压缩成一个固定长度的上下文向量 $C$。这种机制在处理长序列时，很容易丢失序列开头的关键信息，同时也无法让解码器在生成不同词元时，有选择性地关注输入的不同部分。

用上一节提到的对联任务举例，当上联是“两个黄鹂鸣翠柳”时，期望模型在生成下联时：
-   生成第一个词“一行”时，主要关注上联的“两个”。
-   生成第二个词“白鹭”时，主要关注上联的“黄鹂”。
-   ...

然而，标准的 Seq2Seq 架构的模型在生成“一行”、“白鹭”、“上青天”的每一个词时，所依赖的都是**同一个、包含了整个上联概要**的上下文向量 $C$。模型缺乏一种动态的、有倾向性的“关注”能力，这极大地限制了其性能。为了解决这个问题，**注意力机制 (Attention Mechanism)**[^1]被提出。

## 一、注意力机制的设计原理

注意力机制的原理，可以通俗地理解为从“一言以蔽之”到“择其要者而观之”的转变。人类在进行阅读理解或翻译时，并不会将整个句子或段落的信息平均地记在脑海里。当回答特定问题或翻译特定词组时，我们的注意力会自然地聚焦到原文中的相关部分。

注意力机制正是对这种认知行为的模拟。其原理是：**在解码器生成每一个词元时，不再依赖一个固定的上下文向量，而是允许它“回头看”一遍完整的输入序列，并根据当前解码的需求，自主地为输入序列的每个部分分配不同的注意力权重，然后基于这些权重将输入信息加权求和，生成一个动态的、专属当前时间步的上下文向量。**

通过这种方式，模型便获得了“择其要者而观之”的能力：
-   在生成“一行”时，模型可以学会将最大的权重分配给“两个”所对应的编码器状态。
-   在生成“白鹭”时，则将最大的权重分配给“黄鹂”所对应的状态。

这个动态计算的权重，就是**注意力权重**；而整个动态计算上下文向量的过程，就是**注意力机制**。

## 二、注意力机制的动机与推导

为了更直观地理解注意力机制的必要性，可以跟随一个逐步深入的思路。

### 2.1 问题的根源与固定对齐策略的局限

标准的 Seq2Seq 模型之所以表现不佳，根源在于它试图将源序列的所有信息**无差别**地压缩进一个向量。但在对联这类任务中，输入和输出之间存在着明显的**局部对应关系**。

一个直观的想法是：能不能建立一种固定的对齐策略？例如，在生成下联第一个词时，就只使用上联第一个词的编码信息；生成第二个词时，就只用第二个词的信息，以此类推。

这个想法可以表示为：
-   $c_1 = h_1$   (生成第一个词的上下文是第一个编码状态)
-   $c_2 = h_2$   (生成第二个词的上下文是第二个编码状态)
-   ...

这种方法在处理像对联这样长度相等、词序对应的“特例”时似乎是可行的。但它的局限性非常明显：
1.  **要求序列等长**：对于不等长的序列（如中英文翻译），这种一对一的映射关系立刻失效。
2.  **对齐关系僵化**：它假设了输入和输出的对齐关系是固定不变的，但实际任务中的对应关系可能非常复杂（如一对多、多对一）。

这种固定对齐策略过于理想化，缺乏通用性。我们需要一种更灵活、更具普适性的方法。

### 2.2 注意力机制的动态加权原理

既然只取一个输入信息过于绝对，那么退一步，是否可以把所有输入信息都利用起来，但给它们分配不同的“重要性”呢？这就是通过动态加权进行对齐的思想，即**加权求和**。

我们可以为解码的第 $t$ 步，动态地计算一个上下文向量 $c_t$，它由编码器**所有**的隐藏状态 $(h_1, h_2, \dots, h_{T_x})$ 加权求和得到：

$$
c_t = \sum_{j=1}^{T_x} \alpha_{tj} h_j
$$

其中，$\alpha_{tj}$ 就是在解码第 $t$ 个词时，分配给输入第 $j$ 个词的**注意力权重**。

在这个框架下，前面提到的“固定对齐策略”可以看作是它的一个特例。例如，当 $\alpha_{11}=1$ 且其他所有 $\alpha_{1j}=0$ 时，就实现了 $c_1 = h_1$ 的效果。

### 2.3 如何确定权重？

加权求和的思路虽然灵活，但它引入了一个新的、至关重要的问题：**权重 $\alpha_{tj}$ 从何而来？**

这个权重显然不能是固定的。它必须是**动态的**，应该根据当前的解码需求来决定。例如，当解码器正要生成与“黄鹂”对应的词时，权重 $\alpha_{t, \text{黄鹂}}$ 就应该最大。

这意味着，需要一个额外的模块或机制，它能够：
1.  审视当前解码器的状态（例如，解码器上一时刻的隐藏状态 $h^{\prime}_{t-1}$）。
2.  将这个状态与编码器的每一个隐藏状态 $h_j$ 进行比较。
3.  根据比较结果，生成一组相应的权重 $(\alpha_{t1}, \alpha_{t2}, \dots, \alpha_{t,T_x})$。

让模型**自行学习**如何根据当前上下文来计算这组权重，正是注意力机制的关键。

## 三、注意力机制详解

带有注意力机制的 Encoder-Decoder 模型，其整体结构与标准 Seq2Seq 类似，主要区别在于解码器部分。编码器的工作保持不变，但有一个关键点：它需要向解码器提供**所有时间步**的隐藏状态序列 $(h_1, h_2, \dots, h_{T_x})$，而不仅仅是最后一个时间步的状态。

解码器在生成第 $t$ 个目标词元 $y_t$ 时，会通过三步进行“注意力计算”，来动态生成该时刻的上下文向量 $c_t$。这个过程通常以上一时刻的解码器隐藏状态 $h^{\prime}_{t-1}$ 为起点。

### 3.1 注意力计算三部曲

**第一步：计算相似度**

使用解码器上一时刻的隐藏状态 $h^{\prime}_{t-1}$ 与编码器的每一个隐藏状态 $h_j$ 计算一个分数，这个分数衡量了在当前解码时刻，应当对第 $j$ 个输入词元投入多少“关注”。

$$
e_{tj} = \text{score}(h^{\prime}_{t-1}, h_j)
$$

这个分数越高，代表关联性越强。计算这个分数的方式有很多种，例如简单的点积、或者引入一个可学习的神经网络层。

**第二步：计算注意力权重**

得到输入序列所有位置的注意力分数 $(e_{t1}, e_{t2}, \dots, e_{t,T_x})$ 后，为了将它们转换成一种“权重”的表示，可使用 **Softmax** 函数对其进行归一化。这样，就能得到一组总和为 1、且均为正数的注意力权重 $(\alpha_{t1}, \alpha_{t2}, \dots, \alpha_{t,T_x})$。

$$
\alpha_{tj} = \text{softmax}(e_{tj}) = \frac{\exp(e_{tj})}{\sum_{i=1}^{T_x} \exp(e_{ti})}
$$

这组权重 $\alpha_t$ 构成了一个概率分布，清晰地表明了在当前解码步骤 $t$，注意力应该如何分配在输入序列的各个位置上。

**第三步：加权求和，生成上下文向量**

最后，使用上一步得到的注意力权重 $\alpha_{tj}$，对编码器的所有隐藏状态 $h_j$ 进行加权求和，从而得到当前解码时刻 $t$ 专属的上下文向量 $c_t$。

$$
c_t = \sum_{j=1}^{T_x} \alpha_{tj} h_j
$$

这个 $c_t$ 向量，由于是根据当前解码需求动态生成的，它比原始 Seq2Seq 的那个固定向量 $C$ 包含了更具针对性的信息。

### 3.2 结合上下文进行预测

得到动态上下文向量 $c_t$ 后，模型会将其与当前解码器自身的输入词元 $y_{t-1}$ 的词嵌入结合起来（最常见的方式是将两者**拼接**），形成一个新的、信息更丰富的向量。

最后，将这个拼接后的向量连同上一时刻的状态 $h^{\prime}_{t-1}$ 一起送入解码器的 RNN 单元，计算出当前时刻的状态 $h^{\prime}_{t}$，并基于 $h^{\prime}_{t}$ 预测出最有可能的输出词元 $y_t$。

整个过程可以用下图来概括：

![Attention 工作流程](./images/attention_Flow.svg)

### 3.3 一种高效的注意力打分函数

计算相关性分数的函数有多种设计，其中一种非常高效且影响深远的方法，是直接计算查询向量（$h'_{t-1}$）和键向量（$h_j$）的**点积**，并对其进行**缩放**。这种思想也是后续通用注意力框架的核心。

其计算方式非常简洁：

$$
\text{score}(h^{\prime}_{t-1}, h_j) = \frac{{h^{\prime}_{t-1}}^T \cdot h_j}{\sqrt{d_k}}
$$

这里：
-   ${h^{\prime}_{t-1}}^T \cdot h_j$ 就是两个向量的**点积**。点积是衡量向量相似度的一种有效方式。
-   $d_k$ 是键向量（在这里是编码器隐藏状态）的维度。
-   **除以 $\sqrt{d_k}$** 是一个关键的**缩放**步骤。需要注意的是，当向量维度 $d_k$ 很大时，点积的结果的方差也会很大，这可能导致一些维度的值非常大，从而将 Softmax 函数推向其梯度极小的区域（即概率值极端地趋近于 0 或 1），造成梯度消失，使模型难以训练。通过除以 $\sqrt{d_k}$ 进行缩放，可以有效地缓解这个问题，使训练过程更加稳定。

### 3.4 注意力机制的价值

引入注意力机制，不仅仅是对 Seq2Seq 架构的一个小修补，它带来了一个全新的视角。

**1. 解决信息瓶颈，提升性能**
最直接的好处是，注意力机制彻底打破了信息必须被压缩成一个固定长度向量的限制。解码器在每一步都可以直接访问到源序列的全部信息，并根据需要动态聚焦。这使得模型在处理尤其是长序列时，性能得到了巨大的提升。

**2. 提供可解释性，实现“词对齐”**
注意力机制的另一个巨大价值在于它提供了很好的**可解释性**。注意力权重矩阵 $\alpha$ 本身就蕴含了丰富的信息。可以将这个矩阵可视化，来观察当模型生成某个输出词时，它的“注意力”主要集中在输入的哪些词上。

## 四、查询-键-值 (QKV) 范式

为了将刚刚描述的注意力计算过程抽象出来，形成一个更通用的思想，可以引入一个概念框架，即 **查询-键-值 (Query-Key-Value, QKV)** 。这个范式将注意力的计算过程类比为一次信息检索：

- **查询 (Query)**：代表了当前的需求或意图。在 Seq2Seq 中，这就是解码器在生成下一个词元前的状态 $h^{\prime}_{t-1}$，可以理解为它在“查询”：“根据我现在的情况，我最需要输入序列的哪部分信息？”
- **键 (Key)**：可以看作是输入序列中各个信息片段的“标签”或“索引”，用于和查询进行匹配。在 Seq2Seq 中，输入序列的每个词元的隐藏状态 $h_j$ 都对应一个“键”。
- **值 (Value)**：是与“键”对应的实际信息内容。在基础的注意力机制中，“键”和“值”通常是相同的，都来自于编码器的隐藏状态 $h_j$。

具体计算过程，可以用一个凝练的数学公式来统一表达，这就是 **缩放点积注意力 (Scaled Dot-Product Attention)** ，它也是 Transformer 模型的核心组件之一：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

这个公式准确概括了注意力的计算步骤：
1.  **$QK^T$**：计算查询矩阵 $Q$ 和键矩阵 $K$ 的转置的点积，得到原始的**注意力分数**。
2.  **$/\sqrt{d_k}$**：对分数进行**缩放**，以维持训练稳定性，其中 $d_k$ 是键向量的维度。
3.  **softmax(...)**：通过 Softmax 函数将分数归一化，得到**注意力权重**。
4.  **...V**：将得到的权重矩阵与值矩阵 $V$ 相乘，进行**加权求和**，得到最终的输出。

这个通用的范式很实用，是理解后续 Transformer 等更先进模型的基础。在不同的任务中，只需要思考如何定义场景中的 Q、K、V 即可应用注意力机制。
- 在刚刚讨论的 Seq2Seq 中：Q 是解码器状态，K 和 V 都是编码器状态序列。
- 在后续的自注意力 (Self-Attention) 机制中，Q, K, V 将全部来源于同一个序列自身。

此外，为了增加模型的表达能力，还可以在计算注意力之前，对原始的 Q, K, V 向量各自通过一个独立的全连接层进行**线性变换 (Linear Transformation)**，得到新的 Q', K', V'，再用它们进行注意力的计算。这种做法可以让模型学习到在不同的“子空间”中进行信息匹配和聚合。

## 五、PyTorch 实现与代码解析

> [本节完整代码](https://github.com/FutureUnreal/base-nlp/blob/main/code/C4/02_attention.py)

### 5.1 整体思路

要在 PyTorch 中实现一个带注意力的 Seq2Seq 模型，需要对原有的代码进行一些关键的调整：

1.  **编码器 `Encoder`**：
    -   `forward` 函数的返回值需要改变。除了最后一个时间步的隐藏状态 `(hidden, cell)`，还需要返回**所有时间步的输出** `outputs`，这正是注意力机制计算所需要的 Key 和 Value。
    -   如果编码器是**双向 (Bidirectional)** 的，其输出维度会是 `hidden_size * 2`。那么就需要增加一个线性层对其进行降维，或将其状态进行合并，以便与单向的解码器状态维度相匹配。

2.  **新增 `Attention` 模块**：
    -   创建一个独立的 `nn.Module` 类来实现注意力的计算逻辑。其 `forward` 方法接收解码器状态 (Query) 和编码器所有输出 (Keys/Values)，返回计算得到的上下文向量。

3.  **解码器 `Decoder`**：
    -   解码器的结构变化是最大的。它需要实例化一个 `Attention` 模块。
    -   其 `forward` 函数必须**以循环的方式**，逐个时间步进行解码。这是因为在第 $t$ 步计算注意力时，需要依赖第 $t-1$ 步的解码器状态。这种时序依赖性决定了它无法像基础版 Seq2Seq 那样进行并行计算。在循环的每一步，它都会调用 `Attention` 模块计算上下文向量，并将其与当前词元的词嵌入融合后，再送入 RNN 单元。

### 5.2 编码器 (Encoder)

为支持 Attention，编码器通常使用双向 RNN 以捕获更丰富的上下文，并需要返回所有时间步的输出序列作为注意力计算的 Key 和 Value。

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # 使用双向LSTM
        )
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.rnn(embedded)
        
        # 将双向RNN的输出通过线性层降维，使其与解码器维度匹配
        outputs = torch.tanh(self.fc(outputs))

        return outputs, hidden, cell
```
- **`bidirectional=True`**：启用双向 LSTM，使 `outputs` 维度变为 `(batch, src_len, hidden_size * 2)`。
- **`self.fc`**：定义一个线性层，将拼接后的双向输出映射回 `hidden_size` 维度，方便后续计算。
- **`return outputs, ...`**：返回降维后的所有时间步输出 `outputs` (作为后续的 K 和 V)，以及原始的最终状态 `hidden` 和 `cell`。

### 5.3 注意力模块 (Attention) 的两种实现

这是模型的核心，我们实现两个版本来体现其演进思路。

#### 无参数的注意力 (AttentionSimple)

这个版本直接使用缩放点积来计算注意力，不引入额外的可学习参数，对应了注意力机制最基础的数学思想。

```python
class AttentionSimple(nn.Module):
    """1: 无参数的注意力模块"""
    def __init__(self, hidden_size):
        super(AttentionSimple, self).__init__()
        # 确保缩放因子是一个 non-learnable buffer
        self.register_buffer("scale_factor", torch.sqrt(torch.FloatTensor([hidden_size])))

    def forward(self, hidden, encoder_outputs):
        # hidden shape: (num_layers, batch_size, hidden_size)
        # encoder_outputs shape: (batch_size, src_len, hidden_size)
        
        # Q: 解码器最后一层的隐藏状态
        query = hidden[-1].unsqueeze(1)  # -> (batch, 1, hidden)
        # K/V: 编码器的所有输出
        keys = encoder_outputs  # -> (batch, src_len, hidden)

        # energy shape: (batch, 1, src_len)
        energy = torch.bmm(query, keys.transpose(1, 2)) / self.scale_factor
        
        # attention_weights shape: (batch, src_len)
        return torch.softmax(energy, dim=2).squeeze(1)
```
- **`forward`**:
    - `query = hidden[-1].unsqueeze(1)`: 取解码器 LSTM 最后一层的状态作为 Query。
    - `energy = torch.bmm(...)`: 通过矩阵乘法 `torch.bmm` 计算 Query 和 Keys 的点积，并除以缩放因子。
    - `return torch.softmax(...)`: 对分数进行 softmax，得到归一化的注意力权重。

#### 带参数的注意力 (AttentionParams)

这个版本引入了可学习的参数（一个线性层和一个向量 `v`），让模型可以自主学习如何更好地对齐 Query 和 Keys，即 Bahdanau 风格的注意力。

```python
class AttentionParams(nn.Module):
    """2: 带参数的注意力模块"""
    def __init__(self, hidden_size):
        super(AttentionParams, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden_last_layer = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat((hidden_last_layer, encoder_outputs), dim=2)))
        attention = torch.sum(self.v * energy, dim=2)
        
        return torch.softmax(attention, dim=1)
```
- **`__init__`**: 定义了一个线性层 `self.attn` 和一个参数向量 `self.v` 作为可学习参数。
- **`forward`**:
    - `hidden_last_layer = ...`: 将解码器状态（Query）复制扩展，使其能与 `encoder_outputs` 的每个时间步对齐。
    - `energy = torch.tanh(...)`: 将扩展后的 Query 和 `encoder_outputs` 拼接起来，送入线性层计算一个“能量”或“对齐分数”。
    - `attention = torch.sum(...)`: 将能量值与参数向量 `v` 相乘并求和，得到最终的注意力分数。

### 5.4 通用解码器 (DecoderWithAttention)

为了能同时适配上述两种 Attention 模块，我们设计一个通用的解码器。其核心改动是在每个时间步都**调用 Attention 模块**，并将计算出的上下文向量**融入到当前步的输入**中。

```python
class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, attention_module):
        super(DecoderWithAttention, self).__init__()
        self.attention = attention_module
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        self.rnn = nn.LSTM(
            input_size=hidden_size * 2,  # 输入维度是 词嵌入(hidden_size) + 上下文向量(hidden_size)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        embedded = self.embedding(x.unsqueeze(1))

        # 1. 计算注意力权重
        # a shape: [batch, src_len]
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        
        # 2. 计算上下文向量
        context = torch.bmm(a, encoder_outputs)

        # 3. 将上下文向量与当前输入拼接
        rnn_input = torch.cat((embedded, context), dim=2)

        # 4. 传入RNN解码
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        # 5. 预测输出
        predictions = self.fc(outputs.squeeze(1))
        
        return predictions, hidden, cell
```
- **`__init__`**:
    - `self.attention`: 持有传入的 Attention 实例（可以是 `AttentionSimple` 或 `AttentionParams`）。
    - `self.rnn`: `input_size` 变为 `hidden_size * 2`，因为它接收的是**词嵌入向量**和**上下文向量**拼接后的结果。
- **`forward`**: 完整地演示了注意力的应用流程。
    - `a = self.attention(...)`: 调用 attention 模块计算权重 `a`。
    - `context = torch.bmm(a, encoder_outputs)`: 对应注意力计算的**第三步**。使用矩阵乘法，通过权重 `a` 对 `encoder_outputs` (Values) 进行加权求和，得到上下文向量 `context`。
    - `rnn_input = torch.cat(...)`: 将动态生成的上下文向量和当前词的嵌入向量拼接起来，形成一个信息更丰富的输入，再送入 RNN 进行解码。

### 5.5 Seq2Seq 包装模块

这个模块负责将上述组件串联起来，并处理好双向编码器与单向解码器之间的状态传递问题。

```python
class Seq2Seq(nn.Module):
    """带注意力的Seq2Seq"""
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        # 适配Encoder(双向)和Decoder(单向)的状态维度
        hidden = hidden.view(self.encoder.rnn.num_layers, 2, batch_size, -1).sum(axis=1)
        cell = cell.view(self.encoder.rnn.num_layers, 2, batch_size, -1).sum(axis=1)

        input = trg[:, 0]
        for t in range(1, trg_len):
            # 在循环的每一步，都将 encoder_outputs 传递给解码器
            # 这是 Attention 机制能够"回顾"整个输入序列的关键
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
            
        return outputs
```
- **状态适配**: 编码器是双向的，其 `hidden` 状态形状为 `(num_layers * 2, ...)`。解码器是单向的，需要 `(num_layers, ...)` 的初始状态。这里的 `hidden.view(...).sum(axis=1)` 通过 `view` 操作将状态拆分为 `(层数, 方向, ...)`，然后在方向维度上求和，巧妙地将双向状态合并为单向状态。
- **循环解码**: 正如之前所强调的，Attention 机制下的解码必须是串行的。在 `for` 循环的每一步，都将 `encoder_outputs` 完整地传递给解码器，确保解码器在每个时间步都能基于上一时刻的状态，动态计算出当前最需要的上下文信息。

这个实现完整地展示了 Attention 机制如何解决信息瓶颈问题：解码器不再只依赖于一个固定的上下文向量，而是在生成的每一步，都通过 Attention 模块动态地计算出一个与当前解码状态最相关的上下文向量，极大地提升了模型性能。

注意力机制的出现，是深度学习在自然语言处理领域的一个里程碑。它不仅解决了 RNN 时代的一个核心难题，其“动态加权”的核心思想更是直接启发了后来完全抛弃 RNN、完全基于注意力机制构建的 **Transformer** 模型，开启了 NLP 的新纪元。

## 参考文献
[^1]: [Bahdanau, D., Cho, K., & Bengio, Y. (2014). *Neural machine translation by jointly learning to align and translate*. arXiv preprint arXiv:1490.0473.](https://arxiv.org/abs/1490.0473)
