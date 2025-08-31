# 第一节 Seq2Seq 架构

在前面的章节中，已经学习了如何使用 RNN 和 LSTM 处理序列数据。这些模型在三类任务中表现出色：

1.  **多对一 (Many-to-One)** ：将整个序列信息压缩成一个特征向量，用于文本分类、情感分析等任务。

2.  **多对多 (Many-to-Many, Aligned)** ：为输入序列的每一个词元（Token）都生成一个对应的输出，如词性标注、命名实体识别等。

3.  **一对多 (One-to-Many)** ：从一个固定的输入（如一张图片、一个类别标签）生成一个可变长度的序列，例如图像描述生成、音乐生成等。

然而，在自然语言处理中，还存在一类更复杂的、被称为 **多对多 (Many-to-Many, Unaligned)** 的任务，它们的 **输入序列和输出序列的长度可能不相等** ，且元素之间没有严格的对齐关系。最典型的例子就是 **机器翻译**：“我是中国人” (3个词) 翻译成 "I am Chinese" (3个词)，但 "我爱人工智能" (3个词) 翻译成 "I love artificial intelligence" (4个词)。

> 注：此处将“人工智能”视为单个单元仅为方便举例，旨在说明输入与输出序列长度可能不等的概念，不代表严格的分词标准。

对于这类问题，简单的 RNN 或 LSTM 架构难以胜任。为了解决这一挑战，2014年，研究者们提出了 **序列到序列（Sequence-to-Sequence, Seq2Seq）** 架构，它成功地将一种通用的 **编码器-解码器（Encoder-Decoder）** 架构应用于序列转换任务。[^1][^2] 该架构一经提出，便在机器翻译、文本摘要、对话系统等领域取得了巨大成功。

## 一、从自编码器到 Seq2Seq

要理解 Seq2Seq，可以先从一种更基础的、同样使用编码器-解码器思想的无监督神经网络—— **自编码器 (Autoencoder)** ——说起。

自编码器由两个部分组成：

1.  **编码器 (Encoder)** : 读取输入数据（如一张图片、一个向量），并将其压缩成一个低维度的、紧凑的 **潜在表示 (Latent Representation)** 。这个过程可以看作是特征提取或数据压缩。

2.  **解码器 (Decoder)** : 接收这个潜在表示，并尝试将其**重构**回原始的输入数据。

自编码器的训练目标是让 **输出与输入尽可能地相同**。通过这个过程，模型被迫学习到数据中最具代表性的核心特征，并将其编码在潜在表示中。它的目标是 **数据重构**，常被用于降维、特征学习或数据去噪等任务。

## 二、Seq2Seq 架构详解

Seq2Seq 架构借鉴了自编码器的结构，但对其核心目标进行了关键的 **泛化**：它不再要求解码器的输出与编码器的输入相同，而是要生成一个全新的、与输入语义相关的 **目标序列** 。

### 2.1 整体架构

- **核心思想** ：借鉴人类进行翻译的过程——先完整地阅读并理解源语言的整个句子，形成一个综合的 **语义表示**；然后，基于这个语义表示，开始用目标语言逐词生成译文。

- **目标** ：从 `Input` 到 `Output` 的 **转换**，而非重构。

模型同样被拆分为两个组件：

1.  **编码器 (Encoder)** ：扮演“阅读和理解”的角色。它负责接收整个输入序列，并将其信息压缩成一个固定长度的 **上下文向量 (Context Vector)** ，通常记为 $C$。这个向量就是输入序列的“语义概要”。

2.  **解码器 (Decoder)** ：扮演“组织语言并生成”的角色。它接收上下文向量 $C$ 作为初始信息，然后逐个生成输出序列中的词元。

在最初基于 Seq2Seq 架构的模型中，编码器和解码器通常都由 RNN 或其变体（如 LSTM、GRU）构成。

![Seq2Seq详细工作流程](./images/10_2_1.svg)

上图以“I love you” -> “我爱你”的翻译任务为例，展示了一个基于 LSTM 的 Seq2Seq 架构如何将编码、解码与自回归机制结合在一起的完整工作流程。

### 2.2 编码器 (Encoder)

编码器的任务是生成上下文向量 $C$。
- 它可以是一个标准的 RNN（或 LSTM），逐个读取输入序列的词元 $x_1, x_2, \dots, x_T$。

- 在每个时间步，它都会根据前一时刻的状态和当前输入来更新自身状态。对于标准 RNN，这个过程可以简化为 $h_t = f(h_{t-1}, x_t)$；而对于 LSTM，则同时更新隐藏状态和细胞状态： $(h_t, c_t) = \text{LSTM}((h_{t-1}, c_{t-1}), x_t)$。

- 当处理完最后一个输入词元 $x_T$后，编码器最终的状态就被用作整个输入序列的上下文向量 $C$。对于 LSTM，上下文向量 $C$ 通常就是最后一个时间步的隐藏状态和细胞状态的元组，即 $C = (h_T, c_T)$。虽然这是最常见的做法，但上下文向量 $C$ 也可以由所有时间步的隐藏状态 $\{h_1, h_2, \dots, h_T\}$ 经过某种变换（如拼接后通过一个线性层、或取平均池化）得到，以期保留更全面的序列信息。在图中，编码器依次处理英文单词 “I”、“love”、“you” 的词嵌入向量，并将最终的状态打包成上下文向量（Context Vector）传递给解码器。

> 由于编码器在处理时可以访问整个输入序列，因此它可以使用**双向 RNN**。通过同时从正向和反向两个方向读取序列，编码器可以为每个词元生成更全面的上下文表示，从而得到一个信息更丰富的上下文向量 $C$。

### 2.3 解码器 (Decoder)

解码器的任务是根据上下文向量 $C$ 生成输出序列 $y_1, y_2, \dots, y_{T'}$。
- 解码器同样可以使用一个标准的 RNN（或 LSTM）作为核心，但它扮演的角色是**生成器**而非信息压缩器，因此其工作流程与编码器有显著差异。

- **初始化** ：解码器的初始状态直接由编码器生成的上下文向量 $C$ 初始化。对于 LSTM，这意味着初始的隐藏状态和细胞状态 $(h^{\prime}_0, c^{\prime}_0)$ 都被设置为编码器的最终状态 $C=(h_T, c_T)$。这相当于将整个输入序列的“语义概要”交给了解码器。

- **自回归生成 (Auto-regressive Generation)** ：解码器逐个生成词元。
    
    1.  在第一个时间步，它以初始状态（对 LSTM 而言是 $(h^{\prime}_0, c^{\prime}_0)$）和一个特殊的起始符 `<SOS>` (Start of Sentence) 作为输入，生成第一个目标词元 $y_1$。

    2.  在第二个时间步，它将上一步的状态（$(h^{\prime}_1, c^{\prime}_1)$）和 **上一步生成的词元 $y_1$** 作为输入，生成第二个目标词元 $y_2$。

    3.  这个过程不断重复，状态也随之更新。对于 LSTM，这个更新过程可以表示为 $(h^{\prime}_t, c^{\prime}_t) = \text{LSTM}((h^{\prime}_{t-1}, c^{\prime}_{t-1}), y_{t-1})$。这个过程将持续进行，直到生成一个特殊的终止符 `<EOS>` (End of Sentence) 或达到预设的最大长度。图中展示的正是这个过程：解码器首先接收 `<SOS>` 符和上下文向量，生成第一个汉字“我”；接着，它将“我”作为下一步的输入，生成“爱你”；这个过程将持续进行，直到生成句子结束符 `<EOS>` 为止。

> **注意**：解码器在生成序列时，是按照从左到右的顺序逐词生成的，它在预测当前词元时不能“看到”未来的词元。为满足因果性约束，解码器通常使用单向 RNN（或采用因果掩码的解码结构）。

在每个生成步骤中，解码器的隐藏状态 $h^{\prime}_t$ 会经过一个额外的全连接层（通常带有 Softmax 激活函数），以计算出词汇表中每个单词的概率分布。然后，模型会选择概率最高的单词作为当前时间步的输出。

#### 2.3.1 解码器：一个条件语言模型
从更深层次看，解码器本身就是一个强大的 **条件语言模型 (Conditional Language Model)** 。

- **语言模型 (Language Model)** ：一个核心任务是 **预测下一个词元** 的模型。就像日常使用的手机输入法，输入“今天天气”后，它会预测出“真好”、“不错”等可能的后续词语。一个不带任何条件的标准语言模型，可以基于上文持续生成文本。

- **Seq2Seq 的解码器** ：它执行的也是“预测下一个词元”的任务，但它不是凭空预测，而是 **以编码器生成的上下文向量 $C$ 为初始条件**。一旦接收了 $C$，解码器就开启了它的生成过程，将这个语义概要“翻译”成目标序列。

因此，可以认为 **Seq2Seq 的本质 = 编码器（用于理解和压缩信息）+ 条件语言模型（用于在特定条件下生成信息）**。这个“编码-生成”的范式是现代生成模型的基石。

#### 2.3.2 解码器与大语言模型的关系

如果去掉编码器，只保留解码器部分，会发生什么？

此时，模型不再接收外部的上下文向量 $C$ 作为条件。它可视为仅基于自身前缀（提示词）的 **decoder-only 语言模型** ：根据已有前缀预测下一个词元。例如，给定一个起始词元或提示前缀，它可以自回归地生成后续词元，写出完整的句子或段落。

这与 **GPT (Generative Pre-trained Transformer)** 等现代大语言模型的训练范式一致：它们采用解码器（“Decoder-only”）架构，以“预测下一个词”为目标在海量文本上预训练；使用时通过提示词进行条件化生成，学习并利用语言规律、事实知识与一定的推理能力。

### 2.4 实现细节与考量

#### 2.4.1 词嵌入层共享

编码器和解码器都需要将输入的词元ID转换为向量，这通常由一个 `Embedding` 层完成。这里存在一个设计选择：

- **不共享**：编码器和解码器各自拥有独立的 `Embedding` 层。若源语言和目标语言的词汇表彼此独立（如未采用联合子词/合并词表的英译中），通常选择不共享。

- **共享**：编码器和解码器使用同一个 `Embedding` 层。如果源语言和目标语言词汇表有大量重叠（如文本摘要任务，输入和输出都是中文），或者干脆将两种语言的词汇合并成一个大词汇表，那么共享 `Embedding` 层是可行的。这样做可以减少模型参数，并可能让模型学到两种语言之间词元的潜在联系。

#### 2.4.2 上下文向量的传递与使用

理论上，编码器的最终状态 $C$ 会被传递给解码器。但在实践中，如何使用这个上下文向量 $C$ 有两种主流方式，这体现了架构设计的灵活性。

**方式一：作为解码器的初始状态 (Initial State)**

这是最经典的做法。将编码器输出的上下文向量 $C$ 经过适配层（如全连接层、`reshape`、`permute`等操作）变换后，作为解码器RNN的初始隐藏状态 $h^{\prime}_0$（以及 $c^{\prime}_0$ for LSTM）。

-   **优点**：概念直观，符合“先理解全文（生成$C$），再开始生成（初始化解码器）”的逻辑。

-   **缺点**：解码器的所有生成步骤都源于这“唯一一次”的初始信息输入。对于长序列，RNN自身的长距离依赖问题可能会导致初始状态的信息在多步传递后逐渐“稀释”或“遗忘”。

**方式二：作为解码器每个时间步的输入 (Input at Every Step)**

另一种方式是，不改变解码器默认的零向量初始状态，而是将上下文向量 $C$ 作为解码器**每一个时间步**的额外输入。

具体实现上，在第 $t$ 个时间步，将常规的词元输入 $y_{t-1}$ 经过`Embedding`层后得到的向量，与上下文向量 $C$ 进行合并。合并的方式可以是：

1.  **拼接 (Concatenate)**：将两个向量拼接在一起。这会改变输入特征的维度，需要相应地调整后续RNN层的输入大小。

2.  **相加 (Add/Sum)**：将两个向量逐元素相加。这要求词嵌入向量的维度与上下文向量 $C$ 的维度相同。同时，为了进行广播加法，需要先将二维的 $C$ (`Batch Size, Hidden Size`) 通过 `unsqueeze` 操作扩展成与输入词嵌入序列匹配的三维形状 (`Batch Size, 1, Hidden Size`)。

-   **优点**：在每个生成步骤都直接“提醒”解码器全局上下文信息是什么，理论上可以更好地对抗信息遗忘。

-   **缺点**：虽然信息在每个时间步都存在，但输入的全局信息始终是**同一个**静态向量 $C$，它仍然无法解决更深层次的“对齐”问题（即，在生成某个特定词时，应该重点关注输入的哪个部分）。

总的来说，这两种方式都无法从根本上解决信息瓶颈问题，但这也正是它们激发后续注意力机制（Attention Mechanism）诞生的重要原因。

#### 2.4.3 损失函数计算

在训练过程中，解码器的目标是让其在每个时间步 $t$ 的输出概率分布，尽可能地接近真实的目标词元 $y_t$。具体来说：

1.  解码器在时间步 $t$ 的隐藏状态 $h^{\prime}_t$ 会经过一个全连接层（分类器），并使用 Softmax 函数计算出词汇表中每个词的概率，得到一个概率分布向量 $p_t$。模型的原始输出形状通常是 `(Batch Size, Sequence Length, Vocab Size)`。

2.  损失函数（通常是 **交叉熵损失 Cross-Entropy Loss** ）会计算这个预测概率分布 $p_t$ 与真实目标 $y_t$ 之间的差异。以 PyTorch 为例，`CrossEntropyLoss` 接受形状为 `(N, C, ...)` 的输入，也可将 `(N, L, C)` 展平为 `(N·L, C)` 与目标 `(N·L)` 计算；若使用 `(N, C, L)` 形式，可通过 `permute` 将 `(N, L, C)` 交换至 `(N, C, L)`。[^3]

3.  训练时通常配合 `ignore_index` 忽略 `<PAD>` 位置的损失，从而避免填充对梯度的干扰。

4.  损失函数的计算本质上是取出 $p_t$ 中对应真实词元 $y_t$ 的那个概率值，取其负对数：$Loss_t = -\log p_t(y_t)$。

5.  整个序列的总损失是所有时间步损失的累加或平均：$Loss_{total} = \sum_{t=1}^{T'} Loss_t$。

6.  最后，通过反向传播算法，根据这个总损失来更新模型的所有参数（包括编码器和解码器）。

#### 2.4.4 数据填充与特殊词元

在实际处理中，一个批次（Batch）中的序列长度往往不同。为了能够进行高效的矩阵运算，需要将它们填充（Pad）到相同的长度。此外，还需要引入一些特殊的词元（Token）来辅助模型处理。

- **特殊词元**：
    -   `<PAD>`：填充符，用于对齐长度，在计算损失时会被忽略。
    -   `<SOS>` 或 `<GO>`：句子起始符，作为解码器第一个时间步的输入，启动生成过程。
    -   `<EOS>`：句子终止符，是解码器生成的目标之一。当模型生成它时，表示句子已完整，可以停止生成。
    -   `<UNK>`：未知词元。用于替换在训练词汇表中未出现过的词，增强模型的鲁棒性。

- **编码器输入**：对源语言序列进行填充，通常在末尾添加 `<PAD>`。
- **解码器输入与目标**：解码器的输入和目标序列需要精心构造，以实现“错位”训练，即用上一个真实词元预测下一个词元。
  - **原始目标序列**: `W, X, Y, Z`
  - **解码器输入**: 在序列开头添加起始符 `<SOS>`，并移除最后一个词元。如果需要填充，则在末尾添加 `<PAD>`。
    `-> <SOS>, W, X, Y`
  - **解码器目标**: 在序列末尾添加终止符 `<EOS>`。如果需要填充，则在末尾添加 `<PAD>`，且在计算损失时**忽略**填充位置的损失。
    `-> W, X, Y, Z, <EOS>`

通过这种方式，模型在每个时间步都能学到从正确的历史信息到下一个正确词元的映射关系。

## 三、训练与推理：两种不同的模式

基于Seq2Seq架构的模型在训练和推理（即实际生成）时，解码器的工作模式有很大差异。

### 3.1 训练模式：教师强制 (Teacher Forcing)

如果在训练时也采用推理时的自回归模式（将上一时刻的预测作为下一时刻的输入），会存在两个问题：

1.  **收敛缓慢**：模型在训练初期预测不准，错误的预测会不断被喂给后续的步骤，导致误差累积，模型很难收敛。

2.  **难以并行**：每个时间步的计算都依赖于上一步的结果，使得训练过程无法并行化，效率低下。

为了解决这个问题，Seq2Seq 采用了一种名为 **教师强制 (Teacher Forcing)** [^4] 的高效训练策略。

在教师强制模式下，解码器在计算第 $t$ 步的输出时，其输入 **不再是自己上一时刻的预测值 $y^{\prime}_{t-1}$** ，而是直接使用 **数据集中真实的标签值 $y_{t-1}$** ，具体构造方式如 `2.4.4` 节所述。

通过这种方式，解码器的每个时间步都可以接收到正确的历史信息，避免了误差累积，显著提升了收敛稳定性与速度。需要注意：对于基于 RNN 的解码器，时间维的计算仍是串行依赖的；在 Transformer 等非递归结构中，训练时可在时间步上并行（配合适当掩码）。

### 3.2 推理模式：自回归 (Auto-regressive)

在模型训练完毕，进行实际的翻译或生成任务时，我们并没有“正确答案”可以喂给解码器。此时，模型必须工作在**自回归模式**下，即“自己教自己”：

1.  编码器处理输入序列，生成上下文向量 $C$。

2.  解码器以 $C$ 和 `<SOS>` 为初始输入，生成第一个词元 $y_1$。

3.  将生成的 $y_1$ 作为解码器下一时间步的输入，生成 $y_2$。

4.  不断将上一时刻的输出作为下一时刻的输入，循环此过程。

5.  当解码器生成 `<EOS>` 标志，或达到预设的最大输出长度时，生成过程停止。

> **推理效率的优化**
>
> 在朴素的自回归实现中，存在大量的重复计算。例如：
> - **第1步**：输入 `<SOS>`，RNN 内部计算 $h^{\prime}_1 = f(h^{\prime}_0, y_0)$。
> - **第2步**：输入 `<SOS>, $y_1$`，RNN 会**重新**计算 $h^{\prime}_1 = f(h^{\prime}_0, y_0)$，然后再计算 $h^{\prime}_2 = f(h^{\prime}_1, y_1)$。
> - **第3步**：输入 `<SOS>, $y_1$, $y_2$`，RNN 会**再次重新**计算 $h^{\prime}_1$ 和 $h^{\prime}_2$，然后再计算 $h^{\prime}_3$。
>
> 这种“从头算起”的方式效率极低。高效的实现方式是**缓存并利用上一个时间步的输出状态**：
> 1.  在生成第 $t$ 个词元时，只将**第 $t-1$ 个词元** $y_{t-1}$ 和**上一步的隐藏状态** $h^{\prime}_{t-1}$ 作为RNN的输入。
> 2.  RNN 仅执行一步计算，得到新的隐藏状态 $h^{\prime}_{t}$ 和当前词元的预测 logits。
> 3.  这个新的状态 $h^{\prime}_{t}$ 会被缓存，用于下一步的计算。
>
> 通过这种方式，每个时间步都只进行一次RNN单元的计算，极大地提升了推理速度。

这种在每一步都选择当前概率最高的词元作为输出的策略，被称为 **贪心搜索 (Greedy Search)** 。它简单高效，但在某些情况下可能会导致次优解。例如，如果在某一步选择了一个局部最优但在全局看来是错误的词，这个错误可能会影响后续所有词元的生成，导致整个输出序列的质量下降。这就好比下棋时只看眼前一步的最好走法，却可能导致满盘皆输。

要缓解这个问题，通常有两种思路：
1.  **提升模型能力**：通过使用更深、更复杂的模型架构（例如，从3层网络变成30层网络）和更大规模的训练数据，让模型本身在每一步做出正确预测的概率大大提高。
2.  **改进解码策略**：使用更复杂的解码算法，如 **束搜索 (Beam Search)** 。它在每一步都会保留多个（而不是一个）最可能的候选序列，并在最后选择整体概率最高的序列作为最终输出，从而在全局上找到更优的解，避免“一步错，步步错”的陷阱。

---

## 四、PyTorch 代码实现与分析

理论结合实践有助于更深刻地理解相关概念。接下来，将基于 PyTorch 来实现一个遵循 Seq2Seq 架构的具体模型。

> [本节完整代码](https://github.com/FutureUnreal/base-nlp/blob/main/code/C4/01_Seq2Seq.py)

### 4.1 核心组件：标准的 Encoder-Decoder

首先，来构建模型的基础骨架：编码器、解码器以及将它们组合在一起的 Seq2Seq 包装器。

#### 4.1.1 编码器 (Encoder)

编码器的职责是读取输入序列并生成上下文向量。在这个示例实现中，将单向 LSTM 的最终隐藏状态 `hidden` 和细胞状态 `cell` 直接作为上下文，传递给解码器。

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
            bidirectional=False
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)
        # 返回最终的隐藏状态和细胞状态作为上下文
        _, (hidden, cell) = self.rnn(embedded)
        return hidden, cell
```

**代码解析:**

1.  **`__init__`**:
    -   `self.embedding`: 定义词嵌入层，将输入的词元ID（整数）映射为稠密的`hidden_size`维度向量。
    -   `self.rnn`: 定义核心的 LSTM 层。`input_size` 和 `hidden_size` 均为 `hidden_size`，因为词嵌入向量的维度与 LSTM 隐藏状态的维度在此设计中保持一致。此处为简化演示选择单向（`bidirectional=False`）；实际工程中编码器常使用双向 RNN 以获取更充分的上下文，需要将双向状态（如拼接/线性映射）转换为解码器的初始状态。

2.  **`forward(self, x)`**:
    -   输入 `x` 是一个形状为 `(batch_size, seq_length)` 的张量，代表了一批句子的词元ID序列。
    -   `embedded = self.embedding(x)`: 输入经过词嵌入层，形状变为 `(batch_size, seq_length, hidden_size)`。
    -   `_, (hidden, cell) = self.rnn(embedded)`: 这是编码过程的核心。`self.rnn` 处理整个嵌入序列后，会返回两个内容：
        -   `outputs`: 包含了序列中**每一个时间步**的隐藏状态。对于编码器而言，中间步骤的输出通常不被使用，因此用 `_` 接收。
        -   `(hidden, cell)`: 一个元组，包含了整个序列**最后一个时间步**的隐藏状态和细胞状态。这正是我们需要的、概括了整个输入序列信息的**上下文向量**。
    -   `return hidden, cell`: 函数最终返回这两个状态，作为上下文传递给解码器。

#### 4.1.2 解码器 (Decoder)

解码器在每一步接收一个词元和前一步的状态，然后输出预测和新的状态。这个实现体现了为**高效推理**而设计的**单步前向传播**逻辑，即 `forward` 函数一次只处理一个时间步。

```python
class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, hidden, cell):
        # x shape: (batch_size)，只包含当前时间步的token
        x = x.unsqueeze(1) # -> (batch_size, 1)

        embedded = self.embedding(x)
        # 接收上一步的状态 (hidden, cell)，计算当前步
        outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        predictions = self.fc(outputs.squeeze(1)) # -> (batch_size, vocab_size)
        return predictions, hidden, cell
```

**代码解析:**

1.  **`__init__`**:
    -   `self.embedding` 和 `self.rnn`: 与编码器中的定义类似。
    -   `self.fc`: 增加了一个全连接层（`Linear`），它的作用是将 LSTM 输出的 `hidden_size` 维度的隐藏状态，映射到 `vocab_size` 维度的向量上。这个向量的每一个元素对应词汇表中一个词的得分（logit），后续可以通过 Softmax 函数转换为概率。

2.  **`forward(self, x, hidden, cell)`**:
    -   这是一个 **单步 (step-by-step)** 的前向传播函数，其输入 `x` 是一个形状为 `(batch_size,)` 的张量，仅包含当前时间步的词元ID。
    -   `x = x.unsqueeze(1)`: 为了适应 `nn.Embedding` 和 `nn.LSTM` 对输入形状（需要有序列长度维度）的要求，需要给 `x` 增加一个长度为1的“伪序列”维度，使其形状变为 `(batch_size, 1)`。
    -   `embedded = self.embedding(x)`: 词元经过嵌入，形状变为 `(batch_size, 1, hidden_size)`。
    -   `outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))`: 解码器的 RNN 接收两个输入：当前步的嵌入向量 `embedded`，以及**上一步**传递过来的隐藏状态 `(hidden, cell)`。它只进行一步计算，然后返回当前步的输出 `outputs` 和更新后的状态 `(hidden, cell)`。
    -   `predictions = self.fc(outputs.squeeze(1))`: RNN 的输出 `outputs` 形状是 `(batch_size, 1, hidden_size)`，需要用 `squeeze(1)` 移除长度为1的序列维度，再送入全连接层，得到形状为 `(batch_size, vocab_size)` 的最终预测。
    -   `return predictions, hidden, cell`: 返回当前步的预测，以及更新后的状态，用于下一步的计算。

#### 4.1.3 Seq2Seq 包装模块

这个包装模块将编码器和解码器连接起来，并负责实现**训练**时的逻辑，特别是**教师强制**。

```python
class Seq2Seq(nn.Module):
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
        hidden, cell = self.encoder(src)

        # 第一个输入是 <SOS>
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output

            # 决定是否使用 Teacher Forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            # 如果 teacher_force，下一个输入是真实值；否则是模型的预测值
            input = trg[:, t] if teacher_force else top1

        return outputs
```

**代码解析 (`forward`):**

`forward` 函数模拟了训练过程中的一个批次计算：

1.  **初始化**:
    -   `outputs = torch.zeros(...)`: 创建一个全零张量，用于存储解码器在每一个时间步的输出 logits。
    -   `hidden, cell = self.encoder(src)`: 调用编码器，处理源序列 `src`，得到初始的上下文向量。

2.  **启动解码**:
    -   `input = trg[:, 0]`: 取出目标序列 `trg` 的第一个词元，这通常是 `<SOS>` 标志，作为解码器循环的起始输入。

3.  **循环解码**:
    -   `for t in range(1, trg_len)`: 循环从第二个词元（索引为1）开始，直到目标序列结束。
    -   `output, hidden, cell = self.decoder(input, hidden, cell)`: 调用解码器执行单步计算，得到当前步的预测 `output` 和更新后的状态。
    -   `outputs[:, t, :] = output`: 将当前步的预测存入 `outputs` 张量中。

4.  **教师强制**:
    -   `teacher_force = random.random() < teacher_forcing_ratio`: 以一定的概率决定是否启用教师强制。
    -   `top1 = output.argmax(1)`: 找出当前步预测概率最高的词元，作为“学生”自己的答案。
    -   `input = trg[:, t] if teacher_force else top1`: 这是教师强制的核心。如果 `teacher_force` 为 `True`，则**无视模型的预测**，直接将**真实的下一个词元 `trg[:, t]`** 作为解码器下一步的输入（“老师”纠正）；否则，将模型自己的预测 `top1` 作为下一步的输入（“学生”自学）。

5.  **返回**: 最终返回包含了所有时间步预测的 `outputs` 张量，用于后续与真实标签计算损失。

### 4.2 模式详解：推理的高效与低效

在推理时，模型必须工作在自回归模式下。如何实现自回归，直接关系到模型的推理效率。

#### 4.2.1 低效实现：从头计算（教学对比用）

一个常见的、但效率极低的错误是，在生成每个新词元时，都将**已生成的完整序列**重新喂给解码器。这会导致大量的重复计算。为了演示这一点，这里特意构建了一个 `DecoderForBadInference`，它的 `forward` 函数接收的是一个序列而不是单个词元。

```python
# 这是一个专门为演示错误而设计的解码器
class DecoderForBadInference(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(DecoderForBadInference, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, hidden, cell):
        # x shape: (batch_size, current_seq_len)
        embedded = self.embedding(x)
        # 重新计算整个序列的RNN状态
        outputs, _ = self.rnn(embedded, (hidden, cell))
        # !! 关键缺陷：只取最后一个时间步的输出，前面的计算被浪费了
        last_output = outputs[:, -1, :]
        predictions = self.fc(last_output)
        return predictions

# 对应的低效推理循环
def inefficient_greedy_decode(encoder, decoder, src, max_len=12, sos_idx=1):
    with torch.no_grad():
        hidden, cell = encoder(src)
        trg_indexes = [sos_idx]
        for i in range(max_len):
            # !! 关键缺陷: 每次都把完整的 trg_indexes 作为输入
            trg_tensor = torch.LongTensor([trg_indexes]).to(device)
            # 打印输入形状，会看到它在不断变长: [1,1], [1,2], [1,3]...
            print(f"第 {i+1} 步，解码器输入 shape: {trg_tensor.shape}")
            output = decoder(trg_tensor, hidden, cell)
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)
            if pred_token == eos_idx:
                break
    return trg_indexes
```

**代码解析:**

这段代码的低效根源在于**重复计算**：

1.  **`DecoderForBadInference.forward`**:
    -   它的输入 `x` 是一个不断增长的序列，如 `[<SOS>]`, `[<SOS>, w1]`, `[<SOS>, w1, w2]`...
    -   `outputs, _ = self.rnn(embedded, (hidden, cell))`: 每次调用，RNN 都会**从头到尾**处理一遍当前输入的整个序列。例如，在计算第3个词时，它会重新计算第1、2个词对应的隐藏状态。
    -   `last_output = outputs[:, -1, :]`: 计算完整个序列后，却只取**最后一个时间步**的输出来做预测，前面所有的计算结果都被**浪费**了。

2.  **`inefficient_greedy_decode` 循环**:
    -   `trg_tensor = torch.LongTensor([trg_indexes]).to(device)`: 这一行是“罪魁祸首”。在每次循环中，它都将 `trg_indexes` 这个**完整的、不断变长的列表**转换成张量，作为解码器的输入，从而触发了上述的重复计算。

#### 4.2.2 高效实现：传递并更新状态（标准实践）

正确的做法是利用 RNN 的“记忆”能力。只需将**上一步的输出词元**和**上一步的隐藏状态**传入解码器，进行单步计算，然后用返回的新状态覆盖旧状态即可。这个逻辑被封装在 `Seq2Seq` 类的 `greedy_decode` 方法中。

```python
class Seq2Seq(nn.Module):
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
        hidden, cell = self.encoder(src)

        # 第一个输入是 <SOS>
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output

            # 决定是否使用 Teacher Forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            # 如果 teacher_force，下一个输入是真实值；否则是模型的预测值
            input = trg[:, t] if teacher_force else top1

        return outputs

    def greedy_decode(self, src, max_len=12, sos_idx=1, eos_idx=2):
        """推理模式下的高效贪心解码。"""
        self.eval()
        with torch.no_grad():
            hidden, cell = self.encoder(src)
            trg_indexes = [sos_idx]
            for _ in range(max_len):
                # 1. 输入只有上一个时刻的词元
                trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device)

                # 2. 解码一步，并传入上一步的状态
                output, hidden, cell = self.decoder(trg_tensor, hidden, cell)
                
                # 3. 获取当前步的预测，并更新状态用于下一步
                pred_token = output.argmax(1).item()
                trg_indexes.append(pred_token)
                if pred_token == eos_idx:
                    break
        return trg_indexes
```

**代码解析:**

这个高效的实现避免了任何重复计算，其核心在于**状态的传递与更新**：

1.  **`hidden, cell = self.encoder(src)`**: 在循环开始前，只调用一次编码器，获取初始上下文。
2.  **循环内部**:
    -   `trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device)`: 每次的输入**仅仅是上一步生成的最后一个词元** `trg_indexes[-1]`，而不是整个序列。这是一个形状为 `(1, 1)` 的张量。
    -   `output, hidden, cell = self.decoder(trg_tensor, hidden, cell)`: 将这个单词元输入和**上一步的 `hidden`, `cell` 状态**送入解码器。解码器只执行一步计算，并返回**新的 `hidden`, `cell` 状态**。
    -   这两个新状态会**覆盖**旧的状态变量，并在下一次循环中被用作输入。

通过这种方式，信息流和状态在时间步之间平稳地传递，每个时间步都只进行一次必要的计算，极大地提升了推理效率。

### 4.3 设计变体：上下文向量的另一种用法

除了将上下文向量用作解码器的初始状态外，还可以将其作为解码器**每个时间步的额外输入**。这种方式可以持续地为解码器提供全局信息。

下面是这种变体解码器的实现。注意 `rnn` 层的输入维度和 `forward` 函数中的**拼接**操作。

```python
class DecoderAlt(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(DecoderAlt, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        # !! 核心改动1: RNN的输入维度是 词嵌入+上下文向量
        self.rnn = nn.LSTM(
            input_size=hidden_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, hidden_ctx, hidden, cell):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)

        # !! 核心改动2: 将上下文向量与当前输入拼接
        # 这里简单地取编码器最后一层的 hidden state 作为上下文代表
        context = hidden_ctx[-1].unsqueeze(1).repeat(1, embedded.shape[1], 1)
        rnn_input = torch.cat((embedded, context), dim=2)

        # 解码器的初始状态 hidden, cell 在第一步可设为零；之后需传递并更新上一步状态
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        predictions = self.fc(outputs.squeeze(1))
        return predictions, hidden, cell
```

**代码解析:**

1.  **`__init__`**:
    -   `self.rnn = nn.LSTM(...)`: 这里的核心改动是 `input_size=hidden_size + hidden_size`。因为在每个时间步，输入给 LSTM 的不再仅仅是词嵌入向量（维度 `hidden_size`），而是**词嵌入向量**与**上下文向量**（维度也是 `hidden_size`）拼接后的新向量，因此输入维度加倍。

2.  **`forward(self, x, hidden_ctx, hidden, cell)`**:
    -   `context = hidden_ctx[-1].unsqueeze(1).repeat(1, embedded.shape[1], 1)`: 这一步是为了准备用于拼接的上下文向量。
    -   `rnn_input = torch.cat((embedded, context), dim=2)`: 核心操作，在最后一个维度（特征维度）上，将词嵌入向量和上下文向量拼接起来，形成 RNN 的最终输入。
    -   `outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))`: 将拼接后的向量送入 RNN。注意，这里传入的 `(hidden, cell)` 是解码器自身的上一步状态（初始是零向量），而不是编码器传来的上下文 `hidden_ctx`。上下文信息已经通过输入端注入了。

通过这些代码示例的对比，可以更清晰地理解实现Seq2Seq架构时的不同技术细节，以及这些细节对性能和效率的影响。

---

## 五、应用与泛化：不止于文本

Encoder-Decoder 架构的强大之处在于其通用性。它本质上定义了一个“将一种数据形态转换为另一种数据形态”的通用范式，因此其应用远不止于文本到文本的任务。

-   **语音识别 (Audio-to-Text)**：编码器可以是一个处理音频信号的模型（如基于RNN或卷积的模型），提取语音特征并生成上下文向量；解码器则基于此向量生成识别出的文本序列。
-   **图像描述生成 (Image-to-Text)**：编码器可以是一个卷积神经网络（CNN），负责“阅读”整张图片并提取其视觉特征，生成一个概括图片内容的上下文向量；解码器则根据该向量生成一段描述性的文字，实现“看图说话”。
-   **文本到语音 (Text-to-Speech, TTS)**：与语音识别相反，编码器处理输入文本，解码器则生成对应的音频波形数据。
-   **问答系统 (QA)**：模型可以将一篇参考文章和用户提问一起编码，然后解码生成问题的答案。
-   **任务范式统一**：甚至传统的分类任务也可以被“生成化”。例如，在文本分类任务中，可以构造一个特殊的输入（即Prompt），引导模型直接生成类别名称。这种方式极大地统一了不同NLP任务的处理范式。一个具体的例子如下：
    - **输入**: `"请判断以下文本的类别。可选类别列表为：[科技, 体育, 财经]。文本：中国队在世界游泳锦标赛上获得了五枚金牌。"`
    - **期望输出**: `"体育"`

通过替换不同的编码器和解码器实现，Seq2Seq 框架可以灵活地应用于各种跨模态的转换任务中，展现了其作为现代AI模型核心思想的巨大潜力。

## 六、Seq2Seq 的局限性：信息瓶颈

尽管基于 Seq2Seq 架构的模型取得了巨大成功，但它也存在一个明显的缺陷：**信息瓶颈 (Information Bottleneck)**。

编码器必须将输入序列的所有信息，无论其长短，都压缩成一个**固定长度**的上下文向量 $C$。当输入句子很长时，这个向量很难承载全部的语义细节，模型可能会“遗忘”掉句子开头的关键信息，导致生成质量下降。这就好比让一个人记住一篇长文的所有细节，然后仅凭记忆复述，难度非常大。

可以用一个更具体的例子来理解这个问题。假设在做一个对联生成的任务，上联是“风云三尺剑”。在生成下联时，期望第一个字（如“花”）能够主要参考上联的第一个字“风”，第二个字（如“鸟”）主要参考“云”，以此类推，形成对仗。

然而，在标准的 Seq2Seq 架构中，存在两个核心问题：
1.  **信息稀释**：“风”这个词的信息经过多步 RNN 传递后，在最终的上下文向量 $C$ 中可能已经变得非常微弱。
2.  **信息无差别（缺乏倾向性）**：解码器在生成每一个字（“花”、“鸟”、“一”、“创”、“书”）时，所依赖的全局信息都是同一个、包含了整个上联概要的上下文向量 $C$。它没有一种机制去“特别关注”或“倾向于”当前生成位置所对应的输入部分。

> 即使采用 `2.4.2` 节讨论的第二种方式，将 $C$ 作为解码器**每个时间步的额外输入**，问题依然存在。因为每个时间步输入的都是**同一个 $C$**，模型仍然无法学会有选择性地、有侧重地利用输入信息，缺乏这种动态的“倾向性”。

为了解决这个信息瓶颈和对齐问题，研究者们引入了**注意力机制 (Attention Mechanism)**。Attention 允许解码器在生成每个词元时，都能“回头看”并动态地计算一个权重分布，从而重点关注输入序列的不同部分，而不是仅仅依赖于单一的上下文向量。这极大地提升了长序列任务的性能，并直接催生了后来更强大的 Transformer 模型。我们将在下一节详细探讨这一革命性的机制。

---

## 参考文献

[^1]: [Sutskever, I., Vinyals, O., & Le, Q. V. (2014). *Sequence to sequence learning with neural networks*. Advances in Neural Information Processing Systems, 27.](https://proceedings.neurips.cc/paper/2014/hash/a14ac55a4f27472c5d894ec1c3c743d2-Abstract.html)

[^2]: [Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau,D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). *Learning phrase representations using RNN encoder-decoder for statistical machine translation*. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP).](https://aclanthology.org/D14-1179/)

[^3]: [PyTorch Documentation. *torch.nn.CrossEntropyLoss*.](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

[^4]: [Bengio, S., Vinyals, O., Jaitly, N., & Shazeer, N. (2015). *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks*. Advances in Neural Information Processing Systems (NeurIPS).](https://proceedings.neurips.cc/paper/2015/hash/e995f98d56967d946471af29d7bf9a1e-Abstract.html)
