# 第三节 Hugging Face PEFT 实战

在前两个小节中，深入探讨了参数高效微调（PEFT）的理论背景和主流方法，特别是 LoRA 的核心原理。理论知识为我们指明了方向，但要真正驾驭这些技术，需要一个强大而易用的工具。本节将正式进入实战环节，聚焦于当前社区最核心的 PEFT 工具库——Hugging Face 的 `peft`。[^1]

<p align="center">
  <img src="./images/12_3_1.png" width="90%" alt="Hugging Face PEFT 库官方文档首页" />
  <br />
  <em>图 12-11：Hugging Face PEFT 库官方文档首页</em>
</p>

`peft` 库的设计理念与 Hugging Face 生态系统一脉相承，都是希望将复杂的 PEFT 技术（如 LoRA, Prefix Tuning, Adapter 等）抽象成统一、简洁的接口，让开发者能够以最小的代码改动，将这些高效微调方法无缝地应用到 Hugging Face Hub 上的大模型上。如图 12-11，`peft` 库的官方文档将其内容划分为快速入门、方法指南、概念指南和参考手册，便于开发者上手。

## 一、`peft` 库的设计理念

要理解 `peft` 库，首先要明白它并非要取代基础的模型库（例如 `transformers`），而是作为其 **插件** 或 **增强模块** 而存在。它巧妙地继承并扩展了 Hugging Face 的设计哲学，尤其是自动化配置的理念。

可以从软件工程的角度来理解它们的关系：
-   **基础预训练模型**：可以看作一个功能强大但行为固定的 **框架**。它具备强大的通用能力，但要改变其内部逻辑以适应特定任务，通常需要修改其源码或进行完整的重新训练，成本高昂。
-   **`peft` 库**：则提供了一套 **插件接口** 和多种 **插件实现**（如 LoRA、Prompt Tuning 等）。这些插件可以在不侵入核心库代码的前提下，动态地改变或扩展其功能。
-   **`PeftConfig`**：相当于一个 **配置文件**。这份文件详细定义了要加载哪个插件（例如 `peft_type='LORA'`），以及这个插件的具体参数（例如 LoRA 的 `r`、`lora_alpha` 等超参数）。
-   **`get_peft_model` 函数**：扮演着一个 **插件管理器** 的角色。它接收核心库（`base_model`）和配置文件（`peft_config`），然后依据配置，将相应的插件模块（例如 LoRA 的低秩矩阵）动态地注入到核心库模型的指定位置，生成一个功能增强后的新对象 `PeftModel`。

通过这种方式，无需改动庞大的基础模型本身（冻结其大部分权重），只需定义、训练和切换不同的轻量级插件（Adapter），就能让模型高效地适应各种下游任务。这不仅节省了大量的计算和存储资源，也使得模型的管理和部署变得极为灵活。

## 二、`peft` 库的核心抽象

`peft` 库通过几个核心的类和函数，实现了对各种 PEFT 方法的统一封装，使其遵循一致的调用逻辑。

### 2.1 `PeftConfig`：PEFT 方法的“声明式配置”

`PeftConfig` 是所有 PEFT 方法配置的基类，它采用声明式的方式定义了微调的策略。其中最重要的两个通用参数是：

-   `peft_type`：一个枚举类型，用于 **指定要使用的 PEFT 插件类型**。例如，`PeftType.LORA` 明确表示使用 LoRA 方法。这是 `peft` 库能够自动检索和应用不同微调算法的关键。
-   `task_type`：一个枚举类型，用于 **指定模型的下游任务类型**。例如，`TaskType.CAUSAL_LM` 用于自回归语言模型（如 GPT），`TaskType.SEQ_2_SEQ_LM` 用于序列到序列模型（如 T5）。这个参数能够帮助 `peft` 库为特定任务对模型的头部（Head）或其他结构进行正确的适配。

针对每一种具体的 PEFT 方法，`peft` 库都提供了一个继承自 `PeftConfig` 的子类，例如 `LoraConfig`、`PromptTuningConfig` 等。以 `LoraConfig` 为例，它包含了 LoRA 方法专属的超参数，这些参数直接源于 LoRA 论文中的定义：

-   `r`：LoRA 的**秩（rank）**，决定了低秩矩阵 A 和 B 的中间维度 `(d, r)` 和 `(r, k)`。它是控制新增参数量和模型适应能力的核心超参数。
-   `lora_alpha`：LoRA 的**缩放因子**。在 LoRA 的计算中，低秩矩阵的输出 `BAx` 会乘以一个缩放系数 `alpha/r`。`lora_alpha` 就是这个公式中的 `alpha`，它用于调整低秩适应矩阵与原始权重矩阵合并时的尺度。
-   `target_modules`：一个字符串或正则表达式列表，用于**精确指定要将 LoRA 应用于基础模型中的哪些模块**。例如，`["q_proj", "v_proj"]` 表示仅在 Transformer 层的 `query` 和 `value` 投影矩阵上应用 LoRA。
-   `lora_dropout`：在 LoRA 层上应用的 Dropout 比例，用于防止过拟合。
-   `bias`：偏置（bias）参数的训练方式，可选值为 `'none'`（冻结所有 bias）、`'all'`（训练所有 bias）或 `'lora_only'`（仅训练 LoRA 模块自身的 bias）。

在本次的文本生成实战中，正是通过这两个关键参数来告知 `peft` 库我们的意图：
-   `peft_type=PeftType.LORA`：明确我们选择 LoRA 作为本次微调的“插件”。
-   `task_type=TaskType.CAUSAL_LM`：指明我们的目标任务是因果语言模型（即文本生成），`peft` 库会据此对模型的结构进行正确的适配，例如确保只有语言模型头部的参数参与计算损失。

### 2.2 `PeftModel` 与 `get_peft_model`：模型动态注入与封装

`get_peft_model` 是 `peft` 库中的核心工厂函数。它接收一个原始的预训练模型和一个 `PeftConfig` 对象，然后执行以下操作：
1.  解析 `PeftConfig`，确定要使用的 PEFT 方法和相关参数。
2.  遍历基础模型的网络结构，根据 `target_modules` 找到需要注入 LoRA 模块的目标层。
3.  将原始的目标层（如 `nn.Linear`）替换/封装为注入了 LoRA 的线性模块（如 LoraLinear 或其 k-bit 量化变体）。该模块内部保留冻结的原始权重，并引入可训练的低秩分支 A 和 B。
4.  返回一个 `PeftModel` 实例。

返回的 `peft_model` 对象是一个高度封装的模型。它内部保留了对原始基础模型的引用，并通过动态修改其 `forward` 传递路径，实现了 LoRA 逻辑的注入。这个 `peft_model` 实例拥有与基础模型完全兼容的接口，可以直接用于 `Trainer` 或自定义的训练循环中。

一个非常有用的调试方法是 `peft_model.print_trainable_parameters()`，它可以精确计算并打印出模型中可训练参数的数量及其占总参数量的比例，可以直观地感受到 PEFT 在节约资源上的巨大优势。

## 三、LoRA 微调实战流程

结合 `peft` 库，可以形成一个标准的 LoRA 微调流程。下面将以微调 `EleutherAI/pythia-2.8b-deduped` 模型为例，展示一个与配套代码仓库中 `code/C11/peft_lora_pythia-2.8b.ipynb` Notebook 完全一致的实战 Pipeline。

### 3.1 加载依赖、基础模型与分词器

首先，加载所需的库和预训练模型。为了在消费级硬件上运行数十亿参数的大模型，关键在于采用**量化**技术。这里，将使用 `bitsandbytes` 库，在加载模型时直接对其进行 8-bit 量化，并指定 `dtype=torch.float16` 以进一步优化显存。

根据 `transformers` 库的最新实践，现已不再推荐使用已被弃用的 `load_in_8bit=True` 参数，而是通过定义一个 `BitsAndBytesConfig` 对象，并将其传递给 `quantization_config` 参数来精确地控制量化行为。这种方式更灵活，也是未来的标准做法。

同时，通过设置 `device_map="auto"`，可以让 `accelerate` 库自动地、智能地将模型层分配到可用的硬件上（例如，将所有层都放到唯一的 GPU 上）。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_id = "EleutherAI/pythia-2.8b-deduped"

# --- 使用 BitsAndBytesConfig 定义 8-bit 量化配置 ---
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

# 加载模型，并将量化配置传给 `quantization_config` 参数
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    dtype=torch.float16,
    device_map="auto",
)
```

执行完这段代码后，如果打印 `model` 对象，你会看到模型架构的详细信息。其中，类似 `(query_key_value): Linear8bitLt(in_features=2560, out_features=7680, bias=True)` 的层表明，原始的 `nn.Linear` 已经被成功替换为 8-bit 量化版本 `Linear8bitLt`，这意味着模型加载和量化已成功完成。

```bash
GPTNeoXForCausalLM(
  (gpt_neox): GPTNeoXModel(
    (embed_in): Embedding(50304, 2560)
    (emb_dropout): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0-31): 32 x GPTNeoXLayer(
        (input_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (post_attention_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (post_attention_dropout): Dropout(p=0.0, inplace=False)
        (post_mlp_dropout): Dropout(p=0.0, inplace=False)
        (attention): GPTNeoXAttention(
          (query_key_value): Linear8bitLt(in_features=2560, out_features=7680, bias=True)
          (dense): Linear8bitLt(in_features=2560, out_features=2560, bias=True)
        )
        (mlp): GPTNeoXMLP(
          (dense_h_to_4h): Linear8bitLt(in_features=2560, out_features=10240, bias=True)
          (dense_4h_to_h): Linear8bitLt(in_features=10240, out_features=2560, bias=True)
          (act): GELUActivation()
        )
      )
    )
    (final_layer_norm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
    (rotary_emb): GPTNeoXRotaryEmbedding()
  )
  (embed_out): Linear(in_features=2560, out_features=50304, bias=False)
)
```

模型加载完成后，加载其对应的分词器。对于 Pythia 这类模型，其分词器默认可能没有 `pad_token`。在进行批量训练时，数据整理器（Data Collator）需要 `pad_token` 来将序列填充至相同长度，因此需要手动将其设置为 `eos_token`。

```python
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Pythia模型的tokenizer默认没有pad_token，我们将其设置为eos_token
tokenizer.pad_token = tokenizer.eos_token
```

### 3.2 模型预处理

在使用 `peft` 对 8-bit 量化模型进行微调之前，需要进行一些必要的预处理。`peft` 库提供了一个非常方便的函数 `prepare_model_for_kbit_training` 来完成这项工作。

> **注意**：在 PEFT 0.10.0 及更高版本中，原来的 `prepare_model_for_int8_training` 已被 `prepare_model_for_kbit_training` 替代，新函数同时支持 4-bit 和 8-bit 量化。

这个函数主要执行几个关键操作：
1.  **类型转换**：将模型中一些需要以更高精度（如 FP32）计算的层（例如 LayerNorm）进行类型转换，以保证训练的数值稳定性。
2.  **启用梯度检查点**：调用 `model.gradient_checkpointing_enable()`，这是一种用计算时间换取显存的技术。它在反向传播时会重新计算中间层的激活值，而不是将它们全部存储在显存中，从而显著降低了训练过程中的显存峰值。
3.  **输出嵌入层预处理**：对模型的输出嵌入层进行一些必要的处理，以使其与 LoRA 兼容。
4.  **输入梯度处理**：为需要的输入启用梯度，保证在冻结大部分权重且使用 k-bit 训练时的反向传播兼容性。

```python
from peft import prepare_model_for_kbit_training

# 对量化后的模型进行预处理
model = prepare_model_for_kbit_training(model)
```

### 3.3 定义 LoRA 配置并创建 `PeftModel`

这是整个 PEFT 流程中最核心的一步。此步骤将实例化一个 `LoraConfig` 对象，用于声明式地定义 LoRA 微调的策略，然后使用 `get_peft_model` 函数将其应用到预处理过的基础模型上，从而得到一个 `PeftModel`。

在 `LoraConfig` 中，会详细设置 LoRA 的各个超参数，这些参数的选择直接关系到微调的效果和效率，与在 `02_lora.md` 中讨论的理论紧密相关：
-   `r`：LoRA 的秩（rank）。这是最关键的超参数之一。`r` 越大，意味着低秩矩阵的表达能力越强，可训练的参数也越多。但正如 `02_lora.md` 的实验所示，`r` 并非越大越好，过大的 `r` 可能会增加噪声，且会线性增加可训练参数量。通常建议从 8 或 16 开始尝试。
-   `lora_alpha`：LoRA 的缩放因子。在 `02_lora.md` 中提到，最终的权重更新量会以 `alpha/r` 的比例进行缩放。这意味着，`lora_alpha` 的值可以理解为对学习到的低秩矩阵的“增强系数”。一个常见的做法是将其设置为 `r` 的两倍。
-   `target_modules`：指定要将 LoRA 应用于模型中的哪些模块。这是一个非常关键的参数，因为不同模型的模块命名方式不同。
    > **如何确定 `target_modules`？** 可以先打印出基础模型 `model` 的结构，并以其显示的层命名为准。对于大多数 Transformer 模型，注意力机制中的“查询（Query）”、“键（Key）”和“值（Value）”层（如 `q_proj`, `k_proj`, `v_proj`）是首选。而对于 `Pythia` 或 `GPT-NeoX` 系列模型，其注意力权重常被合并在一个 `query_key_value` 层中，前馈网络（FFN）中的线性层则常见 `dense`、`dense_h_to_4h` 和 `dense_4h_to_h`。将 LoRA 应用于这些层通常都能带来收益。
-   `bias`：偏置参数的训练方式。`'none'` 是最常用的设置，意味着不训练任何偏置参数，这与 LoRA 的原始思想保持一致，以最大化参数效率。在数据量充足的情况下，可以尝试 `'lora_only'`，仅训练 LoRA 模块自身的偏置。

`LoraConfig` 的其他参数（如 `lora_dropout`、`task_type`）也都提供了对微调过程的精细控制。

```python
from peft import LoraConfig, get_peft_model

# 定义 LoRA 配置
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value", "dense"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 应用配置，获得 PEFT 模型
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
# trainable params: 7,864,320 || all params: 2,783,073,280 || trainable%: 0.2826
```
可以看到，可训练参数仅占总参数量的 0.28%，极大地降低了微调的硬件门槛。

### 3.4 数据处理

现在模型已经准备就绪，需要为它准备“教材”——也就是训练数据。

本次微调的目标是让模型学会生成名人名言。这里将使用 `Abirate/english_quotes` 这个数据集，它包含了大量的英文名言。

数据处理流程如下：
1.  **加载数据集**：使用 `datasets` 库从 Hugging Face Hub 下载数据集。
2.  **数据预处理**：定义一个 `tokenize` 函数，该函数会接收一批数据，提取出所关心的 `quote` 字段，然后使用之前加载的分词器 `tokenizer` 对其进行编码，将其转换为模型可以理解的 `input_ids`。
3.  **应用处理**：使用 `dataset.map()` 方法，将 `tokenize` 函数批量应用到整个数据集上。这是 `datasets` 库一个非常高效的特性。

首先，加载数据集并查看一条样本。
```python
from datasets import load_dataset

# 加载数据集
quotes_dataset = load_dataset("Abirate/english_quotes")

# 查看数据集示例
quotes_dataset['train'][0]
```
输出显示了数据集的结构，包含 `quote`、`author` 和 `tags` 字段。
```
{'quote': '“Be yourself; everyone else is already taken.”',
 'author': 'Oscar Wilde',
 'tags': ['be-yourself',
  'gilbert-perreira',
  'honesty',
  'inspirational',
  'misattributed-oscar-wilde',
  'quote-investigator']}
```
接下来，定义分词函数并将其应用到整个数据集上。
```python
# 定义分词函数
def tokenize_quotes(batch):
    # 只对 "quote" 列进行分词
    return tokenizer(batch["quote"], truncation=True)

# 对整个数据集进行分词处理
tokenized_quotes = quotes_dataset.map(tokenize_quotes, batched=True)

# 查看处理后的数据结构，会发现多了 "input_ids" 和 "attention_mask" 两列
tokenized_quotes['train'][0]
```
处理后的数据集新增了模型所需的 `input_ids` 和 `attention_mask` 列。
```
{'quote': '“Be yourself; everyone else is already taken.”',
 'author': 'Oscar Wilde',
 'tags': ['be-yourself',
  'gilbert-perreira',
  'honesty',
  'inspirational',
  'misattributed-oscar-wilde',
  'quote-investigator'],
 'input_ids': [1628, 4678, 4834, 28, 4130, 2010, 310, 2168, 2668, 1425],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

### 3.5 定义 Trainer 并开始训练

`Trainer` 是 `transformers` 库提供的一个高度抽象化的训练器，它封装了标准的 PyTorch 训练循环。只需通过 `TrainingArguments` 定义训练的“策略”，而无需手动编写繁琐的训练代码（如梯度更新、学习率调度、日志记录等）。

在 `TrainingArguments` 中，会设置一些关键的训练参数：
-   `per_device_train_batch_size` & `gradient_accumulation_steps`：这两个参数共同决定了有效批量大小（effective batch size）。`per_device_train_batch_size` 是指每个 GPU 单次前向传播处理的样本数，而 `gradient_accumulation_steps` 则指定了梯度累积的步数。有效批量大小 = `per_device_train_batch_size` * `gradient_accumulation_steps` * `num_gpus`。通过梯度累积，可以在显存有限的情况下，模拟出更大的批量大小，这通常有助于稳定训练过程。
-   `warmup_steps`: 学习率预热的步数。在训练初期，学习率会从一个很小的值线性增加到设定的 `learning_rate`，这有助于模型在开始阶段更好地适应数据。
-   `max_steps`: 训练的总步数。为了快速演示，这里只训练 200 步。
-   `learning_rate`: 学习率，控制模型参数更新的幅度。
-   `fp16`: 启用 16-bit 混合精度训练。这可以在不牺牲太多性能的情况下，进一步减少显存占用并加速训练。

最关键的是，将之前创建的 `PeftModel` 实例直接传递给 `Trainer`。`Trainer` 会足够智能，自动识别出只有 LoRA 相关的参数是可训练的，并在训练时冻结所有其他参数。

除了上述基础参数外，还有两个关于训练策略的要点值得注意：
- **`max_steps` vs `num_train_epochs`**：`TrainingArguments` 允许通过设置 `max_steps`（总训练步数）或 `num_train_epochs`（总训练轮数）来控制训练的总长度。在快速原型验证或演示时，使用 `max_steps` 可以精确控制训练量，便于快速看到结果。在正式的项目中，使用 `num_train_epochs` 更为常见，它能确保模型完整地学习过所有训练数据指定的轮数。
- **验证集的缺失**：在专业的训练流程中，通常会从数据集中划分出一部分作为验证集（validation set），并在 `TrainingArguments` 中通过 `evaluation_strategy` 参数设置评估时机（例如，每 N 步或每个 epoch 结束后），以便监控模型是否过拟合，并据此进行早停（Early Stopping）等操作。为了简化演示流程，本教程省略了这一环节。

```python
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# 推荐操作：关闭缓存可提高训练效率
peft_model.config.use_cache = False

# 定义训练参数
train_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    max_steps=200,
    learning_rate=2e-4,
    fp16=True, # 启用混合精度训练
    logging_steps=1,
    output_dir="outputs",
)

# 数据整理器，用于处理批量数据
quote_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# 实例化 Trainer
trainer = Trainer(
    model=peft_model,
    train_dataset=tokenized_quotes["train"],
    args=train_args,
    data_collator=quote_collator,
)

# 开始训练
trainer.train()
```
执行 `trainer.train()` 后，控制台会实时打印训练日志。训练完成后，`train` 方法会返回一个包含所有训练指标的 `TrainOutput` 对象，方便进行分析和记录。
```
TrainOutput(global_step=200, training_loss=2.308660719394684, metrics={'train_runtime': 517.6098, 'train_samples_per_second': 6.182, 'train_steps_per_second': 0.386, 'total_flos': 4566109437050880.0, 'train_loss': 2.308660719394684, 'epoch': 1.2743221690590112})
```

### 3.6 模型保存与推理

训练完成后，可以将学到的知识——也就是轻量的 LoRA 适配器——保存下来，以备将来使用。

对 `PeftModel`（即 `peft_model`）调用 `save_pretrained()` 时，`peft` 会只保存增量的、可训练的适配器权重，而不是整个庞大的基础模型。通常，保存下来的文件（`adapter_model.safetensors` 和 `adapter_config.json`）只有几十 MB。

> **合并权重**
>
> 正如前文中所讨论的，LoRA 的一个核心优势是它不会在推理时引入额外的延迟。这是因为它训练出的旁路矩阵 $A$ 和 $B$ 可以被 **合并（merge）** 回原始的权重矩阵中。训练完成后，可以调用 `merged_model = peft_model.merge_and_unload()` 方法，它会返回一个标准的 `transformers` 模型，其权重已经包含了 LoRA 的更新。这个 `merged_model` 的结构与原始模型完全一致，因此可以像任何普通模型一样进行部署，而没有任何额外的计算开销。
> 若基础模型以 8/4-bit 量化加载，合并后返回的标准模型通常会转为 FP16/FP32；若需继续以 k-bit 部署，可在合并后按需重新量化。

为了验证微调的效果，可以进行一次推理测试，观察模型在续写名言开头的表现。

### 推理配置要点

为了获得最佳的推理效果并避免警告，需要注意以下几点：
1. **传递 attention_mask**：显式传递 `attention_mask`，确保模型能够正确识别有效的 token。
2. **启用采样**：设置 `do_sample=True` 以启用温度采样和核采样参数。
3. **启用 use_cache**：推理前将 `use_cache=True` 可提升生成效率；训练阶段通常配合梯度检查点将其关闭。

### 生成参数说明
-   `max_length`: 生成文本的最大长度（包括输入）。
-   `do_sample`: 是否使用采样策略。设置为 `True` 时，`temperature`、`top_p`、`top_k` 才会生效。
-   `temperature`: 控制生成的随机性。较低的值（如 0.7）会使生成更具确定性，而较高的值则会增加多样性。
-   `top_p`: 核采样（nucleus sampling）的概率阈值。只考虑累积概率达到 `top_p` 的最小 token 集合。
-   `top_k`: 每步只从概率最高的 `k` 个 token 中采样。
-   `repetition_penalty`: 重复惩罚因子，大于 1.0 会降低重复内容的概率。

```python
# 将模型设置为评估模式
peft_model.eval()

# 设置 pad_token_id 到模型配置中（避免警告）
peft_model.config.pad_token_id = tokenizer.pad_token_id

prompt = "Be yourself; everyone"

# 对输入进行分词，并获取 attention_mask
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(peft_model.device)
attention_mask = inputs["attention_mask"].to(peft_model.device)

# 生成文本
with torch.no_grad():
    # 使用 autocast 提高混合精度推理的效率
    with torch.amp.autocast('cuda'):
        outputs = peft_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=50,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id
        )

# 解码并打印结果
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
decoded_output
```

输出如下：
```bash
'Be yourself; everyone else is taken.” - Oscar Wilde"I have found that people will forget what you said, people will forget what you did, but people will never forget how you made them feel” Maya Angelou“The worst thing we'
```

从输出可以看到，模型成功地补全了这句来自奥斯卡·王尔德的名言，并且还继续生成了另一句风格相似的名言。这表明，仅仅通过200步的微调，模型就已经从数据集中大致学习到了名言的风格和内容，证明了 PEFT 方法的高效性。

> **模型输出的非确定性**
>
> 大语言模型输出的非确定性主要来源于其解码阶段的 **采样策略**。当 `do_sample=True` 时，模型会根据计算出的词汇表概率分布进行随机抽样，而不是像确定性的贪心搜索那样总是选择概率最高的词。`temperature`、`top_p` 等参数正是用来调节这种抽样过程的随机程度的。
>
> 所以，这些采样参数是引入输出多样性的**主要和意图性**的来源。除此之外，底层的CUDA算子、浮点数计算精度等因素也可能导致即使在固定随机种子的情况下，两次运行结果仍存在微小差异，但这并非主要原因。在本地运行时得到与文档不完全相同的结果，属于正常现象。

## 四、小结

本节通过一个完整的端到端实例，亲手实践了如何利用 `peft` 库对一个数十亿参数的大语言模型进行 LoRA 参数高效微调。内容涵盖从加载量化模型、准备数据，到配置 LoRA、定义 `Trainer` 并进行训练，最终验证了微调的效果。

`peft` 库将复杂的 PEFT 技术封装在简洁、统一的接口背后，极大地降低了研究者和开发者使用这些前沿技术的门槛。通过 `PeftConfig` 和 `get_peft_model` 这两个核心抽象，可以轻松地将 LoRA 等方法应用到 `transformers` 生态的任意模型上，实现“四两拨千斤”的高效微调。随着 AdaLoRA、QLoRA 等更先进技术的集成，在资源有限的条件下驾驭大模型将变得越来越普及。

---

## 参考文献

[^1]: [Hugging Face PEFT Documentation. (2024).](https://huggingface.co/docs/peft/index)
