# 通过 MiniMax API 快速入门大模型 API 调用

> **阅读建议**：适合在读完第 5 章（预训练模型）或第 6 章（深入大模型架构）之后阅读。本专题介绍如何通过商业云端 API 调用大模型，作为本地模型部署的补充视角。

## 背景与动机

在本教程主线中，我们学习了如何从零理解、训练和部署大模型。然而在工程实践中，直接调用商业大模型 API 是更常见的选择——它无需本地 GPU 资源，开箱即用，且具备极大的上下文窗口，非常适合快速验证想法、构建应用原型。

**MiniMax** 是国内领先的大模型提供商，其旗舰模型 **MiniMax-M3** 具备 **512K tokens 超长上下文**，并提供兼容 OpenAI 接口规范的 API，方便开发者无缝切换。

本专题将带你：
1. 了解 MiniMax API 的核心特性
2. 完成环境配置与第一次 API 调用
3. 实践多轮对话、流式输出等常用功能
4. 理解 API 调用与本地推理的异同

---

## MiniMax 模型简介

| 模型 | 上下文窗口 | 适用场景 |
|------|-----------|----------|
| `MiniMax-M3` | 512K tokens | 旗舰模型，长文档、复杂推理任务首选 |
| `MiniMax-M2.7` | 204K tokens | 上一代高精度模型，兼容老项目 |
| `MiniMax-M2.7-highspeed` | 204K tokens | 上一代高速版，对响应速度有要求的场景 |

> **注意**：MiniMax 所有模型的 `temperature` 参数范围为 **(0.0, 1.0]**（不包含 0，包含 1），与部分其他提供商不同，请注意调整。

---

## 环境准备

### 1. 获取 API Key

前往 [MiniMax 开放平台](https://www.minimax.io) 注册并创建 API Key。

### 2. 安装依赖

MiniMax 的 API 完全兼容 OpenAI SDK，无需安装额外客户端：

```bash
pip install openai python-dotenv
```

### 3. 配置环境变量

创建 `.env` 文件（不要将其提交到 Git）：

```bash
MINIMAX_API_KEY=your_api_key_here
```

---

## 快速开始

### 示例一：基础对话

```python
# code/01_basic_chat.py
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("MINIMAX_API_KEY"),
    base_url="https://api.minimax.io/v1",
)

response = client.chat.completions.create(
    model="MiniMax-M3",
    messages=[
        {
            "role": "system",
            "content": "你是一个专业的 NLP 教学助手，擅长解释自然语言处理概念。",
        },
        {
            "role": "user",
            "content": "请用简洁的语言解释 Transformer 中 Self-Attention 的核心思想。",
        },
    ],
    temperature=0.7,  # MiniMax: temperature ∈ (0.0, 1.0]
)

print(response.choices[0].message.content)
```

运行：

```bash
python code/01_basic_chat.py
```

---

### 示例二：流式输出

对于长文本生成，流式输出（Streaming）能显著提升用户体验——模型生成内容时实时返回，而非等待全部完成：

```python
# code/02_streaming.py
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("MINIMAX_API_KEY"),
    base_url="https://api.minimax.io/v1",
)

stream = client.chat.completions.create(
    model="MiniMax-M3",
    messages=[
        {"role": "user", "content": "请详细介绍 BERT 和 GPT 在预训练目标上的主要区别。"},
    ],
    temperature=0.7,
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # 换行
```

---

### 示例三：多轮对话

```python
# code/03_multi_turn.py
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("MINIMAX_API_KEY"),
    base_url="https://api.minimax.io/v1",
)

messages = [
    {"role": "system", "content": "你是一个 NLP 学习助手，负责解答 base-llm 教程相关问题。"},
]

print("多轮对话示例（输入 'quit' 退出）")
print("-" * 40)

while True:
    user_input = input("你: ").strip()
    if user_input.lower() == "quit":
        break
    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="MiniMax-M3",
        messages=messages,
        temperature=0.7,
    )

    assistant_reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_reply})

    print(f"助手: {assistant_reply}\n")
```

---

### 示例四：长文档分析（利用 512K 超长上下文）

MiniMax-M3 提供的 512K 超长上下文在处理长文档时具有明显优势。以下示例展示如何将长文本一次性送入模型进行分析：

```python
# code/04_long_context.py
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("MINIMAX_API_KEY"),
    base_url="https://api.minimax.io/v1",
)

# 模拟一段较长的技术文档
long_document = """
[Transformer 论文摘要（模拟长文档）]

Attention Is All You Need

Abstract:
The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms, dispensing with recurrence and convolutions
entirely. Experiments on two machine translation tasks show these models to be
superior in quality while being more parallelizable and requiring significantly
less time to train.

... [此处可放入实际的长文档内容，MiniMax-M3 支持最长 512K tokens] ...
"""

response = client.chat.completions.create(
    model="MiniMax-M3",
    messages=[
        {
            "role": "user",
            "content": f"请对以下技术文档进行摘要，并提炼出 3 个核心创新点：\n\n{long_document}",
        }
    ],
    temperature=0.5,
)

print(response.choices[0].message.content)
print(f"\n--- Token 使用情况 ---")
print(f"输入 tokens: {response.usage.prompt_tokens}")
print(f"输出 tokens: {response.usage.completion_tokens}")
```

---

## API 调用 vs 本地推理：核心差异

| 维度 | API 调用（MiniMax） | 本地推理（HuggingFace） |
|------|-------------------|------------------------|
| **硬件要求** | 无需 GPU | 需要高显存 GPU |
| **上下文长度** | 512K tokens（M3） | 受限于 GPU 显存 |
| **延迟** | 依赖网络 | 本地推理延迟低 |
| **成本** | 按 token 计费 | 一次性硬件投入 |
| **数据隐私** | 数据发送至云端 | 数据完全本地 |
| **模型控制** | 固定模型版本 | 可自定义微调 |

> **选型建议**：原型开发、快速验证 → 优先 API；生产部署、数据敏感 → 考虑本地部署。

---

## 与 OpenAI 的兼容性

MiniMax 的 API 端点完全兼容 OpenAI SDK，只需修改两处：

```python
# OpenAI 配置
client = OpenAI(
    api_key="sk-...",
    # base_url 默认为 https://api.openai.com/v1
)

# 切换到 MiniMax（仅需修改这两处）
client = OpenAI(
    api_key=os.environ.get("MINIMAX_API_KEY"),
    base_url="https://api.minimax.io/v1",  # ← 修改 base_url
)
# model 参数改为 "MiniMax-M3" 等 MiniMax 模型名
```

这意味着已有的基于 OpenAI SDK 的代码，可以以最小改动切换到 MiniMax。

---

## 实践练习

1. **基础练习**：修改 `01_basic_chat.py`，尝试不同的 `temperature` 值（如 0.1、0.5、1.0），观察输出的差异。

2. **进阶练习**：使用 `MiniMax-M3` 和 `MiniMax-M2.7-highspeed` 分别请求同一问题，对比响应时间与质量。

3. **综合实践**：选取一篇英文论文的 Abstract，让模型翻译并总结，体验长上下文能力。

---

## 经验总结

- **temperature 范围**：MiniMax 的 temperature 参数区间为 (0.0, 1.0]，设置为 0 会报错，建议最小值设为 0.01。
- **模型选择**：默认推荐 `MiniMax-M3`（512K 上下文，旗舰能力）；对响应延迟有要求的场景可使用上一代 `MiniMax-M2.7-highspeed`。
- **长文档处理**：M3 的 512K 超长上下文可以一次性处理大量文本，避免繁琐的文本切分。
- **成本控制**：开发阶段可设置较短的 `max_tokens` 限制输出长度，节省 API 费用。

---

## 参考资料

- [MiniMax 开放平台文档](https://www.minimax.io/docs)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- Base LLM 主教程：[第 5 章 - 预训练模型](../../docs/chapter5/14_GPT.md)、[第 6 章 - 深入大模型架构](../../docs/chapter6/17_handcraft_llama2.md)
