"""
示例四：长文档分析
演示 MiniMax-M3 512K 超长上下文窗口的处理能力。
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 模拟长文档（实际使用时可替换为真实长文档内容）
SAMPLE_DOCUMENT = """
Attention Is All You Need (Vaswani et al., 2017)

Abstract:
The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms, dispensing with recurrence and convolutions
entirely. Experiments on two machine translation tasks show these models to be
superior in quality while being more parallelizable and requiring significantly
less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-
to-German translation task, improving over the existing best results, including
ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task,
our model establishes a new single-model state-of-the-art BLEU score of 41.8 after
training for 3.5 days on eight GPUs, a small fraction of the training costs of the
best models from the literature.

1. Introduction:
Recurrent neural networks, long short-term memory and gated recurrent neural
networks in particular, have been firmly established as state of the art approaches
in sequence modeling and transduction problems such as language modeling and
machine translation. Numerous efforts have since continued to push the boundaries
of recurrent language models and encoder-decoder architectures.

Recurrent models typically factor computation along the symbol positions of the
input and output sequences. Aligning the positions to steps in computation time,
they generate a sequence of hidden states ht, as a function of the previous hidden
state ht−1 and the input for position t. This inherently sequential nature precludes
parallelization within training examples, which becomes critical at longer sequence
lengths, as memory constraints limit batching across examples.

The Transformer architecture moves away from recurrent connections, instead relying
entirely on an attention mechanism to draw global dependencies between input and
output. The Transformer allows for significantly more parallelization and can reach
a new state of the art in translation quality after being trained for as little as
twelve hours on eight P100 GPUs.
"""


def analyze_long_document(document: str = SAMPLE_DOCUMENT) -> None:
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 MINIMAX_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.minimax.io/v1",
    )

    print("=== 长文档分析（MiniMax-M3，512K 超长上下文）===\n")

    response = client.chat.completions.create(
        model="MiniMax-M3",
        messages=[
            {
                "role": "user",
                "content": (
                    "请对以下技术文档进行分析，输出：\n"
                    "1. 一段不超过 100 字的摘要\n"
                    "2. 3 个核心创新点（每点一句话）\n\n"
                    f"文档内容：\n{document}"
                ),
            }
        ],
        temperature=0.5,
    )

    print(response.choices[0].message.content)
    print(f"\n--- Token 使用情况 ---")
    print(f"输入 tokens: {response.usage.prompt_tokens}")
    print(f"输出 tokens: {response.usage.completion_tokens}")
    print(f"（MiniMax-M3 支持最大 512K 输入 tokens）")


if __name__ == "__main__":
    analyze_long_document()
