"""
示例二：流式输出
实时获取模型生成内容，提升用户体验。
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def streaming_chat() -> None:
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 MINIMAX_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.minimax.io/v1",
    )

    print("=== 流式输出示例（MiniMax-M3）===")
    print()

    stream = client.chat.completions.create(
        model="MiniMax-M3",
        messages=[
            {
                "role": "user",
                "content": "请详细介绍 BERT 和 GPT 在预训练目标上的主要区别。",
            },
        ],
        temperature=0.7,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)

    print()  # 换行


if __name__ == "__main__":
    streaming_chat()
