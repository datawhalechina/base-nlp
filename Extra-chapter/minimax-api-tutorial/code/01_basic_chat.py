"""
示例一：基础对话
使用 MiniMax API（兼容 OpenAI SDK）进行单轮对话。
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def basic_chat() -> None:
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 MINIMAX_API_KEY")

    client = OpenAI(
        api_key=api_key,
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

    print("=== MiniMax-M3 回复 ===")
    print(response.choices[0].message.content)
    print(f"\n--- Token 使用情况 ---")
    print(f"输入 tokens: {response.usage.prompt_tokens}")
    print(f"输出 tokens: {response.usage.completion_tokens}")


if __name__ == "__main__":
    basic_chat()
