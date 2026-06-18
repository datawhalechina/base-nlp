"""
示例三：多轮对话
维护对话历史，实现上下文连贯的多轮交互。
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def multi_turn_chat() -> None:
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 MINIMAX_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.minimax.io/v1",
    )

    messages: list[dict] = [
        {
            "role": "system",
            "content": "你是一个 NLP 学习助手，负责解答 base-llm 教程相关问题。",
        },
    ]

    print("多轮对话示例（输入 'quit' 退出）")
    print("-" * 40)

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() == "quit":
            print("对话结束。")
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


if __name__ == "__main__":
    multi_turn_chat()
