import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. 环境和模型配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "gpt2"

# --- 2. 加载模型和分词器 ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

# 打印模型结构
print("\n--- 模型结构 ---")
print(model)
print("-" * 50)

# --- 英文示例 ---
print("\n--- 英文示例 ---")
prompt_en = "I like eating fried"
input_ids_en = tokenizer(prompt_en, return_tensors="pt")['input_ids'].to(device)

print(f"英文输入: '{prompt_en}'")
print(f"被编码为 {input_ids_en.shape[1]} 个 token: {tokenizer.convert_ids_to_tokens(input_ids_en[0])}")
print("-" * 30)
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
print(f"\n英文最终生成结果: \n'{full_decoded_text_en}'")
print("-" * 50)


# --- 中文示例 ---

print("\n--- 中文示例 ---")
prompt_zh = "我喜欢吃炸"
input_ids_zh = tokenizer(prompt_zh, return_tensors="pt")['input_ids'].to(device)

print(f"中文输入: '{prompt_zh}'")
print(f"被编码为 {input_ids_zh.shape[1]} 个 token: {tokenizer.convert_ids_to_tokens(input_ids_zh[0])}")
print("-" * 30)
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
print(f"\n中文最终生成结果: \n'{full_decoded_text_zh}'")
print("-" * 50)

# pipeline 应用
from transformers import pipeline

print("\n\n--- Pipeline 快速演示 (英文) ---")
generator = pipeline("text-generation", model=model_name, device=device)
pipeline_outputs = generator(prompt_en, max_new_tokens=5, num_return_sequences=1)
print(pipeline_outputs[0]['generated_text'])
