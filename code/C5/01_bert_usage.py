import torch
import os
from transformers import AutoTokenizer, AutoModel

# 1. 环境和模型配置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # 可选：设置镜像
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-chinese"
texts = ["我来自中国", "我喜欢自然语言处理"]

# 2. 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

print("\\n--- BERT 模型结构 ---")
print(model)

# 3. 文本预处理
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

# 打印 Tokenizer 的完整输出，以理解其内部结构
print("--- Tokenizer 输出 ---")
for key, value in inputs.items():
    print(f"{key}: \n{value}\n")

# 4. 模型推理
with torch.no_grad():
    outputs = model(**inputs)

# 5. 提取特征
last_hidden_state = outputs.last_hidden_state
sentence_features_pooler = getattr(outputs, "pooler_output", None)

# (1) 提取句子级别的特征向量 ([CLS] token)
sentence_features = last_hidden_state[:, 0, :]

# (2) 提取第一个句子的词元级别特征
first_sentence_tokens = last_hidden_state[0, 1:6, :]


print("\n--- 特征提取结果 ---")
print(f"句子特征 shape: {sentence_features.shape}")
if sentence_features_pooler is not None:
    print(f"pooler_output shape: {sentence_features_pooler.shape}")
print(f"第一个句子的词元特征 shape: {first_sentence_tokens.shape}")