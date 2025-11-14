import torch
import json
import os
import argparse
from src.models.ner_model import BiGRUNerNetWork
from src.tokenizer.vocabulary import Vocabulary
from src.tokenizer.char_tokenizer import CharTokenizer
from src.utils.file_io import load_json

class NerPredictor:
    def __init__(self, model_dir, device='cpu'):
        self.device = torch.device(device)
        
        # --- 1. 加载配置文件以获取模型参数 ---
        config_path = os.path.join(model_dir, 'config.json')
        self.config = load_json(config_path)

        # --- 2. 加载词汇表和标签映射 ---
        vocab_path = os.path.join(self.config["data_dir"], self.config["vocab_file"])
        tags_path = os.path.join(self.config["data_dir"], self.config["tags_file"])

        self.vocab = Vocabulary.load_from_file(vocab_path)
        self.tokenizer = CharTokenizer(self.vocab)
        tag_map = load_json(tags_path)
        self.id2tag = {v: k for k, v in tag_map.items()}

        # --- 3. 初始化模型并加载权重 ---
        self.model = BiGRUNerNetWork(
            vocab_size=len(self.vocab),
            hidden_size=self.config["hidden_size"],
            num_tags=len(tag_map),
            num_gru_layers=self.config["num_gru_layers"]
        )
        model_path = os.path.join(model_dir, 'best_model.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        tokens = self.tokenizer.text_to_tokens(text)
        token_ids = self.tokenizer.tokens_to_ids(tokens)
        
        # --- 预处理 ---
        token_ids_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.ones_like(token_ids_tensor)

        # --- 模型预测 ---
        with torch.no_grad():
            logits = self.model(token_ids_tensor, attention_mask)
        
        # --- 后处理 ---
        predictions = torch.argmax(logits, dim=-1).squeeze(0)
        tags = [self.id2tag[id_.item()] for id_ in predictions]

        return self._extract_entities(tokens, tags)

    def _extract_entities(self, tokens, tags):
        entities = []
        current_entity = None
        for i, tag in enumerate(tags):
            if tag.startswith('B-'):
                # 如果前一个实体未正确结束，则放弃
                if current_entity:
                    pass # 或者可以根据业务逻辑决定是否保存不完整的实体
                current_entity = {"text": tokens[i], "type": tag[2:], "start": i}
            elif tag.startswith('M-'):
                # M 标签必须跟在 B- 或 M- 之后
                if current_entity and current_entity["type"] == tag[2:]:
                    current_entity["text"] += tokens[i]
                else:
                    # 非法 M 标签，重置当前实体
                    current_entity = None
            elif tag.startswith('E-'):
                # E 标签必须跟在 B- 或 M- 之后
                if current_entity and current_entity["type"] == tag[2:]:
                    current_entity["text"] += tokens[i]
                    current_entity["end"] = i + 1
                    entities.append(current_entity)
                # 实体已结束，重置
                current_entity = None
            elif tag.startswith('S-'):
                # S 标签表示单个字符的实体
                # 如果有未结束的实体，则放弃
                current_entity = None
                entities.append({"text": tokens[i], "type": tag[2:], "start": i, "end": i + 1})
            else: # 'O' 标签
                # O 标签意味着没有实体，或者实体已经结束
                current_entity = None
        
        # 循环结束后，不再处理任何未闭合的实体
        return entities

def main():
    parser = argparse.ArgumentParser(description="NER Prediction")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of the saved model and config.")
    parser.add_argument("--text", type=str, required=True, help="Text to predict.")
    args = parser.parse_args()

    predictor = NerPredictor(model_dir=args.model_dir)
    entities = predictor.predict(args.text)
    print(f"Text: {args.text}")
    print(f"Entities: {json.dumps(entities, ensure_ascii=False, indent=2)}")

if __name__ == "__main__":
    main()
