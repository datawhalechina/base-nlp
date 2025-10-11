import torch
from torch.utils.data import Dataset
from ..utils.file_io import load_json


class NerDataset(Dataset):
    """
    处理 NER 数据，并将其转换为适用于 PyTorch 模型的格式。
    """
    def __init__(self, data_path, tokenizer, tag_map):
        self.tokenizer = tokenizer
        self.tag_to_id = tag_map
        self.records = load_json(data_path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        text = record['text']
        tokens = self.tokenizer.text_to_tokens(text)
        token_ids = self.tokenizer.tokens_to_ids(tokens)

        tags = ['O'] * len(tokens)
        for entity in record.get('entities', []):
            entity_type = entity['type']
            start = entity['start_idx']
            end = entity['end_idx'] - 1

            if end >= len(tokens): continue

            if start == end:
                tags[start] = f'S-{entity_type}'
            else:
                tags[start] = f'B-{entity_type}'
                tags[end] = f'E-{entity_type}'
                for i in range(start + 1, end):
                    tags[i] = f'M-{entity_type}'
        
        label_ids = [self.tag_to_id.get(tag, self.tag_to_id['O']) for tag in tags]

        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "label_ids": torch.tensor(label_ids, dtype=torch.long)
        }
