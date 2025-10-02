import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def normalize_text(text):
    """
    规范化文本，例如将全角字符转换为半角字符。
    """
    full_width = "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ！＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～＂"
    half_width = r"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&'" + r'()*+,-./:;<=>?@[\]^_`{|}~".'
    mapping = str.maketrans(full_width, half_width)
    return text.translate(mapping)


class Vocabulary:
    """
    负责管理词汇表和 token 到 id 的映射。
    """
    def __init__(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.tokens = json.load(f)
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.pad_id = self.token_to_id['<PAD>']
        self.unk_id = self.token_to_id['<UNK>']

    def __len__(self):
        return len(self.tokens)

    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]


class NerDataProcessor(Dataset):
    """
    处理 NER 数据，并将其转换为适用于 PyTorch 模型的格式。
    """
    def __init__(self, data_path, vocab: Vocabulary, tag_map: dict):
        self.vocab = vocab
        self.tag_to_id = tag_map
        with open(data_path, 'r', encoding='utf-8') as f:
            self.records = json.load(f)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        text = normalize_text(record['text'])
        tokens = list(text)
        
        # 将文本 tokens 转换为 ids
        token_ids = self.vocab.convert_tokens_to_ids(tokens)

        # 初始化标签序列为 'O'
        tags = ['O'] * len(tokens)
        for entity in record.get('entities', []):
            entity_type = entity['type']
            start = entity['start_idx']
            end = entity['end_idx'] - 1  # 转换为包含模式

            if end >= len(tokens): continue

            if start == end:
                tags[start] = f'S-{entity_type}'
            else:
                tags[start] = f'B-{entity_type}'
                tags[end] = f'E-{entity_type}'
                for i in range(start + 1, end):
                    tags[i] = f'M-{entity_type}'
        
        # 将标签转换为 ids
        tag_ids = [self.tag_to_id[tag] for tag in tags]

        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "tag_ids": torch.tensor(tag_ids, dtype=torch.long)
        }


def create_ner_dataloader(data_path, vocab, tag_map, batch_size, shuffle=False):
    """
    创建 NER 任务的 DataLoader。
    """
    dataset = NerDataProcessor(data_path, vocab, tag_map)
    
    def collate_batch(batch):
        token_ids_list = [item['token_ids'] for item in batch]
        tag_ids_list = [item['tag_ids'] for item in batch]

        padded_token_ids = pad_sequence(token_ids_list, batch_first=True, padding_value=vocab.pad_id)
        padded_tag_ids = pad_sequence(tag_ids_list, batch_first=True, padding_value=-100) # -100 用于在计算损失时忽略填充部分

        attention_mask = (padded_token_ids != vocab.pad_id).long()

        return padded_token_ids, padded_tag_ids, attention_mask

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)


if __name__ == '__main__':
    # 文件路径
    train_file = './data/CMeEE-V2_train.json'
    vocab_file = './data/vocabulary.json'
    categories_file = './data/categories.json'

    # 1. 加载词汇表和标签映射
    vocabulary = Vocabulary(vocab_path=vocab_file)
    with open(categories_file, 'r', encoding='utf-8') as f:
        tag_map = json.load(f)
    print("词汇表和标签映射加载完成。")

    # 2. 创建 DataLoader
    train_loader = create_ner_dataloader(
        data_path=train_file,
        vocab=vocabulary,
        tag_map=tag_map,
        batch_size=4,
        shuffle=True
    )
    print("DataLoader 创建完成。")

    # 3. 验证一个批次的数据
    print("\n--- 验证一个批次的数据 ---")
    tokens, labels, mask = next(iter(train_loader))
    
    print(f"  Token IDs (shape): {tokens.shape}")
    print(f"  Label IDs (shape): {labels.shape}")
    print(f"  Attention Mask (shape): {mask.shape}")
    print(f"  Token IDs (sample): {tokens[0][:20]}...")
    print(f"  Label IDs (sample): {labels[0][:20]}...")
    print(f"  Attention Mask (sample): {mask[0][:20]}...")
