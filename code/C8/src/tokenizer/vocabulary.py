import json
from ..utils.file_io import load_json

class Vocabulary:
    """
    负责管理词汇表和 token 到 id 的映射。
    """
    def __init__(self, vocab_path):
        self.tokens = load_json(vocab_path)
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.id_to_token = {i: token for i, token in enumerate(self.tokens)}
        self.pad_id = self.token_to_id['<PAD>']
        self.unk_id = self.token_to_id['<UNK>']

    def __len__(self):
        return len(self.tokens)

    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    @classmethod
    def load_from_file(cls, vocab_path):
        return cls(vocab_path)
