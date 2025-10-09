from abc import ABC, abstractmethod
from typing import List

class BaseTokenizer(ABC):
    @abstractmethod
    def text_to_tokens(self, text: str) -> List[str]:
        """将文本分割成 token 列表。"""
        raise NotImplementedError

    @abstractmethod
    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """将 token 列表转换为 ID 列表。"""
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        """将文本直接编码为 ID 列表的便捷方法。"""
        tokens = self.text_to_tokens(text)
        return self.tokens_to_ids(tokens)

    @abstractmethod
    def get_pad_id(self) -> int:
        """获取填充 token 的 ID。"""
        raise NotImplementedError
