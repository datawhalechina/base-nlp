from .vocabulary import Vocabulary
from .base import BaseTokenizer


def normalize_text(text):
    """
    规范化文本，例如将全角字符转换为半角字符。
    """
    full_width = "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ！＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～＂"
    half_width = r"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&'" + r'()*+,-./:;<=>?@[\]^_`{|}~".'
    mapping = str.maketrans(full_width, half_width)
    return text.translate(mapping)


class CharTokenizer(BaseTokenizer):
    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab

    def text_to_tokens(self, text: str):
        normalized_text = normalize_text(text)
        tokens = list(normalized_text)
        return tokens

    def tokens_to_ids(self, tokens: list[str]):
        return self.vocab.convert_tokens_to_ids(tokens)
    
    def encode(self, text: str):
        """Encodes a text string into a list of token IDs."""
        tokens = self.text_to_tokens(text)
        return self.tokens_to_ids(tokens)

    def get_pad_id(self) -> int:
        return self.vocab.pad_id
