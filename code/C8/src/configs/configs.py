from dataclasses import dataclass, field
import torch

@dataclass
class NerConfig:
    # --- 路径参数 ---
    data_dir: str = "data"
    train_file: str = "CMeEE-V2_train.json"
    dev_file: str = "CMeEE-V2_dev.json"
    vocab_file: str = "vocabulary.json"
    tags_file: str = "categories.json"
    output_dir: str = "output"

    # --- 训练参数 ---
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 模型参数 ---
    hidden_size: int = 256
    num_gru_layers: int = 2

# 实例化配置
config = NerConfig()
