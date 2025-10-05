from dataclasses import dataclass
import torch

@dataclass
class TrainerConfig:
    # --- 路径参数 ---
    train_file: str = './data/CMeEE-V2_train.json'
    dev_file: str = './data/CMeEE-V2_dev.json'
    vocab_file: str = './data/vocabulary.json'
    tags_file: str = './data/categories.json'
    output_dir: str = './output_model'

    # --- 训练参数 ---
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    early_stop_epoch: int = 5
    
    # --- 模型参数 ---
    embedding_dim: int = 256
    hidden_size: int = 256
    gru_num_layers: int = 2
    
    # --- 其他 ---
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    label_ignore_index: int = -100