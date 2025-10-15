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
    
    # --- 损失函数参数 ---
    loss_type: str = "cross_entropy"  # 可选: "cross_entropy", "weighted_ce", "hard_negative_mining"
    entity_loss_weight: float = 10.0 # 在 weighted_ce 和 hard_negative_mining 中, 给实体部分损失的权重
    hard_negative_ratio: float = 0.5 # 在 hard_negative_mining 中, 负样本数量与正样本数量的比例

    # --- 模型参数 ---
    hidden_size: int = 256
    num_gru_layers: int = 2

# 实例化配置
config = NerConfig()
