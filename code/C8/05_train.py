import torch
import json

from _04_model import BiGRUForNer
from _03_data_loader import Vocabulary, create_ner_dataloader
from utils import Trainer, TrainerConfig

def main():
    # 1. 初始化配置
    config = TrainerConfig()
    
    # 2. 加载通用资源 (词汇表, 标签映射)
    vocab = Vocabulary(config.vocab_file)
    with open(config.tags_file, 'r', encoding='utf-8') as f:
        tag_map = json.load(f)

    # 3. 初始化数据加载器
    train_loader = create_ner_dataloader(
        config.train_file, vocab, tag_map, config.batch_size, shuffle=True
    )
    dev_loader = create_ner_dataloader(
        config.dev_file, vocab, tag_map, config.batch_size, shuffle=False
    )

    # 4. 初始化模型
    model = BiGRUForNer(
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_tags=len(tag_map),
        num_gru_layers=config.gru_num_layers
    )

    # 5. 初始化优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=config.label_ignore_index)
    
    # 6. 初始化 Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device(config.device),
        epochs=config.epochs,
        output_dir=config.output_dir,
        early_stop_epoch=config.early_stop_epoch
    )
    
    # 7. 开始训练
    trainer.train()

if __name__ == "__main__":
    main()