import os
import torch
import torch.nn as nn
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

from data.data_loader import create_ner_dataloader
from tokenizer.vocabulary import Vocabulary
from tokenizer.char_tokenizer import CharTokenizer
from models.ner_model import BiGRUNerNetWork
from trainer.trainer import Trainer
from utils.file_io import load_json
from metrics.entity_metrics import calculate_entity_level_metrics


def run_ner_task(config):
    # --- 1. 加载词汇表和标签映射, 并创建分词器 ---
    vocab_path = os.path.join(config.data_dir, config.vocab_file)
    tags_path = os.path.join(config.data_dir, config.tags_file)
    
    vocab = Vocabulary.load_from_file(vocab_path)
    tokenizer = CharTokenizer(vocab)
    tag_map = load_json(tags_path)
    id2tag = {v: k for k, v in tag_map.items()}

    # --- 2. 创建数据加载器 ---
    train_loader = create_ner_dataloader(
        data_path=os.path.join(config.data_dir, config.train_file),
        tokenizer=tokenizer,
        tag_map=tag_map,
        batch_size=config.batch_size,
        shuffle=True,
        device=config.device
    )
    dev_loader = create_ner_dataloader(
        data_path=os.path.join(config.data_dir, config.dev_file),
        tokenizer=tokenizer,
        tag_map=tag_map,
        batch_size=config.batch_size,
        shuffle=False,
        device=config.device
    )

    # --- 3. 初始化模型、优化器、损失函数 ---
    model = BiGRUNerNetWork(
        vocab_size=len(vocab),
        hidden_size=config.hidden_size,
        num_tags=len(tag_map),
        num_gru_layers=config.num_gru_layers
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # --- 4. 定义评估函数 ---
    def eval_metric_fn(all_logits, all_labels, all_attention_mask):
        all_preds_ids = [torch.argmax(logits, dim=-1) for logits in all_logits]
        
        # 将列表中的 tensors 移动到 cpu
        all_labels_cpu = [labels.cpu() for labels in all_labels]
        all_preds_ids_cpu = [preds.cpu() for preds in all_preds_ids]
        all_attention_mask_cpu = [mask.cpu() for mask in all_attention_mask]
        
        # 展平 attention_mask，只保留有效部分
        active_masks = [mask.bool() for mask in all_attention_mask_cpu]

        metrics = calculate_entity_level_metrics(
            all_preds_ids_cpu, 
            all_labels_cpu, 
            active_masks, 
            id2tag
        )
        return metrics

    # --- 5. 初始化并启动训练器 ---
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        dev_loader=dev_loader,
        eval_metric_fn=eval_metric_fn,
        output_dir=config.output_dir,
        device=config.device
    )

    trainer.fit(epochs=config.epochs)
