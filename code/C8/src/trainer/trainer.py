import torch
from tqdm import tqdm
import os

from src.utils.early_stop import EarlyStopping
from src.utils.logger import TensorBoardLogger


class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, dev_loader=None, 
                 eval_metric_fn=None, output_dir=None, device='cpu', 
                 summary_writer_dir=None, early_stopping_patience=5,
                 resume_checkpoint=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.eval_metric_fn = eval_metric_fn
        self.output_dir = output_dir
        self.device = device
        
        self.start_epoch = 1
        self.best_metric = float('-inf')

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            
        self.logger = TensorBoardLogger(summary_writer_dir)
            
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            verbose=True,
            monitor='f1'
        )
            
        if resume_checkpoint:
            self._resume_checkpoint(resume_checkpoint)

        print(f"Trainer will run on device: {self.device}")

    def _resume_checkpoint(self, path):
        """恢复训练状态"""
        print(f"Loading checkpoint from: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint['best_metric']
        print(f"Resumed from checkpoint. Starting at epoch {self.start_epoch}.")

    def fit(self, epochs):
        for epoch in range(self.start_epoch, epochs + 1):
            print(f"--- Epoch {epoch}/{epochs} ---")
            
            # --- 训练 ---
            train_losses = self._train_one_epoch()
            self._on_epoch_end_log(train_losses, epoch, "Train")

            # --- 评估 ---
            eval_metrics = self._evaluate()
            self._on_epoch_end_log(eval_metrics, epoch, "Validation")

            # --- 保存与提前停止 ---
            is_best = False
            current_metric = eval_metrics.get('f1', -eval_metrics.get('loss', float('inf')))
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                is_best = True
                print(f"New best model found! Saving to {self.output_dir}")
                self._save_checkpoint(epoch, is_best=True)
            
            self._save_checkpoint(epoch, is_best=False) # 保存 latest
            
            if self.early_stopping(current_metric):
                print("Early stopping triggered.")
                break
        
        self.logger.close()

    def _on_epoch_end_log(self, metrics, epoch, prefix):
        """在 epoch 结束时打印并记录日志"""
        # 打印到控制台
        if prefix == 'Train':
            log_str = self._format_loss_log(metrics)
        else: # Validation
            log_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"{prefix} Metrics: {log_str}")
        
        # 记录到 TensorBoard
        self.logger.log_metrics(metrics, epoch, prefix)

    def _format_loss_log(self, losses):
        if isinstance(losses, tuple):
            return f"Total Loss: {losses[0]:.4f}, NER Loss: {losses[1]:.4f}, Non-NER Loss: {losses[2]:.4f}"
        else:
            return f"Total Loss: {losses:.4f}"

    def _train_one_epoch(self):
        self.model.train()
        total_loss_sum = 0
        total_ner_loss = 0
        total_non_ner_loss = 0
        custom_loss_used = False

        for batch in tqdm(self.train_loader, desc=f"Training Epoch"):
            outputs = self._train_step(batch)
            loss = outputs['loss']

            if isinstance(loss, tuple):
                custom_loss_used = True
                total_loss_sum += loss[0].item()
                total_ner_loss += loss[1].item()
                total_non_ner_loss += loss[2].item()
            else:
                total_loss_sum += loss.item()

        if custom_loss_used:
            avg_loss = total_loss_sum / len(self.train_loader)
            avg_ner_loss = total_ner_loss / len(self.train_loader)
            avg_non_ner_loss = total_non_ner_loss / len(self.train_loader)
            return avg_loss, avg_ner_loss, avg_non_ner_loss
        else:
            return total_loss_sum / len(self.train_loader)

    def _train_step(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        logits = self.model(token_ids=batch['token_ids'], attention_mask=batch['attention_mask'])
        loss = self.loss_fn(logits.permute(0, 2, 1), batch['label_ids'])
        
        # 如果损失是一个元组，只使用第一个元素进行反向传播
        main_loss = loss[0] if isinstance(loss, tuple) else loss

        self.optimizer.zero_grad()
        main_loss.backward()
        self.optimizer.step()
        
        return {'loss': loss, 'logits': logits}

    def _evaluate(self):
        if self.dev_loader is None:
            return None

        self.model.eval()
        total_loss_sum, total_ner_loss, total_non_ner_loss = 0, 0, 0
        custom_loss_used = False
        all_logits, all_labels, all_attention_mask = [], [], []

        with torch.no_grad():
            for batch in tqdm(self.dev_loader, desc="Evaluating"):
                outputs = self._evaluation_step(batch)
                loss = outputs['loss']

                if isinstance(loss, tuple):
                    custom_loss_used = True
                    total_loss_sum += loss[0].item()
                    total_ner_loss += loss[1].item()
                    total_non_ner_loss += loss[2].item()
                else:
                    total_loss_sum += loss.item()
                
                all_logits.append(outputs['logits'].cpu())
                all_labels.append(batch['label_ids'].cpu())
                all_attention_mask.append(batch['attention_mask'].cpu())
        
        metrics = {}
        if self.eval_metric_fn:
            metrics = self.eval_metric_fn(all_logits, all_labels, all_attention_mask)
        
        # 将各部分损失也加入到 metrics 中
        metrics['loss'] = total_loss_sum / len(self.dev_loader)
        if custom_loss_used:
            metrics['ner_loss'] = total_ner_loss / len(self.dev_loader)
            metrics['non_ner_loss'] = total_non_ner_loss / len(self.dev_loader)
            
        return metrics

    def _evaluation_step(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        logits = self.model(token_ids=batch['token_ids'], attention_mask=batch['attention_mask'])
        loss = self.loss_fn(logits.permute(0, 2, 1), batch['label_ids'])
        return {'loss': loss, 'logits': logits}

    def _save_checkpoint(self, epoch, is_best):
        """保存检查点"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric
        }
        if is_best:
            torch.save(state, os.path.join(self.output_dir, 'best_model.pth'))
        
        torch.save(state, os.path.join(self.output_dir, 'last_model.pth'))
