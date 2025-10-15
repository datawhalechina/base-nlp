import torch
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, dev_loader=None, 
                 eval_metric_fn=None, output_dir=None, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.eval_metric_fn = eval_metric_fn
        self.output_dir = output_dir
        self.device = device
        print(f"Trainer will run on device: {self.device}")
        
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    def fit(self, epochs):
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 默认最大化主要评估指标（如 F1 分数）
        best_metric = float('-inf')
        
        for epoch in range(1, epochs + 1):
            print(f"--- Epoch {epoch}/{epochs} ---")
            
            train_losses = self._train_one_epoch()

            # 根据返回值的类型来格式化训练损失日志
            if isinstance(train_losses, tuple):
                train_loss_str = f"Train Total Loss: {train_losses[0]:.4f}, NER Loss: {train_losses[1]:.4f}, Non-NER Loss: {train_losses[2]:.4f}"
            else:
                train_loss_str = f"Train Total Loss: {train_losses:.4f}"
            print(train_loss_str)

            eval_metrics = self._evaluate()
            
            # 将评估指标格式化为字符串打印
            eval_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in eval_metrics.items()])
            print(f"Validation Metrics: {eval_metrics_str}")
            
            # 判断是否需要保存模型
            is_best = False
            # 保存最佳模型优先依据 F1 分数
            if 'f1' in eval_metrics:
                if eval_metrics['f1'] > best_metric:
                    best_metric = eval_metrics['f1']
                    is_best = True
            # 若无 F1 指标，则回退为最小化验证集损失
            else:
                # 首次进入该分支时，best_metric 为 -inf，需要重置为 +inf 以便进行最小化比较
                if best_metric == float('-inf'):
                    best_metric = float('inf')
                
                if eval_metrics['loss'] < best_metric:
                    best_metric = eval_metrics['loss']
                    is_best = True

            if is_best:
                print(f"New best model found! Saving to {self.output_dir}")
                torch.save({'model_state_dict': self.model.state_dict()}, os.path.join(self.output_dir, "best_model.pth"))

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
        total_loss = 0
        all_logits = []
        all_labels = []
        all_attention_mask = []

        with torch.no_grad():
            for batch in tqdm(self.dev_loader, desc="Evaluating"):
                outputs = self._evaluation_step(batch)

                loss = outputs['loss']
                # 若损失为元组，取第一个元素作为主损进行统计
                main_loss = loss[0] if isinstance(loss, tuple) else loss
                total_loss += main_loss.item()

                all_logits.append(outputs['logits'].cpu())
                all_labels.append(batch['label_ids'].cpu())
                all_attention_mask.append(batch['attention_mask'].cpu())
        
        metrics = {}
        if self.eval_metric_fn:
            metrics = self.eval_metric_fn(all_logits, all_labels, all_attention_mask)
        
        metrics['loss'] = total_loss / len(self.dev_loader)
        return metrics

    def _evaluation_step(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        logits = self.model(token_ids=batch['token_ids'], attention_mask=batch['attention_mask'])
        loss = self.loss_fn(logits.permute(0, 2, 1), batch['label_ids'])
        return {'loss': loss, 'logits': logits}

    def _save_checkpoint(self, is_best):
        state = {'model_state_dict': self.model.state_dict()}
        if is_best:
            torch.save(state, os.path.join(self.output_dir, 'best_model.pth'))
        torch.save(state, os.path.join(self.output_dir, 'last_model.pth'))
