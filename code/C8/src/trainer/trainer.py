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
        self.device = torch.device(device)
        print(f"Trainer will run on device: {self.device}")
        
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    def fit(self, epochs):
        best_metric = float('inf')
        
        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch()
            print(f"Epoch {epoch} - Training Loss: {train_loss:.4f}")

            metrics = self._evaluate()
            if metrics:
                print(f"Epoch {epoch} - Validation Metrics: {metrics}")
                current_metric = metrics.get('loss') # 默认监控loss
                
                if current_metric < best_metric:
                    best_metric = current_metric
                    if self.output_dir:
                        self._save_checkpoint(is_best=True)
                        print(f"New best model saved with validation loss: {best_metric:.4f}")

            if self.output_dir:
                self._save_checkpoint(is_best=False)

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc=f"Training Epoch"):
            outputs = self._train_step(batch)
            total_loss += outputs['loss'].item()
        return total_loss / len(self.train_loader)

    def _train_step(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        logits = self.model(token_ids=batch['token_ids'], attention_mask=batch['attention_mask'])
        loss = self.loss_fn(logits.permute(0, 2, 1), batch['label_ids'])
        
        self.optimizer.zero_grad()
        loss.backward()
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
                total_loss += outputs['loss'].item()
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
