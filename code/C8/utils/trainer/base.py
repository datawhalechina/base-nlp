import torch
import torch.nn as nn
import os
from tqdm import tqdm
from loguru import logger

class Trainer:
    def __init__(self, model, train_loader, dev_loader, optimizer, criterion, device, epochs, output_dir, early_stop_epoch):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.output_dir = output_dir
        self.early_stop_epoch = early_stop_epoch
        
        self.best_metric_value = -1
        self.early_stop_counter = 0

        self.model.to(self.device)
        logger.info(f"Trainer initialized on device: {self.device}")

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}", leave=True)
        for batch in progress_bar:
            token_ids, tag_ids, attention_mask = [t.to(self.device) for t in batch]
            
            logits = self.model(token_ids, attention_mask)
            loss = self.criterion(logits.permute(0, 2, 1), tag_ids)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        logger.info(f"Epoch {epoch} Training Average Loss: {avg_loss:.4f}")
    
    def _evaluate(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.dev_loader, desc=f"Evaluating Epoch {epoch}", leave=True):
                token_ids, tag_ids, attention_mask = [t.to(self.device) for t in batch]
                
                logits = self.model(token_ids, attention_mask)
                loss = self.criterion(logits.permute(0, 2, 1), tag_ids)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.dev_loader)
        logger.info(f"Epoch {epoch} Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def _save_checkpoint(self, is_best=False):
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存时不再需要保存整个 config，如果需要恢复，则由 train.py 重新构建
        state = {
            'model_state_dict': self.model.state_dict(),
        }
        
        if is_best:
            save_path = os.path.join(self.output_dir, "best_model.pth")
            torch.save(state, save_path)
            logger.info(f"Best model checkpoint saved to {save_path}")
        
        last_save_path = os.path.join(self.output_dir, "last_model.pth")
        torch.save(state, last_save_path)

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self._train_one_epoch(epoch)
            
            current_metric = self._evaluate(epoch)
            
            if self.best_metric_value == -1 or current_metric < self.best_metric_value:
                logger.info(f"New best model found! Metric improved from {self.best_metric_value:.4f} to {current_metric:.4f}")
                self.best_metric_value = current_metric
                self.early_stop_counter = 0
                self._save_checkpoint(is_best=True)
            else:
                self.early_stop_counter += 1
                logger.info(f"No improvement for {self.early_stop_counter} epoch(s).")
                if self.early_stop_counter >= self.early_stop_epoch:
                    logger.info(f"Early stopping triggered after {self.early_stop_epoch} epochs.")
                    break
            
            self._save_checkpoint(is_best=False)

        logger.info("Training completed.")