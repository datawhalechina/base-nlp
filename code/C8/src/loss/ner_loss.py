import torch
import torch.nn as nn

class NerLoss(nn.Module):
    """
    自定义 NER 损失函数，集成了两种策略来对抗数据不均衡问题：
    1. 加权交叉熵 (Weighted Cross-Entropy)
    2. 硬负样本挖掘 (Hard Negative Mining)
    """
    def __init__(self, loss_type='cross_entropy', entity_weight=10.0, hard_negative_ratio=0.5, ignore_index=-100):
        super().__init__()
        self.loss_type = loss_type
        self.entity_weight = entity_weight
        self.hard_negative_ratio = hard_negative_ratio
        
        # 使用 'none' 模式，以便对每个 token 的损失进行精细化操作
        self.base_loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, logits, labels):
        """
        根据初始化时选择的 loss_type 计算损失。
        
        Args:
            logits (torch.Tensor): 模型的原始输出, [batch_size, num_tags, seq_len]
            labels (torch.Tensor): 真实标签, [batch_size, seq_len]

        Returns:
            - 如果 loss_type 是 'cross_entropy', 返回单个损失张量。
            - 否则, 返回一个元组: (用于反向传播的总损失, 实体部分损失, 非实体部分损失)。
        """
        if self.loss_type == 'weighted_ce':
            return self._weighted_cross_entropy(logits, labels)
        elif self.loss_type == 'hard_negative_mining':
            return self._hard_negative_mining(logits, labels)
        else: # 默认使用标准的交叉熵损失
            return self.base_loss_fn(logits, labels).mean()

    def _weighted_cross_entropy(self, logits, labels):
        """
        加权交叉熵损失。
        """
        # [batch_size, seq_len]
        loss_per_token = self.base_loss_fn(logits, labels)

        # 区分实体和非实体 token
        entity_mask = (labels > 0).float()
        non_entity_mask = (labels == 0).float()

        # 计算各自部分的损失
        entity_loss = loss_per_token * entity_mask
        non_entity_loss = loss_per_token * non_entity_mask
        
        # 对有效 token 的损失求平均
        ner_loss_mean = torch.sum(entity_loss) / (torch.sum(entity_mask) + 1e-8)
        non_ner_loss_mean = torch.sum(non_entity_loss) / (torch.sum(non_entity_mask) + 1e-8)

        # 加权求和
        total_loss = self.entity_weight * ner_loss_mean + 1.0 * non_ner_loss_mean
        
        return total_loss, ner_loss_mean.detach(), non_ner_loss_mean.detach()

    def _hard_negative_mining(self, logits, labels):
        """
        硬负样本挖掘损失。
        """
        # [batch_size, seq_len]
        loss_per_token = self.base_loss_fn(logits, labels)

        # 实体部分的损失计算不变
        entity_mask = (labels > 0).float()
        entity_loss = loss_per_token * entity_mask
        ner_loss_mean = torch.sum(entity_loss) / (torch.sum(entity_mask) + 1e-8)

        # 非实体部分的损失
        non_entity_mask = (labels == 0).float()
        non_entity_loss = loss_per_token * non_entity_mask
        
        num_hard_negatives = int(torch.sum(entity_mask).item() * self.hard_negative_ratio)
        if num_hard_negatives == 0: # 避免在验证初期实体数量为0时出错
             num_hard_negatives = int(torch.sum(non_entity_mask).item() * 0.1) # 至少选10%的负样本

        # 选出损失最大的 k 个非实体 token
        non_entity_loss_flat = non_entity_loss.view(-1)
        topk_losses, _ = torch.topk(non_entity_loss_flat, k=num_hard_negatives)
        
        non_ner_loss_mean = torch.mean(topk_losses)

        # 加权求和
        total_loss = self.entity_weight * ner_loss_mean + 1.0 * non_ner_loss_mean
        
        return total_loss, ner_loss_mean.detach(), non_ner_loss_mean.detach()
