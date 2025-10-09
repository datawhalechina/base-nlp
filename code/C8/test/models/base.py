import torch.nn as nn
from abc import ABC, abstractmethod

class BaseNerNetwork(nn.Module, ABC):
    @abstractmethod
    def forward(self, token_ids, attention_mask):
        """
        定义所有 NER 模型都必须遵循的前向传播接口。
        
        Args:
            token_ids (torch.Tensor): [batch_size, seq_len]
            attention_mask (torch.Tensor): [batch_size, seq_len]

        Returns:
            torch.Tensor: Logits, [batch_size, seq_len, num_tags]
        """
        raise NotImplementedError
