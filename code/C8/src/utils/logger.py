from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    """
    日志记录器。
    """
    def __init__(self, log_dir):
        """
        Args:
            log_dir (str): TensorBoard 日志文件的保存目录。如果为 None，则不进行日志记录。
        """
        self.writer = SummaryWriter(log_dir) if log_dir else None

    def log_metrics(self, metrics, step, prefix):
        """
        将指标字典记录到 TensorBoard。

        Args:
            metrics (dict or tuple): 包含指标名称和值的字典，或包含损失值的元组。
            step (int): 当前的全局步骤，通常是 epoch 数。
            prefix (str): 指标名称的前缀 (e.g., 'Train', 'Validation')。
        """
        if self.writer is None:
            return
        
        if isinstance(metrics, tuple):  # 训练损失元组
            self.writer.add_scalar(f"{prefix}/Total_Loss", metrics[0], step)
            if len(metrics) > 1:
                self.writer.add_scalar(f"{prefix}/NER_Loss", metrics[1], step)
                self.writer.add_scalar(f"{prefix}/Non-NER_Loss", metrics[2], step)
        elif isinstance(metrics, dict):  # 评估指标字典
            for k, v in metrics.items():
                self.writer.add_scalar(f"{prefix}/{k.capitalize()}", v, step)
    
    def close(self):
        """
        关闭 SummaryWriter。
        """
        if self.writer:
            self.writer.close()

