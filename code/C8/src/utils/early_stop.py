import numpy as np

class EarlyStopping:
    """
    提前停止策略，用于在验证集指标不再提升时停止训练。
    """
    def __init__(self, patience=5, verbose=False, delta=0, monitor='f1', mode='max'):
        """
        Args:
            patience (int): 在性能不再提升后，等待多少个 epoch 才停止训练。
            verbose (bool): 如果为 True，则打印一条消息，说明为什么提前停止。
            delta (float): 监控指标的最小变化量，小于此值的变化被视为没有提升。
            monitor (str): 要监控的指标名称，例如 'f1' 或 'loss'。
            mode (str): 'max' 表示指标越大越好，'min' 表示指标越小越好。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_best = np.inf if mode == 'min' else -np.inf
        self.delta = delta
        self.monitor = monitor
        self.mode = mode

    def __call__(self, val_metric):
        """
        根据当前验证指标判断是否需要停止。
        """
        score = -val_metric if self.mode == 'min' else val_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop

