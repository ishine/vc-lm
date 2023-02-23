import torch
from torch import nn
from torch import optim
import numpy as np

class StageAdaLN(nn.Module):
    def __init__(self, layer_norm: nn.LayerNorm,
                 num_stage: int):
        super().__init__()
        self.layer_norm = layer_norm
        self.num_stage = num_stage
        self.register_parameter('stage_w', nn.Parameter(torch.ones((num_stage,) + self.layer_norm.normalized_shape)))
        self.register_parameter('stage_b', nn.Parameter(torch.zeros((num_stage,) + self.layer_norm.normalized_shape)))

    def forward(self, x: torch.Tensor, stage_id: torch.Tensor):
        """
        Args:
            x: torch.Tensor (batch_size, ..., dim)
            stage_id: torch.Tensor (batch_size,)
        Return:
            y: torch.Tensor (batch_size, ..., dim)
        """
        y = self.layer_norm(x)
        expand_number = y.ndim - len(self.layer_norm.normalized_shape) - 1
        # (batch_size, *dim)
        c_w, c_b = self.stage_w[stage_id], self.stage_b[stage_id]
        for i in range(expand_number):
            c_w = c_w.unsqueeze(1)
            c_b = c_b.unsqueeze(1)
        return y * c_w + c_b

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters,
                 min_lr=1e-5):
        self.warmup = warmup
        self.max_num_iters = max_iters
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [max(base_lr * lr_factor, self.min_lr) for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor