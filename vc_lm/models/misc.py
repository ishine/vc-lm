import torch
from torch import nn

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
