from __future__ import annotations

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """
    LoRA adapter for a frozen linear layer.

    This module wraps an existing nn.Linear, freezes its parameters, and adds
    a low-rank update W + (alpha / r) * B @ A.
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 8.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.lora_A(x)) * self.scaling


def apply_lora(module: nn.Module, rank: int = 8, alpha: float = 8.0) -> nn.Module:
    """
    Replace every nn.Linear with a LoRALayer wrapper.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALayer(child, rank=rank, alpha=alpha))
        else:
            apply_lora(child, rank=rank, alpha=alpha)
    return module
