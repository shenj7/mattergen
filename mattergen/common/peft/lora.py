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


def apply_lora(
    module: nn.Module,
    rank: int = 8,
    alpha: float = 8.0,
    target_modules: list[str] | None = None,
    _inside_target: bool = False,
) -> nn.Module:
    """
    Replace nn.Linear with a LoRALayer wrapper, optionally filtering by module name.

    A Linear is wrapped when either:
    - Its own attribute name contains a target string, OR
    - It is nested inside a parent module whose name contains a target string.

    For example, target_modules=["out_forces"] wraps `out_forces.linear` even
    though the leaf attribute is named "linear" (not "out_forces").
    """
    for name, child in list(module.named_children()):
        name_matches = target_modules is None or any(
            t in name or t in child.__class__.__name__ for t in target_modules
        )
        # Wrap this child if it is a Linear and either its name matches or
        # we are already inside a matched ancestor.
        if isinstance(child, nn.Linear) and (name_matches or _inside_target):
            setattr(module, name, LoRALayer(child, rank=rank, alpha=alpha))
        else:
            # Recurse; propagate _inside_target=True when entering a matched parent.
            apply_lora(
                child,
                rank=rank,
                alpha=alpha,
                target_modules=target_modules,
                _inside_target=_inside_target or name_matches,
            )
    return module
