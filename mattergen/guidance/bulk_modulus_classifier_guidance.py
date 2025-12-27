"""
Compute gradients from the diffusion-time bulk modulus classifier for sampling.

This mirrors the classifier-guidance recipe from Dhariwal & Nichol, but uses a
regressor f_phi(x_t, t) for the bulk modulus instead of a classifier. We take
the gradient of the predicted mean with respect to the noisy state x_t and
feed it back into the reverse dynamics.
"""

from __future__ import annotations
from typing import Mapping

import torch


def _maybe_clip_grad(grad: torch.Tensor | None, max_norm: float | None) -> torch.Tensor | None:
    if grad is None or max_norm is None:
        return grad
    norm = torch.linalg.norm(grad)
    if torch.isfinite(norm) and norm > max_norm:
        grad = grad * (max_norm / (norm + 1e-8))
    return grad


def compute_guidance(
    x_t,
    t: torch.Tensor,
    classifier,
    guidance_scale: float,
    grad_clip: float | None = None,
) -> Mapping[str, torch.Tensor]:
    """
    Compute gradients of mu_phi(x_t, t) with respect to continuous state.

    Args:
        x_t: Noisy ChemGraph batch at timestep t.
        t: Diffusion timestep tensor of shape [batch_size].
        classifier: BulkModulusTimeClassifier (or compatible module).
        guidance_scale: Scalar multiplier for the gradient.
        grad_clip: Optional max norm for the raw gradient to avoid explosions.

    Returns:
        Dict mapping field names (pos, cell, ...) to gradients on those tensors.
    """
    # Enable grads locally even though sampling runs under torch.no_grad().
    with torch.enable_grad():
        # Clone tensors we want gradients for so we do not poison the sampler
        # state with required_grad flags.
        xt_for_grad = x_t.replace(
            pos=x_t["pos"].detach().requires_grad_(True),
            cell=x_t["cell"].detach().requires_grad_(True),
        )

        classifier.zero_grad(set_to_none=True)
        mu, _ = classifier(xt_for_grad, t)
        # Sum over batch to get a scalar objective for autograd.
        mu.sum().backward()

        grad_pos = _maybe_clip_grad(xt_for_grad["pos"].grad, grad_clip)
        grad_cell = _maybe_clip_grad(xt_for_grad["cell"].grad, grad_clip)
        classifier.zero_grad(set_to_none=True)

        guidance: dict[str, torch.Tensor] = {}
        if grad_pos is not None:
            guidance["pos"] = guidance_scale * grad_pos.detach()
        if grad_cell is not None:
            guidance["cell"] = guidance_scale * grad_cell.detach()
        return guidance
