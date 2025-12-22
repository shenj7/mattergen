"""
Noise-conditioned bulk modulus regressor.

This module mirrors the denoiser’s GemNet backbone and timestep encoding so we
can predict bulk modulus directly from a noisy diffusion state (x_t, t). The
output is a Gaussian parameterized by (mu, logvar) to enable simple Gaussian
NLL training. The design stays intentionally small and heavily commented for
clarity and extensibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from mattergen.common.gemnet.gemnet import GemNetT
from mattergen.common.gemnet.layers.embedding_block import AtomEmbedding
from mattergen.diffusion.model_utils import NoiseLevelEncoding


@dataclass
class BulkModulusClassifierConfig:
    """Simple container describing how the classifier was initialized."""

    hidden_dim: int = 512
    mlp_hidden_dim: int = 256
    logvar_bounds: Sequence[float] = (-10.0, 5.0)
    gemnet_kwargs: dict | None = None


class BulkModulusTimeClassifier(nn.Module):
    """
    Time-conditioned bulk modulus regressor.

    Args:
        gemnet: Optional GemNet backbone. If not supplied, a default GemNetT
            mirroring the denoiser setup is created.
        hidden_dim: Dimensionality of the time embedding and GemNet latent.
        mlp_hidden_dim: Width of the MLP prediction head.
        logvar_bounds: Clamp range for log-variance to keep training stable.
    """

    def __init__(
        self,
        gemnet: GemNetT | None = None,
        *,
        hidden_dim: int = 512,
        mlp_hidden_dim: int = 256,
        logvar_bounds: Sequence[float] = (-10.0, 5.0),
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.logvar_bounds = (float(logvar_bounds[0]), float(logvar_bounds[1]))
        self.noise_level_encoding = NoiseLevelEncoding(hidden_dim)

        # Create a GemNet backbone that mirrors the denoiser defaults so the
        # classifier consumes the exact same representation.
        self.gemnet = gemnet or self._build_default_gemnet(hidden_dim=hidden_dim)

        # Pool graph/node embeddings to a crystal representation and predict
        # Gaussian parameters.
        head_in = hidden_dim + hidden_dim  # pooled graph + explicit time embedding
        self.head = nn.Sequential(
            nn.Linear(head_in, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, 2),
        )

    @staticmethod
    def _build_default_gemnet(hidden_dim: int) -> GemNetT:
        # Matches the denoiser defaults: GemNetT with on-the-fly graphs and
        # stress prediction to keep lattice signals consistent.
        atom_embedding = AtomEmbedding(emb_size=hidden_dim, with_mask_type=True)
        return GemNetT(
            num_targets=1,
            latent_dim=hidden_dim,
            atom_embedding=atom_embedding,
            emb_size_atom=hidden_dim,
            emb_size_edge=hidden_dim,
            regress_stress=True,
            otf_graph=True,
            cutoff=7.0,
            max_neighbors=50,
            max_cell_images_per_dim=5,
        )

    @property
    def init_config(self) -> BulkModulusClassifierConfig:
        """Expose init args so checkpoints can recreate the model."""
        return BulkModulusClassifierConfig(
            hidden_dim=self.hidden_dim,
            mlp_hidden_dim=self.head[0].out_features,
            logvar_bounds=self.logvar_bounds,
            gemnet_kwargs=None,
        )

    def forward(self, x, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict bulk modulus parameters from a noisy diffusion state.

        Args:
            x: ChemGraph batch at timestep t (same fields as the denoiser).
            t: Tensor of shape [batch_size] with diffusion times.

        Returns:
            mu: Mean prediction, shape [batch_size]
            logvar: Log-variance prediction, shape [batch_size]
        """
        # Time embedding (same sinusoidal encoding as the denoiser).
        t_emb = self.noise_level_encoding(t).to(x["cell"].device)

        # GemNet takes the time embedding as a per-crystal latent "z".
        gemnet_out = self.gemnet(
            z=t_emb,
            frac_coords=x["pos"],
            atom_types=x["atomic_numbers"],
            num_atoms=x["num_atoms"],
            batch=x.get_batch_idx("pos"),
            lattice=x["cell"],
            edge_index=None,
            to_jimages=None,
            num_bonds=None,
        )

        # Node embeddings -> pooled crystal embedding.
        node_embeddings = gemnet_out.node_embeddings
        batch_idx = x.get_batch_idx("pos")
        pooled = scatter_mean(node_embeddings, batch_idx, dim=0)

        # Concatenate pooled representation with explicit time embedding.
        head_in = torch.cat([pooled, t_emb], dim=-1)
        mu_logvar = self.head(head_in)
        mu, logvar = mu_logvar.split(1, dim=-1)
        logvar = logvar.clamp(min=self.logvar_bounds[0], max=self.logvar_bounds[1])

        return mu.squeeze(-1), logvar.squeeze(-1)
