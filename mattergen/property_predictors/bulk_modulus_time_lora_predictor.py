from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from mattergen.common.gemnet.gemnet import GemNetT
from mattergen.common.gemnet.layers.embedding_block import AtomEmbedding
from mattergen.common.peft.lora import apply_lora
from mattergen.diffusion.model_utils import NoiseLevelEncoding


@dataclass
class BulkModulusLoRAClassifierConfig:
    hidden_dim: int = 512
    logvar_bounds: Sequence[float] = (-10.0, 5.0)
    lora_rank: int = 8
    lora_alpha: float = 8.0
    gemnet_kwargs: dict | None = None


class BulkModulusLoRATimePredictor(nn.Module):
    def __init__(
        self,
        gemnet: GemNetT | None = None,
        *,
        hidden_dim: int = 512,
        logvar_bounds: Sequence[float] = (-10.0, 5.0),
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        gemnet_kwargs: dict | None = None,
        **_: dict,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.logvar_bounds = (float(logvar_bounds[0]), float(logvar_bounds[1]))
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.noise_level_encoding = NoiseLevelEncoding(hidden_dim)

        self.gemnet = gemnet or self._build_default_gemnet(
            hidden_dim=hidden_dim, gemnet_kwargs=gemnet_kwargs or {}
        )
        self._freeze_gemnet_parameters()
        apply_lora(self.gemnet, rank=lora_rank, alpha=lora_alpha)

        head_in = hidden_dim + hidden_dim
        self.head = nn.Linear(head_in, 2)

    @staticmethod
    def _build_default_gemnet(hidden_dim: int, gemnet_kwargs: dict) -> GemNetT:
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
            **gemnet_kwargs,
        )

    def _freeze_gemnet_parameters(self) -> None:
        for param in self.gemnet.parameters():
            param.requires_grad = False

    @property
    def init_config(self) -> BulkModulusLoRAClassifierConfig:
        return BulkModulusLoRAClassifierConfig(
            hidden_dim=self.hidden_dim,
            logvar_bounds=self.logvar_bounds,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            gemnet_kwargs=None,
        )

    def forward(self, x, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t_emb = self.noise_level_encoding(t).to(x["cell"].device)

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

        node_embeddings = gemnet_out.node_embeddings
        batch_idx = x.get_batch_idx("pos")
        pooled = scatter_mean(node_embeddings, batch_idx, dim=0)

        head_in = torch.cat([pooled, t_emb], dim=-1)
        mu_logvar = self.head(head_in)
        mu, logvar = mu_logvar.split(1, dim=-1)
        logvar = logvar.clamp(min=self.logvar_bounds[0], max=self.logvar_bounds[1])

        return mu.squeeze(-1), logvar.squeeze(-1)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, device: torch.device | str = "cpu"
    ) -> "BulkModulusLoRATimePredictor":
        """
        Load a BulkModulusLoRATimePredictor from a checkpoint file.

        This method correctly instantiates the model with its LoRA-adapted
        architecture before loading the weights.
        """
        ckpt = torch.load(checkpoint_path, map_location=device)
        config = ckpt["config"]["model_kwargs"]
        model = cls(**config)
        model.load_state_dict(ckpt["model_state_dict"])
        return model.to(device)
