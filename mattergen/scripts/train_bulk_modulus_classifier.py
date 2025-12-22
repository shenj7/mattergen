"""
Train a diffusion-time bulk modulus predictor using cached MatterGen datasets.

The model takes a noisy diffusion state (x_t, t) and predicts the bulk modulus
of the clean structure x_0. It shares the GemNet backbone and timestep
encoding with the denoiser so it can be dropped into sampling-time guidance.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.dataset import CrystalDataset
from mattergen.common.data.transform import symmetrize_lattice
from mattergen.common.diffusion.corruption import (
    LatticeVPSDE,
    NumAtomsVarianceAdjustedWrappedVESDE,
)
from mattergen.diffusion.corruption.d3pm_corruption import D3PMCorruption
from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.d3pm.d3pm import MaskDiffusion, create_discrete_diffusion_schedule
from mattergen.diffusion.timestep_samplers import UniformTimestepSampler
from mattergen.property_predictors import BulkModulusTimeClassifier


def build_default_corruption() -> MultiCorruption:
    """
    Mirror the default MatterGen corruption: wrapped VE for pos, lattice VPSDE for cell,
    and mask-based D3PM for atom types.
    """
    pos_sde = NumAtomsVarianceAdjustedWrappedVESDE(
        wrapping_boundary=1.0, sigma_min=0.01, sigma_max=5.0, limit_info_key="num_atoms"
    )
    cell_sde = LatticeVPSDE(
        beta_min=0.1,
        beta_max=20,
        limit_density=0.05,
        limit_var_scaling_constant=0.25,
    )
    atom_corruption = D3PMCorruption(
        offset=1,
        d3pm=MaskDiffusion(
            dim=101,
            schedule=create_discrete_diffusion_schedule(kind="standard", num_steps=1000),
        ),
    )
    return MultiCorruption(
        sdes={"pos": pos_sde, "cell": cell_sde},
        discrete_corruptions={"atomic_numbers": atom_corruption},
    )


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inv_var = torch.exp(-logvar)
    return 0.5 * ((mu - target) ** 2 * inv_var + logvar)


def run_epoch(
    *,
    model: BulkModulusTimeClassifier,
    loader: GeoDataLoader,
    corruption: MultiCorruption,
    timestep_sampler: UniformTimestepSampler,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    use_mse: bool,
    property_name: str,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(loader, desc="train" if is_train else "val"):
        batch = batch.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        targets = batch[property_name].detach().view(-1)
        t = timestep_sampler(batch_size=batch.get_batch_size(), device=device)
        noisy_batch = corruption.sample_marginal(batch, t)
        mu, logvar = model(noisy_batch, t)
        loss = F.mse_loss(mu, targets) if use_mse else gaussian_nll(mu, logvar, targets).mean()

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.detach().item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def save_checkpoint(
    out_dir: Path,
    name: str,
    model: BulkModulusTimeClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
        },
        out_dir / f"{name}.pt",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bulk modulus diffusion-time classifier")
    parser.add_argument(
        "--train_cache_dir",
        type=str,
        required=True,
        help="Path to cached train split (e.g., datasets/cache/alex_mp_20/train)",
    )
    parser.add_argument(
        "--val_cache_dir",
        type=str,
        required=True,
        help="Path to cached val split (e.g., datasets/cache/alex_mp_20/val)",
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to store checkpoints")
    parser.add_argument(
        "--property_name",
        type=str,
        default="ml_bulk_modulus",
        help="Dataset property key to supervise on (e.g., ml_bulk_modulus or dft_bulk_modulus)",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--logvar_min", type=float, default=-10.0)
    parser.add_argument("--logvar_max", type=float, default=5.0)
    parser.add_argument("--use_mse", action="store_true", help="Fallback to MSE loss")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = [symmetrize_lattice]
    train_dataset = CrystalDataset.from_cache_path(
        cache_path=args.train_cache_dir, properties=[args.property_name], transforms=transforms
    )
    val_dataset = CrystalDataset.from_cache_path(
        cache_path=args.val_cache_dir, properties=[args.property_name], transforms=transforms
    )

    train_loader = GeoDataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = GeoDataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    corruption = build_default_corruption()
    timestep_sampler = UniformTimestepSampler(min_t=1e-5, max_t=corruption.T)

    model = BulkModulusTimeClassifier(
        hidden_dim=args.hidden_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
        logvar_bounds=(args.logvar_min, args.logvar_max),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_val = float("inf")
    cfg = {
        "model_kwargs": asdict(model.init_config),
        "args": vars(args),
    }
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            corruption=corruption,
            timestep_sampler=timestep_sampler,
            optimizer=optimizer,
            device=device,
            use_mse=args.use_mse,
            property_name=args.property_name,
        )
        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            corruption=corruption,
            timestep_sampler=timestep_sampler,
            optimizer=None,
            device=device,
            use_mse=args.use_mse,
            property_name=args.property_name,
        )
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        save_checkpoint(
            Path(args.out_dir),
            name="last",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            cfg=cfg,
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                Path(args.out_dir),
                name="best",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                cfg=cfg,
            )


if __name__ == "__main__":
    main()
