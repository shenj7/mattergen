"""
Evaluate a diffusion-time bulk modulus predictor on clean (denoised) materials.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm

from mattergen.common.data.transform import symmetrize_lattice
from mattergen.property_predictors import BulkModulusLoRATimePredictor, BulkModulusTimeClassifier
from mattergen.scripts.train_bulk_modulus_classifier import load_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a bulk modulus predictor at a fixed diffusion time."
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument(
        "--predictor_type",
        type=str,
        choices=["mlp", "lora"],
        default="mlp",
        help="Model type used when training the checkpoint.",
    )
    parser.add_argument(
        "--property_name",
        type=str,
        default="dft_bulk_modulus",
        help="Dataset property key to evaluate against.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default="val",
        help="Which dataset split to evaluate.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--t_eval",
        type=float,
        default=0.0,
        help="Diffusion time used for evaluation (0.0 means fully denoised).",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="Optional cap on number of batches to evaluate (0 = no cap).",
    )
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument(
        "--train_cache_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data-release" / "alex-mp" / "train"),
    )
    parser.add_argument(
        "--val_cache_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data-release" / "alex-mp" / "val"),
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data-release" / "alex-mp" / "alex_mp_20.zip"),
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data-release" / "mp-20" / "mp_20" / "train.csv"),
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data-release" / "mp-20" / "mp_20" / "val.csv"),
    )
    parser.add_argument(
        "--csv_cache_root",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data-release" / "mp-20" / "cache"),
    )
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _select_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(args: argparse.Namespace, device: torch.device):
    ckpt_path = Path(args.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if args.predictor_type == "lora":
        return BulkModulusLoRATimePredictor.from_checkpoint(str(ckpt_path), device=device)
    return BulkModulusTimeClassifier.from_checkpoint(str(ckpt_path), device=device)


def _load_dataset(args: argparse.Namespace):
    transforms = [symmetrize_lattice]
    train_dataset, val_dataset, tmpdir = load_datasets(args=args, transforms=transforms)
    dataset = train_dataset if args.split == "train" else val_dataset
    return dataset, tmpdir


def _eval_predictions(
    mu: torch.Tensor, logvar: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    diff = mu - targets
    mse = diff.pow(2).mean()
    mae = diff.abs().mean()
    rmse = torch.sqrt(mse)
    nll = 0.5 * (diff.pow(2) * torch.exp(-logvar) + logvar).mean()

    target_mean = targets.mean()
    sst = (targets - target_mean).pow(2).sum()
    sse = diff.pow(2).sum()
    r2 = 1.0 - (sse / sst) if sst > 0 else torch.tensor(float("nan"), device=targets.device)

    return {
        "mse": mse.item(),
        "rmse": rmse.item(),
        "mae": mae.item(),
        "nll": nll.item(),
        "r2": r2.item(),
    }


def main() -> None:
    args = parse_args()
    device = _select_device(args.device)

    dataset, _tmpdir = _load_dataset(args)
    loader = GeoDataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = _load_model(args, device)
    model.eval()

    mus = []
    logvars = []
    targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="eval")):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            batch = batch.to(device)
            t = torch.full(
                (batch.get_batch_size(),), args.t_eval, dtype=torch.float32, device=device
            )
            mu, logvar = model(batch, t)
            batch_targets = batch[args.property_name].detach().view(-1)
            finite_mask = torch.isfinite(batch_targets)
            if not finite_mask.any():
                continue
            mus.append(mu[finite_mask].detach())
            logvars.append(logvar[finite_mask].detach())
            targets.append(batch_targets[finite_mask].detach())

    if not targets:
        raise RuntimeError("No finite targets found to evaluate.")

    mu_all = torch.cat(mus)
    logvar_all = torch.cat(logvars)
    target_all = torch.cat(targets)

    metrics = _eval_predictions(mu_all, logvar_all, target_all)

    print(f"Evaluated {target_all.numel()} samples at t={args.t_eval:.6f}")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
