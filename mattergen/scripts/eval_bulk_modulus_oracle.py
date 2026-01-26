"""
Evaluate the calc_bulk_modulus.py estimator on clean (denoised) materials.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ase import Atoms
from mattersim.forcefield import MatterSimCalculator
from tqdm import tqdm

from mattergen.calc_bulk_modulus import calc_bulk_modulus_value
from mattergen.common.data.transform import symmetrize_lattice
from mattergen.scripts.train_bulk_modulus_classifier import load_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the bulk modulus estimator from calc_bulk_modulus.py."
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
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Optional cap on number of samples to evaluate (0 = no cap).",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=5,
        help="Number of volume samples to fit E(V).",
    )
    parser.add_argument(
        "--strain",
        type=float,
        default=0.03,
        help="Volume strain range for E(V) sampling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="MatterSim device: cpu|cuda|auto.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional MatterSim checkpoint path.",
    )

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


def _select_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_dataset(args: argparse.Namespace):
    transforms = [symmetrize_lattice]
    train_dataset, val_dataset, tmpdir = load_datasets(args=args, transforms=transforms)
    dataset = train_dataset if args.split == "train" else val_dataset
    return dataset, tmpdir


def _chemgraph_to_atoms(sample) -> Atoms:
    frac_coords = sample["pos"].detach().cpu().numpy()
    cell = sample["cell"].detach().cpu().numpy()[0]
    atomic_numbers = sample["atomic_numbers"].detach().cpu().numpy()
    return Atoms(
        numbers=atomic_numbers,
        cell=cell,
        scaled_positions=frac_coords,
        pbc=True,
    )


def _eval_predictions(preds: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    diff = preds - targets
    mse = diff.pow(2).mean()
    mae = diff.abs().mean()
    rmse = torch.sqrt(mse)

    # Homoscedastic Gaussian NLL with variance fit from residuals.
    var = diff.pow(2).mean()
    logvar = torch.log(var + 1e-8)
    nll = 0.5 * (diff.pow(2) * torch.exp(-logvar) + logvar).mean()

    target_mean = targets.mean()
    sst = (targets - target_mean).pow(2).sum()
    sse = diff.pow(2).sum()
    r2 = 1.0 - (sse / sst) if sst > 0 else torch.tensor(float("nan"))

    return {
        "mse": mse.item(),
        "rmse": rmse.item(),
        "mae": mae.item(),
        "nll": nll.item(),
        "r2": r2.item(),
    }


def main() -> None:
    args = parse_args()
    dataset, _tmpdir = _load_dataset(args)

    device = _select_device(args.device)
    calc_kwargs: dict[str, str] = {"device": device}
    if args.checkpoint:
        calc_kwargs["load_path"] = args.checkpoint
    calculator = MatterSimCalculator(**calc_kwargs)

    preds: list[float] = []
    targets: list[float] = []
    limit = args.max_samples if args.max_samples > 0 else len(dataset)

    for idx in tqdm(range(min(limit, len(dataset))), desc="eval"):
        sample = dataset[idx]
        target = sample[args.property_name].detach().view(-1)
        if not torch.isfinite(target).all():
            continue

        atoms = _chemgraph_to_atoms(sample)
        atoms.calc = calculator
        pred = calc_bulk_modulus_value(
            atoms, n_points=args.n_points, strain=args.strain
        )
        preds.append(float(pred))
        targets.append(float(target.item()))

    if not targets:
        raise RuntimeError("No finite targets found to evaluate.")

    pred_tensor = torch.tensor(preds, dtype=torch.float32)
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    metrics = _eval_predictions(pred_tensor, target_tensor)

    print(f"Evaluated {len(targets)} samples with calc_bulk_modulus_value")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
