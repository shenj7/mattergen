import argparse
import csv
import os
import random
import sys
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from mattergen.common.data.dataset import CrystalDataset
from mattergen.common.data.transform import symmetrize_lattice
from mattergen.property_predictors import (
    BulkModulusLoRAMLPTimePredictor,
    BulkModulusLoRATimePredictor,
    BulkModulusTimeClassifier,
)
from mattergen.common.diffusion.corruption import (
    LatticeVPSDE,
    NumAtomsVarianceAdjustedWrappedVESDE,
)
from mattergen.diffusion.corruption.d3pm_corruption import D3PMCorruption
from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.d3pm.d3pm import MaskDiffusion, create_discrete_diffusion_schedule

def build_default_corruption() -> MultiCorruption:
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

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate bulk modulus predictor robustness to noise.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/bulk_modulus_classifier_mixed_16_large/best.pt",
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
         "--dataset_path",
         type=str,
         default="datasets/cache/mp_20/test",
         help="Path to the cached dataset."
    )
    parser.add_argument(
        "--property_name",
        type=str,
        default="dft_bulk_modulus",
        help="Property to evaluate.",
    )
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="One or more random seeds to run. Results are averaged across seeds in the summary CSV.",
    )
    parser.add_argument("--predictor_type", type=str, default="lora_mlp", choices=["mlp", "lora", "lora_mlp"])
    parser.add_argument("--output_plot", type=str, default="bulk_modulus_noise_robustness.png")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="bulk_modulus_noise_robustness.csv",
        help="Path to write the per-seed summary CSV.",
    )

    return parser.parse_args()

def load_model(checkpoint_path, device, predictor_type=None):
    if not os.path.exists(checkpoint_path):
        # Try relative to project root
        checkpoint_path = project_root / checkpoint_path
    
    if not os.path.exists(checkpoint_path):
         raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Heuristic to determine type if not provided
    if predictor_type is None:
         # Default to assuming standard Classifier unless user says otherwise
         # or we could try to load state dict and check keys, but let's stick to try-except or just default
         pass

    if predictor_type == "lora":
         return BulkModulusLoRATimePredictor.from_checkpoint(str(checkpoint_path), device=device)
    elif predictor_type == "lora_mlp":
        return BulkModulusLoRAMLPTimePredictor.from_checkpoint(str(checkpoint_path), device=device)
    else:
        return BulkModulusTimeClassifier.from_checkpoint(str(checkpoint_path), device=device)

def run_one_seed(seed, args, dataset, valid_indices, model, corruption, device):
    """Run evaluation for a single seed. Returns list of dicts: {step, mae, rmse}."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    noise_steps = [1, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    total_steps = 1000

    if len(valid_indices) < args.num_samples:
        sample_indices = valid_indices
    else:
        sample_indices = np.random.choice(valid_indices, size=args.num_samples, replace=False)

    subset = dataset.subset(sample_indices)
    loader = GeoDataLoader(subset, batch_size=args.batch_size, shuffle=False)

    all_results = {}  # step -> list of (true, pred)

    with torch.no_grad():
        for step in noise_steps:
            t_val = step / total_steps
            results_step = []
            for batch in loader:
                batch = batch.to(device)
                t = torch.full((batch.num_graphs,), t_val, device=device, dtype=torch.float32)
                noisy_batch = corruption.sample_marginal(batch, t)
                mu, logvar = model(noisy_batch, t)
                targets = batch[args.property_name]
                for p, tr in zip(mu.cpu().numpy(), targets.cpu().numpy()):
                    results_step.append((tr, p))
            all_results[step] = results_step

    metrics = []
    for step in noise_steps:
        data = all_results[step]
        trues = np.array([d[0] for d in data])
        preds = np.array([d[1] for d in data])
        mae = np.mean(np.abs(preds - trues))
        rmse = np.sqrt(np.mean((preds - trues) ** 2))
        metrics.append({"seed": seed, "step": step, "t": step / total_steps, "mae": mae, "rmse": rmse})

    return metrics, all_results


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset once
    print(f"Loading dataset from {args.dataset_path}...")
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        dataset_path = project_root / args.dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}")

    dataset = CrystalDataset.from_cache_path(
        str(dataset_path),
        properties=[args.property_name],
        transforms=[symmetrize_lattice],
    )

    prop_values = dataset.properties[args.property_name]
    valid_indices = np.where(np.isfinite(prop_values) & (prop_values > 0))[0]
    print(f"Valid entries: {len(valid_indices)}")

    # Load model once
    print(f"Loading model from {args.checkpoint_path}...")
    model = load_model(args.checkpoint_path, device, args.predictor_type)
    model.eval()

    corruption = build_default_corruption()

    # Run across all seeds
    all_seed_metrics = []  # flat list of per-seed-per-step dicts
    last_seed_results = None  # used for the plot (last seed)

    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        seed_metrics, seed_results = run_one_seed(
            seed, args, dataset, valid_indices, model, corruption, device
        )
        for m in seed_metrics:
            print(f"  step={m['step']:>4d}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")
        all_seed_metrics.extend(seed_metrics)
        last_seed_results = seed_results

    # Write per-seed CSV
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "step", "t", "mae", "rmse"])
        writer.writeheader()
        writer.writerows(all_seed_metrics)
    print(f"\nWrote per-seed summary to {csv_path}")

    # Print averaged summary across seeds
    noise_steps = sorted(set(m["step"] for m in all_seed_metrics))
    print(f"\n{'Step':<8} {'t':<8} {'MAE mean':<12} {'MAE 2σ':<12} {'RMSE mean':<12} {'RMSE 2σ':<12}")
    for step in noise_steps:
        rows = [m for m in all_seed_metrics if m["step"] == step]
        maes = [r["mae"] for r in rows]
        rmses = [r["rmse"] for r in rows]
        t_val = rows[0]["t"]
        print(
            f"{step:<8} {t_val:<8.3f} {np.mean(maes):<12.4f} {2*np.std(maes):<12.4f} "
            f"{np.mean(rmses):<12.4f} {2*np.std(rmses):<12.4f}"
        )

    # Plot using the last seed's raw results
    noise_steps_list = sorted(last_seed_results.keys())
    total_steps = 1000
    n_plots = len(noise_steps_list)
    ncols = 2
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6 * nrows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for i, step in enumerate(noise_steps_list):
        ax = axes[i]
        data = last_seed_results[step]
        trues = np.array([d[0] for d in data])
        preds = np.array([d[1] for d in data])
        mae = np.mean(np.abs(preds - trues))
        rmse = np.sqrt(np.mean((preds - trues) ** 2))

        ax.scatter(trues, preds, alpha=0.6)
        lims = [min(trues.min(), preds.min()), max(trues.max(), preds.max())]
        ax.plot(lims, lims, "k--", alpha=0.5)
        ax.set_title(f"Noise Step {step} (t={step/total_steps:.3f})\nMAE={mae:.2f}, RMSE={rmse:.2f}", fontsize=14)
        ax.set_xlabel("DFT Bulk Modulus", fontsize=12)
        ax.set_ylabel("Predicted Bulk Modulus", fontsize=12)
        ax.tick_params(axis="both", which="major", labelsize=10)

    for i in range(n_plots, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(args.output_plot)
    print(f"Saved plot to {args.output_plot}")


if __name__ == "__main__":
    main()
