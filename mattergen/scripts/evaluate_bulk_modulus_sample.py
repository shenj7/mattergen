import argparse
import csv
import random
from pathlib import Path
import sys
import os

import torch
import numpy as np
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm

# Add the project root to the python path so we can import mattergen
# Assuming this script is in mattergen/scripts/
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from mattergen.common.data.dataset import CrystalDataset
from mattergen.common.data.transform import symmetrize_lattice
from mattergen.property_predictors import (
    BulkModulusLoRAMLPTimePredictor,
    BulkModulusLoRATimePredictor,
    BulkModulusTimeClassifier,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate bulk modulus predictor on a sample.")
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
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predictor_type", type=str, default=None, choices=["mlp", "lora", "lora_mlp"], help="Force predictor type if not inferable.")

    return parser.parse_args()

def load_model(checkpoint_path, device, predictor_type=None):
    if not os.path.exists(checkpoint_path):
         raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Try to infer type or use default/argument
    # For now, we'll try to load as Classifier first, as per existing scripts, 
    # but based on the user request it might be a specific type. 
    # The existing script `eval_bulk_modulus_predictor.py` uses an argument.
    # The user didn't specify, but "bulk_modulus_classifier_mixed_16_large" suggests a classifier (MLP).
    
    # Let's try to load the checkpoint and see if we can guess or just try-catch.
    # Actually, `BulkModulusTimeClassifier.from_checkpoint` is the standard way.
    
    # To be safe, we can inspect the checkpoint slightly or just use the class.
    # If the user provides a predictor_type, we use it.
    
    if predictor_type == "lora":
         return BulkModulusLoRATimePredictor.from_checkpoint(checkpoint_path, device=device)
    elif predictor_type == "lora_mlp":
        return BulkModulusLoRAMLPTimePredictor.from_checkpoint(checkpoint_path, device=device)
    else:
        # Default to MLP / Classifier
        return BulkModulusTimeClassifier.from_checkpoint(checkpoint_path, device=device)

def main():
    args = parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
         # Try relative to project root if not found
         dataset_path = project_root / args.dataset_path
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path} or {dataset_path}")

    dataset = CrystalDataset.from_cache_path(
        str(dataset_path),
        properties=[args.property_name],
        transforms=[symmetrize_lattice]
    )
    
    print(f"Total entries in dataset: {len(dataset)}")

    # Filter for non-null property
    # We can check the property values directly from the dataset's stored properties
    # CrystalDataset stores properties in self.properties[prop_name] as numpy arrays
    
    prop_values = dataset.properties[args.property_name]
    valid_indices = np.where(np.isfinite(prop_values))[0]
    print(f"Entries with valid {args.property_name}: {len(valid_indices)}")
    
    if len(valid_indices) < args.num_samples:
        print(f"Warning: Only {len(valid_indices)} valid entries found, sampling all of them.")
        sample_indices = valid_indices
    else:
        sample_indices = np.random.choice(valid_indices, size=args.num_samples, replace=False)
    
    # Create subset
    subset = dataset.subset(sample_indices)
    
    loader = GeoDataLoader(subset, batch_size=args.batch_size, shuffle=False)
    
    # Load model
    print(f"Loading model from {args.checkpoint_path}...")
    model = load_model(args.checkpoint_path, device, args.predictor_type)
    model.eval()
    
    results = []
    
    print("Evaluating...")
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)
            # Evaluate at t=0 (clean data)
            t = torch.zeros(batch.num_graphs, device=device)
            
            mu, logvar = model(batch, t)
            
            targets = batch[args.property_name]
            
            mu = mu.cpu().numpy()
            logvar = logvar.cpu().numpy()
            targets = targets.cpu().numpy()
            
            for m, lv, t_val in zip(mu, logvar, targets):
                results.append({
                    "predicted": m.item(),
                    "logvar": lv.item(),
                    "true": t_val.item(),
                    "error": m.item() - t_val.item(),
                    "abs_error": abs(m.item() - t_val.item()),
                    "sq_error": (m.item() - t_val.item())**2
                })

    # Calculate metrics
    abs_errors = [r["abs_error"] for r in results]
    sq_errors = [r["sq_error"] for r in results]
    
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(sq_errors))
    
    print(f"\nResults for {len(results)} samples:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Save to CSV
    output_csv = "evaluation_sample_results.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["predicted", "logvar", "true", "error", "abs_error", "sq_error"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Detailed results written to {output_csv}")

if __name__ == "__main__":
    main()
