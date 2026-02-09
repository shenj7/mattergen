#!/usr/bin/env python3
"""
Test the calc_bulk_modulus function against random entries in the dataset.
Compares calculated bulk modulus vs dft_bulk_modulus and ml_bulk_modulus (if available).
Optionally includes predictions from a trained classifier checkpoint.

Usage:
    # From cache directory:
    python test_bulk_modulus.py [--n_samples 10] [--cache_dir path/to/cache] [--output plot.png]
    
    # With classifier predictions:
    python test_bulk_modulus.py --cache_dir datasets/cache/alex_mp_20/train --classifier_ckpt checkpoints/bulk_modulus_classifier_mixed_16_large/best.pt --n_samples 30
"""

import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser

# Import the function we want to test
from calc_bulk_modulus_single import calc_bulk_modulus_value


def load_dataset_from_csv(csv_path: Path) -> dict:
    """Load dataset from a CSV file with CIF structures."""
    import pandas as pd
    
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    data = {
        'structures': [],
        'structure_id': [],
        'dft_bulk_modulus': [],
        'ml_bulk_modulus': [],
        'formulas': [],
    }
    
    print(f"Parsing {len(df)} structures from CSV...")
    for idx, row in df.iterrows():
        try:
            struct = CifParser.from_str(row['cif']).parse_structures(primitive=True)[0]
            data['structures'].append(struct)
            data['structure_id'].append(row.get('material_id', f'idx_{idx}'))
            data['formulas'].append(row.get('reduced_formula', row.get('pretty_formula', str(struct.composition.reduced_formula))))
            
            dft_val = row.get('dft_bulk_modulus', np.nan)
            ml_val = row.get('ml_bulk_modulus', np.nan)
            data['dft_bulk_modulus'].append(float(dft_val) if pd.notna(dft_val) else np.nan)
            data['ml_bulk_modulus'].append(float(ml_val) if pd.notna(ml_val) else np.nan)
        except Exception as e:
            print(f"  Warning: Failed to parse structure {idx}: {e}")
            continue
    
    data['dft_bulk_modulus'] = np.array(data['dft_bulk_modulus'])
    data['ml_bulk_modulus'] = np.array(data['ml_bulk_modulus'])
    
    return data


def load_dataset_from_cache(cache_dir: Path) -> dict:
    """Load properties from numpy/json cache files."""
    import json
    
    data = {}
    
    # Load core structure arrays
    data['pos'] = np.load(cache_dir / 'pos.npy')
    data['cell'] = np.load(cache_dir / 'cell.npy')
    data['atomic_numbers'] = np.load(cache_dir / 'atomic_numbers.npy')
    data['num_atoms'] = np.load(cache_dir / 'num_atoms.npy')
    data['structure_id'] = np.load(cache_dir / 'structure_id.npy', allow_pickle=True)
    
    # Load property files if they exist
    dft_bulk_path = cache_dir / 'dft_bulk_modulus.json'
    if dft_bulk_path.exists():
        with open(dft_bulk_path) as f:
            dft_data = json.load(f)
            data['dft_bulk_modulus'] = np.array(dft_data['values'])
    
    ml_bulk_path = cache_dir / 'ml_bulk_modulus.json'
    if ml_bulk_path.exists():
        with open(ml_bulk_path) as f:
            ml_data = json.load(f)
            data['ml_bulk_modulus'] = np.array(ml_data['values'])
    
    return data


def get_structure_from_cache(data: dict, idx: int) -> Structure:
    """Reconstruct a pymatgen Structure from dataset arrays."""
    from pymatgen.core import Lattice
    
    num_atoms = data['num_atoms']
    pos_offset = np.concatenate([[0], np.cumsum(num_atoms[:-1])])[idx]
    n = num_atoms[idx]
    
    frac_coords = data['pos'][pos_offset:pos_offset + n]
    atomic_numbers = data['atomic_numbers'][pos_offset:pos_offset + n]
    cell = data['cell'][idx]
    
    lattice = Lattice(cell)
    return Structure(lattice, atomic_numbers, frac_coords)


def structure_to_chemgraph(struct: Structure):
    """Convert a pymatgen Structure to a ChemGraph for the classifier."""
    from mattergen.common.data.chemgraph import ChemGraph
    from mattergen.common.data.transform import symmetrize_lattice
    
    frac_coords = torch.tensor(struct.frac_coords, dtype=torch.float32) % 1.0
    cell = torch.tensor(struct.lattice.matrix, dtype=torch.float32).unsqueeze(0)
    atomic_numbers = torch.tensor(struct.atomic_numbers, dtype=torch.long)
    num_atoms = torch.tensor(len(struct), dtype=torch.long)
    
    cg = ChemGraph(
        pos=frac_coords,
        cell=cell,
        atomic_numbers=atomic_numbers,
        num_atoms=num_atoms,
        num_nodes=num_atoms,
    )
    # Apply the same transform used during training
    cg = symmetrize_lattice(cg)
    return cg


def load_classifier(checkpoint_path: str, predictor_type: str, device: torch.device):
    """Load the bulk modulus classifier from checkpoint."""
    if predictor_type == "lora":
        from mattergen.property_predictors import BulkModulusLoRATimePredictor
        return BulkModulusLoRATimePredictor.from_checkpoint(checkpoint_path, device=device)
    elif predictor_type == "lora_mlp":
        from mattergen.property_predictors import BulkModulusLoRAMLPTimePredictor
        return BulkModulusLoRAMLPTimePredictor.from_checkpoint(checkpoint_path, device=device)
    else:  # mlp
        from mattergen.property_predictors import BulkModulusTimeClassifier
        return BulkModulusTimeClassifier.from_checkpoint(checkpoint_path, device=device)


def predict_with_classifier(classifier, struct: Structure, device: torch.device) -> float:
    """Get bulk modulus prediction from classifier at t=0."""
    from torch_geometric.data import Batch
    
    cg = structure_to_chemgraph(struct)
    batch = Batch.from_data_list([cg]).to(device)
    t = torch.zeros(1, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        mu, logvar = classifier(batch, t)
    
    return float(mu.cpu().item())


def main():
    parser = argparse.ArgumentParser(description="Test calc_bulk_modulus against dataset")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file with CIF structures (e.g., datasets/alex_mp_20/train.csv)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "data-release" / "mp-20" / "cache" / "train"),
        help="Path to cached dataset directory (used if --csv not provided)"
    )
    parser.add_argument(
        "--classifier_ckpt",
        type=str,
        default=None,
        help="Path to trained classifier checkpoint (e.g., checkpoints/bulk_modulus_classifier_mixed_16_large/best.pt)"
    )
    parser.add_argument(
        "--predictor_type",
        type=str,
        choices=["mlp", "lora", "lora_mlp"],
        default="lora_mlp",
        help="Model type for the classifier checkpoint (default: lora_mlp)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of random samples to test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bulk_modulus_comparison.png",
        help="Output plot filename"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load classifier if specified
    classifier = None
    if args.classifier_ckpt:
        print(f"Loading classifier ({args.predictor_type}) from {args.classifier_ckpt}...")
        classifier = load_classifier(args.classifier_ckpt, args.predictor_type, device)
        classifier.eval()
        print(f"Classifier loaded on {device}")
    
    # Load data from CSV or cache
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        data = load_dataset_from_csv(csv_path)
        use_csv = True
        n_structures = len(data['structures'])
    else:
        cache_dir = Path(args.cache_dir)
        if not cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {cache_dir}")
        print(f"Loading dataset from {cache_dir}...")
        data = load_dataset_from_cache(cache_dir)
        use_csv = False
        n_structures = len(data['num_atoms'])
    
    has_dft = 'dft_bulk_modulus' in data and len(data['dft_bulk_modulus']) > 0
    has_ml = 'ml_bulk_modulus' in data and len(data['ml_bulk_modulus']) > 0
    
    print(f"Dataset has {n_structures} structures")
    print(f"  dft_bulk_modulus available: {has_dft}")
    print(f"  ml_bulk_modulus available: {has_ml}")
    
    # Get valid indices (entries with finite ml_bulk_modulus AND dft_bulk_modulus)
    valid_indices = []
    for i in range(n_structures):
        dft_ok = has_dft and np.isfinite(data['dft_bulk_modulus'][i]) and data['dft_bulk_modulus'][i] > 0
        ml_ok = has_ml and np.isfinite(data['ml_bulk_modulus'][i]) and data['ml_bulk_modulus'][i] > 0
        if dft_ok and ml_ok:
            valid_indices.append(i)
    
    print(f"  Valid entries with bulk modulus: {len(valid_indices)}")
    
    if len(valid_indices) == 0:
        print("No valid entries found with bulk modulus data!")
        return
    
    # Sample random structures
    n_samples = min(args.n_samples, len(valid_indices))
    sample_indices = random.sample(valid_indices, n_samples)
    
    print(f"\nTesting {n_samples} random structures...")
    
    results = {
        'structure_id': [],
        'calc_bulk_modulus': [],
        'classifier_bulk_modulus': [],
        'dft_bulk_modulus': [],
        'ml_bulk_modulus': [],
    }
    
    for i, idx in enumerate(sample_indices):
        # Get structure based on data source
        if use_csv:
            struct = data['structures'][idx]
            struct_id = data['structure_id'][idx]
            formula = data['formulas'][idx]
        else:
            struct = get_structure_from_cache(data, idx)
            struct_id = data['structure_id'][idx]
            formula = struct.composition.reduced_formula
        
        dft_val = data['dft_bulk_modulus'][idx] if has_dft else np.nan
        ml_val = data['ml_bulk_modulus'][idx] if has_ml else np.nan
        
        print(f"\n[{i+1}/{n_samples}] Structure: {struct_id}")
        print(f"  Formula: {formula}")
        print(f"  Num atoms: {len(struct)}")
        print(f"  DFT bulk modulus: {dft_val:.2f} GPa" if np.isfinite(dft_val) else "  DFT bulk modulus: N/A")
        print(f"  ML bulk modulus: {ml_val:.2f} GPa" if np.isfinite(ml_val) else "  ML bulk modulus: N/A")
        
        # Calculate bulk modulus using our function
        try:
            from ase import Atoms
            ase_atoms = Atoms(
                numbers=struct.atomic_numbers,
                cell=struct.lattice.matrix,
                scaled_positions=struct.frac_coords,
                pbc=True
            )
            calc_val = calc_bulk_modulus_value(ase_atoms)
            print(f"  Calculated bulk modulus: {calc_val:.2f} GPa")
        except Exception as e:
            print(f"  Calculation failed: {e}")
            calc_val = np.nan
        
        # Get classifier prediction if available
        classifier_val = np.nan
        if classifier is not None:
            try:
                classifier_val = predict_with_classifier(classifier, struct, device)
                print(f"  Classifier bulk modulus: {classifier_val:.2f} GPa")
            except Exception as e:
                print(f"  Classifier prediction failed: {e}")
        
        results['structure_id'].append(struct_id)
        results['calc_bulk_modulus'].append(calc_val)
        results['classifier_bulk_modulus'].append(classifier_val)
        results['dft_bulk_modulus'].append(dft_val)
        results['ml_bulk_modulus'].append(ml_val)
    
    # Create comparison plot
    calc_values = np.array(results['calc_bulk_modulus'])
    classifier_values = np.array(results['classifier_bulk_modulus'])
    dft_values = np.array(results['dft_bulk_modulus'])
    ml_values = np.array(results['ml_bulk_modulus'])
    
    # Filter out NaN values for plotting
    valid_calc_dft = np.isfinite(calc_values) & np.isfinite(dft_values)
    valid_calc_ml = np.isfinite(calc_values) & np.isfinite(ml_values)
    valid_clf_dft = np.isfinite(classifier_values) & np.isfinite(dft_values)
    valid_clf_ml = np.isfinite(classifier_values) & np.isfinite(ml_values)
    
    # Determine number of plots
    if classifier is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes = list(axes)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    
    # Plot 1: Calculated vs DFT
    ax1 = axes[0]
    if valid_calc_dft.any():
        ax1.scatter(dft_values[valid_calc_dft], calc_values[valid_calc_dft], alpha=0.7, s=60)
        lims = [
            min(dft_values[valid_calc_dft].min(), calc_values[valid_calc_dft].min()),
            max(dft_values[valid_calc_dft].max(), calc_values[valid_calc_dft].max())
        ]
        ax1.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
        ax1.set_xlabel('DFT Bulk Modulus (GPa)')
        ax1.set_ylabel('Calculated Bulk Modulus (GPa)')
        ax1.set_title('Calculated vs DFT')
        ax1.legend()
        mae = np.mean(np.abs(calc_values[valid_calc_dft] - dft_values[valid_calc_dft]))
        rmse = np.sqrt(np.mean((calc_values[valid_calc_dft] - dft_values[valid_calc_dft])**2))
        ax1.text(0.05, 0.95, f'MAE: {mae:.2f} GPa\nRMSE: {rmse:.2f} GPa',
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax1.text(0.5, 0.5, 'No valid DFT data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Calculated vs DFT')
    
    # Plot 2: Calculated vs ML
    ax2 = axes[1]
    if valid_calc_ml.any():
        ax2.scatter(ml_values[valid_calc_ml], calc_values[valid_calc_ml], alpha=0.7, s=60, color='orange')
        lims = [
            min(ml_values[valid_calc_ml].min(), calc_values[valid_calc_ml].min()),
            max(ml_values[valid_calc_ml].max(), calc_values[valid_calc_ml].max())
        ]
        ax2.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
        ax2.set_xlabel('ML Bulk Modulus (GPa)')
        ax2.set_ylabel('Calculated Bulk Modulus (GPa)')
        ax2.set_title('Calculated vs ML')
        ax2.legend()
        mae = np.mean(np.abs(calc_values[valid_calc_ml] - ml_values[valid_calc_ml]))
        rmse = np.sqrt(np.mean((calc_values[valid_calc_ml] - ml_values[valid_calc_ml])**2))
        ax2.text(0.05, 0.95, f'MAE: {mae:.2f} GPa\nRMSE: {rmse:.2f} GPa',
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax2.text(0.5, 0.5, 'No valid ML data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Calculated vs ML')
    
    # Plot 3: Classifier vs DFT (if classifier is used)
    if classifier is not None:
        ax3 = axes[2]
        if valid_clf_dft.any():
            ax3.scatter(dft_values[valid_clf_dft], classifier_values[valid_clf_dft], alpha=0.7, s=60, color='green')
            lims = [
                min(dft_values[valid_clf_dft].min(), classifier_values[valid_clf_dft].min()),
                max(dft_values[valid_clf_dft].max(), classifier_values[valid_clf_dft].max())
            ]
            ax3.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
            ax3.set_xlabel('DFT Bulk Modulus (GPa)')
            ax3.set_ylabel('Classifier Bulk Modulus (GPa)')
            ax3.set_title('Classifier vs DFT')
            ax3.legend()
            mae = np.mean(np.abs(classifier_values[valid_clf_dft] - dft_values[valid_clf_dft]))
            rmse = np.sqrt(np.mean((classifier_values[valid_clf_dft] - dft_values[valid_clf_dft])**2))
            ax3.text(0.05, 0.95, f'MAE: {mae:.2f} GPa\nRMSE: {rmse:.2f} GPa',
                     transform=ax3.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax3.text(0.5, 0.5, 'No valid classifier data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Classifier vs DFT')
        
        # Plot 4: Classifier vs ML
        ax4 = axes[3]
        if valid_clf_ml.any():
            ax4.scatter(ml_values[valid_clf_ml], classifier_values[valid_clf_ml], alpha=0.7, s=60, color='purple')
            lims = [
                min(ml_values[valid_clf_ml].min(), classifier_values[valid_clf_ml].min()),
                max(ml_values[valid_clf_ml].max(), classifier_values[valid_clf_ml].max())
            ]
            ax4.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
            ax4.set_xlabel('ML Bulk Modulus (GPa)')
            ax4.set_ylabel('Classifier Bulk Modulus (GPa)')
            ax4.set_title('Classifier vs ML')
            ax4.legend()
            mae = np.mean(np.abs(classifier_values[valid_clf_ml] - ml_values[valid_clf_ml]))
            rmse = np.sqrt(np.mean((classifier_values[valid_clf_ml] - ml_values[valid_clf_ml])**2))
            ax4.text(0.05, 0.95, f'MAE: {mae:.2f} GPa\nRMSE: {rmse:.2f} GPa',
                     transform=ax4.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax4.text(0.5, 0.5, 'No valid classifier data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Classifier vs ML')
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {args.output}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if valid_calc_dft.any():
        mae = np.mean(np.abs(calc_values[valid_calc_dft] - dft_values[valid_calc_dft]))
        rmse = np.sqrt(np.mean((calc_values[valid_calc_dft] - dft_values[valid_calc_dft])**2))
        print(f"Calculated vs DFT: MAE = {mae:.2f} GPa, RMSE = {rmse:.2f} GPa")
    if valid_calc_ml.any():
        mae = np.mean(np.abs(calc_values[valid_calc_ml] - ml_values[valid_calc_ml]))
        rmse = np.sqrt(np.mean((calc_values[valid_calc_ml] - ml_values[valid_calc_ml])**2))
        print(f"Calculated vs ML:  MAE = {mae:.2f} GPa, RMSE = {rmse:.2f} GPa")
    if classifier is not None and valid_clf_dft.any():
        mae = np.mean(np.abs(classifier_values[valid_clf_dft] - dft_values[valid_clf_dft]))
        rmse = np.sqrt(np.mean((classifier_values[valid_clf_dft] - dft_values[valid_clf_dft])**2))
        print(f"Classifier vs DFT: MAE = {mae:.2f} GPa, RMSE = {rmse:.2f} GPa")
    if classifier is not None and valid_clf_ml.any():
        mae = np.mean(np.abs(classifier_values[valid_clf_ml] - ml_values[valid_clf_ml]))
        rmse = np.sqrt(np.mean((classifier_values[valid_clf_ml] - ml_values[valid_clf_ml])**2))
        print(f"Classifier vs ML:  MAE = {mae:.2f} GPa, RMSE = {rmse:.2f} GPa")


if __name__ == "__main__":
    main()
