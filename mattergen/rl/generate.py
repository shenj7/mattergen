import argparse
import torch
import numpy as np
import logging
import os
from tqdm import tqdm
from pymatgen.io.cif import CifWriter
from pymatgen.core.structure import Structure

from mattergen.diffusion.lightning_module import DiffusionLightningModule
from mattergen.property_predictors.bulk_modulus_time_classifier import BulkModulusTimeClassifier
from mattergen.common.data.condition_factory import get_number_of_atoms_condition_loader
from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector
from mattergen.common.utils.eval_utils import get_crystals_list, make_structure
from mattergen.common.utils.data_utils import lattice_matrix_to_params_torch
from mattergen.diffusion.sampling.predictors import AncestralSamplingPredictor
from mattergen.diffusion.d3pm.d3pm_predictors_correctors import D3PMAncestralSamplingPredictor
from mattergen.diffusion.corruption.sde_lib import SDE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/mattergen_base/checkpoints/last.ckpt")
    parser.add_argument("--finetuned_checkpoint", type=str, required=True, help="Path to the fine-tuned state dict")
    parser.add_argument("--reward_model_checkpoint", type=str, default="checkpoints/bulk_modulus_classifier_mixed_16_large/best.pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--output_file", type=str, default="results/generated_crystals.cif")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    logger.info("Loading base model...")
    pl_module = DiffusionLightningModule.load_from_checkpoint(args.base_checkpoint)
    
    logger.info(f"Loading fine-tuned weights from {args.finetuned_checkpoint}...")
    state_dict = torch.load(args.finetuned_checkpoint, map_location=device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    pl_module.load_state_dict(state_dict)
    pl_module.to(device)
    pl_module.eval()
    
    # Load Reward Model for Evaluation
    logger.info("Loading reward model...")
    # Helper to load correct class
    def load_reward_model(checkpoint_path, device):
        from mattergen.property_predictors.bulk_modulus_time_classifier import BulkModulusTimeClassifier
        from mattergen.property_predictors.bulk_modulus_time_lora_predictor import BulkModulusLoRATimePredictor
        from mattergen.property_predictors.bulk_modulus_time_lora_mlp_predictor import BulkModulusLoRAMLPTimePredictor

        ckpt = torch.load(checkpoint_path, map_location=device)
        predictor_type = ckpt.get("config", {}).get("args", {}).get("predictor_type", "mlp")
        
        if predictor_type == "lora":
            return BulkModulusLoRATimePredictor.from_checkpoint(checkpoint_path, device=device)
        if predictor_type == "lora_mlp":
            return BulkModulusLoRAMLPTimePredictor.from_checkpoint(checkpoint_path, device=device)
            
        return BulkModulusTimeClassifier.from_checkpoint(checkpoint_path, device=device)

    reward_model = load_reward_model(args.reward_model_checkpoint, device)
    reward_model.eval()
    
    # Setup Sampler
    # Use the same predictors as training
    corruption = pl_module.diffusion_module.corruption
    predictors = {}
    for k, c in corruption.corruptions.items():
        if AncestralSamplingPredictor.is_compatible(c) and isinstance(c, SDE):
             predictors[k] = AncestralSamplingPredictor
        elif D3PMAncestralSamplingPredictor.is_compatible(c):
             predictors[k] = D3PMAncestralSamplingPredictor
    
    # We can use the higher level PredictorCorrector or just reuse the code from ddpo_trainer for consistency
    # But PredictorCorrector is the standard way.
    # Let's instantiate PredictorCorrector with AncestralSamplingPredictor
    # Infer N
    discrete_corruptions = corruption.discrete_corruptions
    if discrete_corruptions:
        N = list(discrete_corruptions.values())[0].N
    else:
        N = 1000

    sampler = PredictorCorrector(
        diffusion_module=pl_module.diffusion_module,
        predictor_partials={k: predictors[k] for k in predictors}, # PredictorPartial expects the class or partial
        corrector_partials={}, # No corrector for now to match training exactly? Or maybe use default? 
        # Usually training with just predictor (scheduler) implies sampling with just predictor is fair.
        device=device,
        n_steps_corrector=0,
        N=N,
        eps_t=1e-3
    )
    
    # Condition Loader
    condition_loader = get_number_of_atoms_condition_loader(
        num_atoms_distribution="ALEX_MP_20",
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    all_crystals = []
    all_bulk_moduli = []
    
    logger.info("Generating crystals...")
    with torch.no_grad():
        for batch_idx, (conditions, _) in enumerate(condition_loader):
            if conditions is None: continue
            
            # Check for small graphs that might cause GemNet to crash
            if "num_atoms" in conditions:
                min_atoms = conditions["num_atoms"].min()
                if min_atoms < 4:
                    logger.warning(f"Skipping batch {batch_idx} due to small graphs (min_atoms={min_atoms}). GemNet requires >3 atoms.")
                    continue
            
            # Generate
            # sample returns (batch, mean_batch)
            sample, mean_batch = sampler.sample(conditions)
            
            # Evaluate using Reward Model
            # Reward model expects (x, t=0)
            t_zeros = torch.zeros((sample.get_batch_size(),), device=device)
            
            # Ensure sample is on correct device (it should be)
            # sample = sample.to(device) 
            
            mu, _ = reward_model(sample, t_zeros)
            
            logger.info(f"Mean Predicted Bulk Modulus: {mu.mean().item():.4f}")
            
            all_bulk_moduli.extend(mu.cpu().numpy())
            
            sample = sample.to("cpu")
            conditions = conditions.to("cpu")
            
            # Reconstruct structures
            # We need to extract lattice, pos, atomic_numbers' (BatchedData) to Pymatgen Structures
            # sample contains 'pos', 'atomic_numbers', 'cell' usually.
            
            # We need to unbatch
            # We can use get_crystals_list utility if available and compatible
            
            # Check structure of sample
            frac_coords = sample["pos"]
            atom_types = sample["atomic_numbers"]
            cell = sample["cell"]
            num_atoms = sample["num_atoms"]
            
            lengths_all, angles_all = lattice_matrix_to_params_torch(cell)
            
            structures = [
                 make_structure(
                    lengths=lengths_all[i].cpu().numpy(),
                    angles=angles_all[i].cpu().numpy(),
                    atom_types=atom_types[sample.get_batch_idx("atomic_numbers") == i].cpu().numpy(),
                    frac_coords=frac_coords[sample.get_batch_idx("pos") == i].cpu().numpy(),
                )
                for i in range(sample.get_batch_size())
            ]
            
            all_crystals.extend(structures)
            
    # Report
    all_bulk_moduli = np.array(all_bulk_moduli)
    logger.info(f"Generated {len(all_crystals)} crystals.")
    logger.info(f"Mean Bulk Modulus: {np.mean(all_bulk_moduli):.2f}")
    logger.info(f"Max Bulk Modulus: {np.max(all_bulk_moduli):.2f}")
    
    # Save best
    # Sort by bulk modulus
    sorted_indices = np.argsort(all_bulk_moduli)[::-1]
    best_indices = sorted_indices[:10] # Top 10
    
    cif_path = args.output_file
    logger.info(f"Saving top 10 crystals to {cif_path}")
    
    cw = CifWriter(all_crystals[sorted_indices[0]]) # Write first
    # Actually CifWriter takes one struct. We can append or separate files.
    # Or use multiple writes.
    
    with open(cif_path, "w") as f:
        for idx in best_indices:
            s = all_crystals[idx]
            val = all_bulk_moduli[idx]
            f.write(f"# Bulk Modulus: {val:.4f}\n")
            f.write(s.to(fmt="cif"))
            f.write("\n")

if __name__ == "__main__":
    main()
