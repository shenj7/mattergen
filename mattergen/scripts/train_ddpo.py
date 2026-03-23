#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Train MatterGen with DDPO (Denoising Diffusion Policy Optimization).

This script implements stable DDPO training with a decoupled hybrid PPO strategy
that handles the mixed continuous (coordinates/lattice) and discrete (atom types)
action space.

Usage:
    python -m mattergen.scripts.train_ddpo \
        --checkpoint checkpoints/mattergen_base \
        --critic-checkpoint checkpoints/bulk_modulus_classifier_mixed_16_large/best.pt \
        --output-dir checkpoints/ddpo_bulk_modulus \
        --num-epochs 100

Key features:
- Separate PPO ratios for continuous and discrete actions
- Stricter clipping for discrete actions (ε=0.1 vs ε=0.2)
- KL divergence anchor against frozen reference model
- Entropy regularization for exploration
- NaN protection via probability clamping
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from hydra.utils import instantiate
from pymatgen.io.ase import AseAtomsAdaptor

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.eval_utils import load_model_diffusion
from mattergen.common.utils.globals import get_device
from mattergen.generator import CrystalGenerator
from mattergen.rl import DDPOConfig, DDPOTrainer, MatterGenActor, ValueNetwork

# Import bulk modulus calculator
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from calc_bulk_modulus_single import calc_bulk_modulus_value
from mattergen.property_predictors.bulk_modulus_time_lora_mlp_predictor import BulkModulusLoRAMLPTimePredictor


def create_reward_fn(reward_net, device: torch.device):
    """
    Create a reward function using the frozen BulkModulusLoRAMLPTimePredictor.
    Evaluates x_0 at t=0 to predict final bulk modulus.

    Returns:
        Callable that takes a ChemGraph and returns a (batch_size,) Tensor
    """
    def reward_fn(x_0: ChemGraph) -> torch.Tensor:
        from mattergen.common.data.transform import symmetrize_lattice
        with torch.no_grad():
            t = torch.zeros((x_0.get_batch_size(),), device=device)

            # Wrap to periodic cell bounds
            x_eval = x_0.clone()
            x_eval = x_eval.replace(pos=x_eval.pos % 1.0)

            # Reconstruct symmetric matrix based on primitive cell lengths + angles
            x_eval = symmetrize_lattice(x_eval)

            # Evaluate the bulk modulus for the batch
            # Shape is (batch_size,)
            mu, logvar = reward_net(x_eval, t)
            return mu

    return reward_fn


def create_mattersim_reward_fn(device: torch.device, n_points: int = 5, strain: float = 0.03):
    """
    Create a reward function using MatterSim force field + E(V) curve fitting.

    For each crystal in the batch: converts ChemGraph → ASE Atoms, samples
    n_points volumes spanning ±strain, fits a quadratic E(V) curve, and returns
    the bulk modulus B = V0 * d²E/dV² in GPa.

    Uses a single shared MatterSimCalculator (loaded once) to avoid reloading the
    1M-parameter model on every call. Energy evaluations run inside torch.enable_grad()
    because MatterSim computes forces via autograd internally and will fail if called
    from inside a torch.no_grad() context.

    Returns:
        Callable that takes a ChemGraph and returns a (batch_size,) Tensor
    """
    import numpy as np
    from ase import Atoms as AseAtoms
    from mattersim.forcefield import MatterSimCalculator
    from calc_bulk_modulus_single import _fit_bulk_modulus

    print("Loading MatterSim calculator (once)...")
    shared_calc = MatterSimCalculator()

    vol_scales = np.linspace(1.0 - strain, 1.0 + strain, n_points)
    len_scales = vol_scales ** (1.0 / 3.0)

    def _eval_one(atoms: AseAtoms) -> float:
        """E(V) sweep + quadratic fit for a single structure."""
        atoms = atoms.copy()
        atoms.calc = shared_calc
        base_cell = atoms.get_cell().array.copy()
        base_pos = atoms.get_positions().copy()
        ev_rows = []
        for scale in len_scales:
            atoms.set_cell(base_cell * scale, scale_atoms=False)
            atoms.set_positions(base_pos * scale)
            # MatterSim uses autograd internally to compute forces, so we need
            # gradient tracking even though we only read the energy scalar.
            with torch.enable_grad():
                E = float(atoms.get_potential_energy())
            V = float(atoms.get_volume())
            ev_rows.append((V, E))
        return _fit_bulk_modulus(ev_rows)

    def reward_fn(x_0: ChemGraph) -> torch.Tensor:
        from mattergen.common.data.transform import symmetrize_lattice

        # Same preprocessing as the neural-net reward
        x_eval = x_0.clone()
        x_eval = x_eval.replace(pos=x_eval.pos % 1.0)
        x_eval = symmetrize_lattice(x_eval)

        batch_size = x_eval.get_batch_size()
        batch_idx = x_eval.get_batch_idx("pos").cpu()
        frac_coords = x_eval["pos"].detach().cpu().numpy()      # (total_atoms, 3) fractional
        atomic_numbers = x_eval["atomic_numbers"].detach().cpu().numpy()  # (total_atoms,)
        cells = x_eval["cell"].detach().cpu().numpy()           # (batch_size, 3, 3)

        rewards = []
        for i in range(batch_size):
            mask = (batch_idx == i).numpy()
            atoms = AseAtoms(
                numbers=atomic_numbers[mask],
                cell=cells[i],
                scaled_positions=frac_coords[mask],
                pbc=True,
            )
            try:
                bm = _eval_one(atoms)
                # Clip to physically plausible range (diamond ~440 GPa is upper bound)
                bm = max(0.0, min(float(bm), 500.0))
            except Exception as e:
                print(f"  MatterSim reward failed for crystal {i}: {e}")
                bm = 0.0
            rewards.append(bm)

        return torch.tensor(rewards, dtype=torch.float32, device=device)

    return reward_fn


def main():
    parser = argparse.ArgumentParser(description="Train MatterGen with DDPO")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/mattergen_base",
        help="Path to pretrained MatterGen checkpoint",
    )
    parser.add_argument(
        "--critic-checkpoint",
        type=str,
        default="checkpoints/bulk_modulus_classifier_mixed_16_large/best.pt",
        help="Path to bulk modulus classifier checkpoint for critic initialization",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/ddpo_bulk_modulus",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (trajectories per epoch)",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=3,
        help="Number of PPO update epochs per rollout",
    )
    parser.add_argument(
        "--actor-lr",
        type=float,
        default=1e-4,
        help="Actor learning rate",
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=1e-4,
        help="Critic learning rate",
    )
    parser.add_argument(
        "--clip-eps-cont",
        type=float,
        default=0.2,
        help="PPO clipping for continuous actions",
    )
    parser.add_argument(
        "--clip-eps-disc",
        type=float,
        default=0.1,
        help="PPO clipping for discrete actions (stricter)",
    )
    parser.add_argument(
        "--kl-coeff",
        type=float,
        default=0.1,
        help="KL divergence coefficient (anchor strength)",
    )
    parser.add_argument(
        "--entropy-coeff",
        type=float,
        default=0.01,
        help="Entropy regularization coefficient",
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        default="bulk_modulus",
        choices=["bulk_modulus", "simple_density", "mattersim"],
        help="Reward function type. 'mattersim' uses MatterSim + E(V) curve fitting (slow but accurate).",
    )
    parser.add_argument(
        "--mattersim-n-points",
        type=int,
        default=5,
        help="Number of volume-strain points for MatterSim E(V) fit (only used with --reward-type mattersim)",
    )
    parser.add_argument(
        "--mattersim-strain",
        type=float,
        default=0.03,
        help="Max volumetric strain fraction for MatterSim E(V) fit (only used with --reward-type mattersim)",
    )
    parser.add_argument(
        "--num-rollout-batches",
        type=int,
        default=1,
        help="Number of sequential diffusion passes to run per collection epoch. Default is 1 wide batch.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )
    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_dict = vars(args)
    config_dict["timestamp"] = datetime.now().isoformat()
    with open(output_path / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    device = get_device()
    print(f"Using device: {device}")
    print("=" * 60)

    # Load MatterGen model
    checkpoint_path = Path(args.checkpoint).resolve()
    print(f"Loading MatterGen from: {checkpoint_path}")
    checkpoint_info = MatterGenCheckpointInfo(model_path=checkpoint_path)
    model = load_model_diffusion(checkpoint_info)
    model = model.to(device)
    
    diffusion_module = model.diffusion_module
    denoiser = diffusion_module.model
    
    # Create MatterGenActor wrapper
    print("Creating MatterGenActor...")
    actor = MatterGenActor(denoiser=denoiser, diffusion_module=diffusion_module)
    
    # Create ValueNetwork (from classifier checkpoint)
    critic_path = Path(args.critic_checkpoint)
    if critic_path.exists():
        print(f"Loading critic from: {critic_path}")
        critic = ValueNetwork.from_classifier_checkpoint(critic_path, device=device)
    else:
        print("Critic checkpoint not found, creating from scratch")
        critic = ValueNetwork(hidden_dim=512, mlp_hidden_dim=256)

    # Load frozen reward network only when using the neural-net reward
    if args.reward_type != "mattersim":
        if critic_path.exists():
            print(f"Loading frozen reward network from: {critic_path}")
            reward_net = BulkModulusLoRAMLPTimePredictor.from_checkpoint(critic_path, device=device)
        else:
            reward_net = BulkModulusLoRAMLPTimePredictor(hidden_dim=512, mlp_hidden_dim=256).to(device)
        reward_net.eval()
        for param in reward_net.parameters():
            param.requires_grad = False
    
    # Create sampler for trajectory collection
    generator = CrystalGenerator(
        checkpoint_info=checkpoint_info,
        batch_size=args.batch_size,
        num_batches=1,
    )
    sampling_config = generator.load_sampling_config(
        batch_size=args.batch_size,
        num_batches=1,
    )
    condition_loader = generator.get_condition_loader(sampling_config)
    sampler_partial = instantiate(sampling_config.sampler_partial)
    sampler = sampler_partial(pl_module=model)
    
    # Create reward function
    if args.reward_type == "mattersim":
        print(f"Using MatterSim reward (n_points={args.mattersim_n_points}, strain={args.mattersim_strain})")
        reward_fn = create_mattersim_reward_fn(
            device=device,
            n_points=args.mattersim_n_points,
            strain=args.mattersim_strain,
        )
    else:
        reward_fn = create_reward_fn(reward_net, device=device)
    
    # Create DDPO config
    ddpo_config = DDPOConfig(
        clip_eps_cont=args.clip_eps_cont,
        clip_eps_disc=args.clip_eps_disc,
        ppo_epochs=args.ppo_epochs,
        kl_coeff=args.kl_coeff,
        entropy_coeff=args.entropy_coeff,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
    )
    # ppo_mb_size is now a proper DDPOConfig field (default 4)
    
    # Create trainer
    print("Creating DDPOTrainer...")
    trainer = DDPOTrainer(
        actor=actor,
        critic=critic,
        reward_fn=reward_fn,
        config=ddpo_config,
        device=device,
    )
    
    # Resume if specified
    start_epoch = 0
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            print(f"Resuming from: {resume_path}")
            start_epoch = trainer.load_checkpoint(resume_path)
    
    print("=" * 60)
    print(f"Starting DDPO training for {args.num_epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  PPO epochs: {args.ppo_epochs}")
    print(f"  Clip (cont/disc): {args.clip_eps_cont}/{args.clip_eps_disc}")
    print(f"  KL coeff: {args.kl_coeff}")
    print(f"  Entropy coeff: {args.entropy_coeff}")
    print("=" * 60)
    
    # Train
    metrics_history = trainer.train(
        sampler=sampler,
        condition_loader=condition_loader,
        num_epochs=args.num_epochs,
        num_diffusion_batches=getattr(args, "num_rollout_batches", 1),
        save_path=output_path,
        save_every=args.save_every,
    )
    
    # Save final metrics
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)
    
    print("=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {output_path}")
    
    # Summary
    if metrics_history:
        rewards = [m.get("mean_reward", 0.0) for m in metrics_history]
        # if reward is a tensor, we make sure it's printed as float
        if isinstance(rewards[-1], torch.Tensor):
             print(f"Final mean reward: {rewards[-1].mean().item():.2f}")
             
        try:
             best_reward = max(m.get("mean_reward", 0.0) for m in metrics_history)
             if isinstance(best_reward, torch.Tensor):
                 best_reward = best_reward.mean().item()
             print(f"Best mean reward: {best_reward:.2f}")
        except Exception:
             pass


if __name__ == "__main__":
    main()
