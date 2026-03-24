"""
Decoupled DDPO Trainer for MatterGen.

This module implements a stable Denoising Diffusion Policy Optimization (DDPO) trainer
that handles MatterGen's hybrid action space (discrete atom types + continuous coordinates/lattice)
using a decoupled PPO strategy with:
- Separate PPO ratios for continuous and discrete actions
- Stricter clipping for discrete actions (ε=0.1 vs ε=0.2)
- KL divergence anchor against frozen reference model
- Entropy regularization for exploration
- NaN protection via probability clamping
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch_scatter import scatter_mean

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.gemnet.gemnet import GemNetT
from mattergen.common.gemnet.layers.embedding_block import AtomEmbedding
from mattergen.diffusion.model_utils import NoiseLevelEncoding
from mattergen.property_predictors import BulkModulusTimeClassifier


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DDPOConfig:
    """Configuration for DDPO training."""
    # PPO hyperparameters
    clip_eps_cont: float = 0.2  # Clipping for continuous actions
    clip_eps_disc: float = 0.1  # Stricter clipping for discrete actions
    ppo_epochs: int = 3  # K epochs of PPO updates per rollout
    
    # Regularization
    kl_coeff: float = 0.1  # KL divergence coefficient (anchor)
    entropy_coeff: float = 0.01  # Entropy bonus coefficient
    value_coeff: float = 0.5  # Value loss coefficient
    
    # Numerical stability
    prob_clamp_min: float = 1e-6
    # NOTE: no prob_clamp_max — continuous log-probs can be positive.
    grad_clip_norm: float = 1.0

    # Performance: fraction of diffusion timesteps to sample per PPO update epoch.
    # With N=100 steps, 0.2 → 20 steps used → ~5x faster updates with negligible quality loss.
    timestep_subsample_frac: float = 0.2

    # Minibatch size (number of timestep-steps) per gradient update
    ppo_mb_size: int = 4

    # Learning rates
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4


# =============================================================================
# Trajectory Storage
# =============================================================================

class TrajectoryStep(NamedTuple):
    """Single step in a diffusion trajectory."""
    state: ChemGraph  # Noisy state x_t
    timestep: torch.Tensor  # t
    action_pos: torch.Tensor  # Sampled position noise
    action_cell: torch.Tensor  # Sampled cell noise
    action_atoms: torch.Tensor  # Sampled atom types
    log_prob_cont: torch.Tensor  # Log prob of continuous actions
    log_prob_disc: torch.Tensor  # Log prob of discrete actions
    dt: torch.Tensor  # Timestep spacing parameter


@dataclass
class Trajectory:
    """Full diffusion trajectory with terminal reward."""
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_sample: ChemGraph | None = None
    reward: float = 0.0


# =============================================================================
# MatterGenActor: RL Wrapper around Diffusion Model
# =============================================================================

class ActorOutput(NamedTuple):
    """Output from the actor forward pass."""
    loc_pos: torch.Tensor  # Mean for position denoising
    scale_pos: torch.Tensor  # Std for position denoising
    loc_cell: torch.Tensor  # Mean for cell denoising
    scale_cell: torch.Tensor  # Std for cell denoising
    logits_atoms: torch.Tensor  # Logits for atom type prediction


class MatterGenActor(nn.Module):
    
    """RL wrapper around the MatterGen denoiser that exposes the policy interface.
    
    Treats:
    - Coordinates/lattice: Continuous actions with Gaussian distribution
    - Atom types: Discrete actions with Categorical distribution
    """
    
    
    def __init__(
        self,
        denoiser: nn.Module,
        diffusion_module: nn.Module,
    ) -> None:
        super().__init__()
        self.diffusion_module = diffusion_module
        
        # Apply LoRA to the denoiser and freeze the base weights
        from mattergen.common.peft.lora import apply_lora
        
        # Freeze all original parameters
        for param in denoiser.parameters():
            param.requires_grad = False
            
        # Apply LoRA layers ONLY to specific subsets to save VRAM on 8GB cards
        # We target the final projection heads rather than intermediate embeddings
        self.denoiser = apply_lora(
            denoiser,
            rank=16,
            alpha=16.0,
            target_modules=["fc_atom", "out_energy", "out_forces"]
        )
        
    def forward(
        self,
        x: ChemGraph,
        t: torch.Tensor,
        dt: torch.Tensor,
    ) -> ActorOutput:
        
        """
        Forward pass returning distribution parameters.
        
        Args:
            x: Noisy crystal state at timestep t
            t: Diffusion timesteps (batch_size,)
            dt: Timestep spacing (scalar tensor)
            
        Returns:
            ActorOutput with loc, scale, and logits
        """
        # Get score model output (this is the denoiser prediction)
        score_output = self.denoiser(x, t)
        
        score_pos = score_output["pos"]
        score_cell = score_output["cell"]
        score_atoms = score_output["atomic_numbers"]
        
        batch_idx_pos = x.get_batch_idx("pos")
        corruptions = getattr(self.diffusion_module.corruption, "corruptions", {})
        
        sde_pos = corruptions.get("pos")
        sde_cell = corruptions.get("cell")
        sde_atoms = corruptions.get("atomic_numbers")
        
        # 1. POSITIONS (Wrapped Ancestral Sampling)
        if sde_pos is not None:
            from mattergen.diffusion.wrapped.wrapped_predictors_correctors import WrappedAncestralSamplingPredictor
            from mattergen.diffusion.sampling.predictors import AncestralSamplingPredictor
            
            if hasattr(sde_pos, "wrap"):
                pred_pos = WrappedAncestralSamplingPredictor(corruption=sde_pos, score_fn=None)
            else:
                pred_pos = AncestralSamplingPredictor(corruption=sde_pos, score_fn=None)
            
            x_coeff_pos, score_coeff_pos, std_pos = pred_pos._get_coeffs(
                x=x["pos"], t=t, dt=dt, batch_idx=batch_idx_pos, batch=x
            )
            loc_pos = x_coeff_pos * x["pos"] + score_coeff_pos * score_pos
            scale_pos = std_pos.clamp(min=1e-6)
            
            if hasattr(sde_pos, "wrap"):
                loc_pos = sde_pos.wrap(loc_pos)
        else:
            loc_pos = score_pos
            scale_pos = torch.ones_like(loc_pos) * 0.1
            
        # 2. CELL (Lattice Ancestral Sampling)
        if sde_cell is not None:
            from mattergen.common.diffusion.predictors_correctors import LatticeAncestralSamplingPredictor
            pred_cell = LatticeAncestralSamplingPredictor(corruption=sde_cell, score_fn=None)
            x_coeff_cell, score_coeff_cell, std_cell = pred_cell._get_coeffs(
                x=x["cell"], t=t, dt=dt, batch_idx=None, batch=x
            )
            mean_coeff = 1.0 - x_coeff_cell
            limit_mean = sde_cell.get_limit_mean(x=x["cell"], batch=x) if hasattr(sde_cell, "get_limit_mean") else 0.0
            loc_cell = x_coeff_cell * x["cell"] + score_coeff_cell * score_cell + mean_coeff * limit_mean
            scale_cell = std_cell.clamp(min=1e-6)
        else:
            loc_cell = score_cell
            scale_cell = torch.ones_like(loc_cell) * 0.1
            
        # 3. ATOMIC NUMBERS (D3PM Discrete Posterior q)
        if sde_atoms is not None:
            from mattergen.diffusion.discrete_time import to_discrete_time
            t_discrete = to_discrete_time(t=t, N=sde_atoms.N, T=sde_atoms.T)
            class_probs = torch.softmax(score_atoms, dim=-1)
            
            class_logits, _ = sde_atoms.d3pm.sample_and_compute_posterior_q(
                x_0=class_probs,
                t=t_discrete[batch_idx_pos].to(torch.long),
                make_one_hot=False,
                samples=sde_atoms._to_zero_based(x["atomic_numbers"]),
                return_logits=True,
            )
            logits_atoms = class_logits
        else:
            logits_atoms = score_atoms
        
        return ActorOutput(
            loc_pos=loc_pos,
            scale_pos=scale_pos,
            loc_cell=loc_cell,
            scale_cell=scale_cell,
            logits_atoms=logits_atoms,
        )
    
    def evaluate_actions(
        self,
        states: list[ChemGraph],
        timesteps: torch.Tensor,
        dts: torch.Tensor,
        actions_pos: torch.Tensor,
        actions_cell: torch.Tensor,
        actions_atoms: torch.Tensor,
        config: DDPOConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        """
        Evaluate log probabilities of given actions.
        
        Returns:
            log_prob_cont: Log probability of continuous actions (batched)
            log_prob_disc: Log probability of discrete actions (batched)
            entropy_disc: Entropy of discrete distribution (batched)
        """
        log_probs_cont = []
        log_probs_disc = []
        entropies = []
        
        for i, (state, t, dt) in enumerate(zip(states, timesteps, dts)):
            # Forward pass for this state. t and dt are (batch_size,) scalars representing the batch
            output = self.forward(state, t, dt)
            
            # Continuous distribution (position + cell)
            dist_pos = Normal(output.loc_pos, output.scale_pos)
            dist_cell = Normal(output.loc_cell, output.scale_cell)
            
            # Get step's actions
            step_actions_pos = actions_pos[i] if actions_pos.dim() == 3 else actions_pos
            step_actions_cell = actions_cell[i] if actions_cell.dim() == 4 else actions_cell
            step_actions_atoms = actions_atoms[i] if actions_atoms.dim() == 2 else actions_atoms
            
            from torch_scatter import scatter
            batch_idx_pos = state.get_batch_idx("pos")
            
            # Log prob continuous: grouped by individual graphs
            lp_pos_atoms = dist_pos.log_prob(step_actions_pos).sum(dim=-1)
            lp_pos = scatter(lp_pos_atoms, batch_idx_pos, dim=0, reduce="sum")
            
            lp_cell = dist_cell.log_prob(step_actions_cell).sum(dim=(1, 2))
            log_prob_cont = lp_pos + lp_cell
            
            # Only clamp the lower bound (see sample_action for rationale).
            log_prob_cont = log_prob_cont.clamp(
                min=torch.log(torch.tensor(config.prob_clamp_min, device=log_prob_cont.device)),
            )
            
            # Discrete distribution (atom types)
            logits_clamped = output.logits_atoms.clamp(-20, 20)
            dist_atoms = Categorical(logits=logits_clamped)
            
            corruptions = getattr(self.diffusion_module.corruption, "corruptions", {})
            d3pm_corr = corruptions.get("atomic_numbers")
            
            if d3pm_corr is not None:
                atoms_0_idx = d3pm_corr._to_zero_based(step_actions_atoms)
            else:
                atoms_0_idx = step_actions_atoms
                
            lp_atoms = dist_atoms.log_prob(atoms_0_idx)
            log_prob_disc = scatter(lp_atoms, batch_idx_pos, dim=0, reduce="sum").clamp(
                min=torch.log(torch.tensor(config.prob_clamp_min, device=lp_atoms.device)),
                max=-1e-6,  # log(1 - 1e-6); discrete log-probs are always ≤ 0
            )
            
            entropy_atoms = dist_atoms.entropy()
            entropy = scatter(entropy_atoms, batch_idx_pos, dim=0, reduce="mean")
            
            log_probs_cont.append(log_prob_cont)
            log_probs_disc.append(log_prob_disc)
            entropies.append(entropy)
        
        return (
            torch.stack(log_probs_cont),
            torch.stack(log_probs_disc),
            torch.stack(entropies),
        )
    
    @torch.no_grad()
    def sample_action(
        self,
        x: ChemGraph,
        t: torch.Tensor,
        dt: torch.Tensor,
        config: DDPOConfig,
    ) -> tuple[ChemGraph, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ChemGraph]:
        
        """
        Sample an action (denoising step) from the policy.
        
        Returns:
            next_state: Denoised state
            action_pos: Sampled position noise
            action_cell: Sampled cell noise
            action_atoms: Sampled atom types
            log_prob_cont: Log probability of continuous actions
            log_prob_disc: Log probability of discrete actions
        """
        output = self.forward(x, t, dt)
        
        # Sample from continuous distributions using explicit reparameterization for symmetric constraints
        eps_pos = torch.randn_like(output.loc_pos)
        action_pos = output.loc_pos + output.scale_pos * eps_pos
        
        eps_cell = torch.randn_like(output.loc_cell)
        corruptions = getattr(self.diffusion_module.corruption, "corruptions", {})
        sde_cell = corruptions.get("cell")
        sde_pos = corruptions.get("pos")
        
        if sde_cell is not None:
            try:
                from mattergen.common.diffusion.corruption import make_noise_symmetric_preserve_variance
                eps_cell = make_noise_symmetric_preserve_variance(eps_cell)
            except ImportError:
                pass
                
        action_cell = output.loc_cell + output.scale_cell * eps_cell
        
        # Log probs for continuous
        dist_pos = Normal(output.loc_pos, output.scale_pos)
        dist_cell = Normal(output.loc_cell, output.scale_cell)
        
        # Calculate per-crystal log probabilities (do not sum across the entire batch of 4 graphs simultaneously)
        from torch_scatter import scatter
        batch_idx_pos = x.get_batch_idx("pos")
        
        lp_pos_atoms = dist_pos.log_prob(action_pos).sum(dim=-1)
        log_prob_pos = scatter(lp_pos_atoms, batch_idx_pos, dim=0, reduce="sum")
        
        log_prob_cell = dist_cell.log_prob(action_cell).sum(dim=(1, 2))
        
        # Only clamp the lower bound: continuous log-probs can legitimately be positive
        # (probability density > 1 for tight Gaussians near t=0). Clamping the upper
        # bound to log(1-1e-6)≈0 would zero-out the PPO signal for low-noise steps.
        log_prob_cont = (log_prob_pos + log_prob_cell).clamp(
            min=torch.log(torch.tensor(config.prob_clamp_min, device=log_prob_pos.device)),
        )
        
        # Sample from discrete distribution
        logits_clamped = output.logits_atoms.clamp(-20, 20)
        dist_atoms = Categorical(logits=logits_clamped)
        
        action_atoms_0nd = dist_atoms.sample()
        corruptions = getattr(self.diffusion_module.corruption, "corruptions", {})
        d3pm_corr = corruptions.get("atomic_numbers")
        
        if d3pm_corr is not None:
            action_atoms = d3pm_corr._to_non_zero_based(action_atoms_0nd)
        else:
            action_atoms = action_atoms_0nd
        
        lp_atoms = dist_atoms.log_prob(action_atoms_0nd)
        log_prob_disc = scatter(lp_atoms, batch_idx_pos, dim=0, reduce="sum").clamp(
            min=torch.log(torch.tensor(config.prob_clamp_min, device=action_atoms.device)),
            max=-1e-6,  # log(1 - 1e-6); discrete log-probs are always ≤ 0
        )
        
        # Wrap pos correctly so the next state lies in [0, 1] without breaking gradient graphs
        next_pos = action_pos
        if sde_pos is not None and hasattr(sde_pos, "wrap"):
            next_pos = sde_pos.wrap(action_pos)
            
        # Construct next state
        next_state = x.replace(
            pos=next_pos,
            cell=action_cell,
            atomic_numbers=action_atoms,
        )
        
        # Construct mean state (denoised representation without injected gaussian noise)
        mean_pos = output.loc_pos
        if sde_pos is not None and hasattr(sde_pos, "wrap"):
            mean_pos = sde_pos.wrap(mean_pos)
            
        mean_atoms_0nd = torch.argmax(logits_clamped, dim=-1)
        if d3pm_corr is not None:
            mean_atoms = d3pm_corr._to_non_zero_based(mean_atoms_0nd)
        else:
            mean_atoms = mean_atoms_0nd
            
        mean_state = x.replace(
            pos=mean_pos,
            cell=output.loc_cell,
            atomic_numbers=mean_atoms,
        )
        
        return next_state, action_pos, action_cell, action_atoms, log_prob_cont, log_prob_disc, mean_state


# =============================================================================
# ValueNetwork: Critic for Advantage Estimation
# =============================================================================

class ValueNetwork(nn.Module):
    """
    Critic network that predicts state values V(s_t).
    
    Can be initialized from a pretrained bulk modulus classifier checkpoint
    to leverage learned representations.
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        mlp_hidden_dim: int = 256,
        gemnet_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.noise_level_encoding = NoiseLevelEncoding(hidden_dim)
        
        # GemNet backbone (same architecture as BulkModulusTimeClassifier)
        atom_embedding = AtomEmbedding(emb_size=hidden_dim, with_mask_type=True)
        self.gemnet = GemNetT(
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
            **(gemnet_kwargs or {}),
        )
        
        # Value head: outputs single scalar
        head_in = hidden_dim + hidden_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(mlp_hidden_dim, 1),
        )
        
    def forward(self, x: ChemGraph, t: torch.Tensor) -> torch.Tensor:
        """
        Predict state value.
        
        Args:
            x: Crystal state at timestep t
            t: Diffusion timesteps (batch_size,)
            
        Returns:
            values: State values (batch_size,)
        """
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
        
        # Pool node embeddings to crystal-level
        node_embeddings = gemnet_out.node_embeddings
        batch_idx = x.get_batch_idx("pos")
        pooled = scatter_mean(node_embeddings, batch_idx, dim=0)
        
        # Concatenate with time embedding and predict value
        head_in = torch.cat([pooled, t_emb], dim=-1)
        value = self.head(head_in)
        
        return value.squeeze(-1)
    
    @classmethod
    def from_classifier_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | str = "cpu",
    ) -> "ValueNetwork":
        """
        Initialize from a BulkModulusTimeClassifier checkpoint.
        
        Copies GemNet weights and adapts the head for value prediction.
        """
        checkpoint_path = Path(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        # Get config from checkpoint
        if "config" in ckpt and "model_kwargs" in ckpt["config"]:
            config = ckpt["config"]["model_kwargs"]
            hidden_dim = config.get("hidden_dim", 512)
            mlp_hidden_dim = config.get("mlp_hidden_dim", 256)
        else:
            hidden_dim = 512
            mlp_hidden_dim = 256
        
        # Create network
        model = cls(hidden_dim=hidden_dim, mlp_hidden_dim=mlp_hidden_dim)
        
        # Load compatible weights
        state_dict = ckpt.get("model_state_dict", ckpt)
        
        # Filter to only load GemNet and noise_level_encoding weights.
        # The classifier applies LoRA to every Linear in GemNet, which renames
        # "gemnet.xxx.weight" → "gemnet.xxx.base_layer.weight". ValueNetwork has
        # a plain GemNet (no LoRA), so we strip the ".base_layer" infix and skip
        # the LoRA-specific keys (lora_A / lora_B) entirely.
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if not (key.startswith("gemnet.") or key.startswith("noise_level_encoding.")):
                continue
            if ".lora_A." in key or ".lora_B." in key:
                continue  # LoRA adapter weights — no matching layer in ValueNetwork
            remapped = key.replace(".base_layer.", ".")
            filtered_state_dict[remapped] = value

        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        print(f"ValueNetwork loaded from {checkpoint_path}")
        print(f"  Missing keys (should be ~6 head params only): {len(missing)}")
        print(f"  Loaded keys: {len(filtered_state_dict)}")
        
        return model.to(device)


# =============================================================================
# Helpers
# =============================================================================

def _cat_chemgraph_states(states: list) -> "ChemGraph":
    """Concatenate a list of batched ChemGraphs into a single mega-batch.

    Each element is already a batched ChemGraph (e.g. 128 crystals). The result
    has len(states) * batch_size crystals with corrected batch indices so GemNet
    treats every crystal independently — identical to what a DataLoader would
    produce for a single larger batch.
    """
    offset = 0
    all_pos, all_atomic, all_cells, all_num_atoms, all_batch = [], [], [], [], []

    for state in states:
        batch_idx = state.batch          # (total_atoms,)  per-atom graph index
        n_graphs = state["cell"].shape[0]

        all_pos.append(state["pos"])
        all_atomic.append(state["atomic_numbers"])
        all_cells.append(state["cell"])
        all_num_atoms.append(state["num_atoms"])
        all_batch.append(batch_idx + offset)
        offset += n_graphs

    return states[0].replace(
        pos=torch.cat(all_pos, dim=0),
        atomic_numbers=torch.cat(all_atomic, dim=0),
        cell=torch.cat(all_cells, dim=0),
        num_atoms=torch.cat(all_num_atoms, dim=0),
        batch=torch.cat(all_batch, dim=0),
    )


# =============================================================================
# DDPOTrainer: Main Training Loop
# =============================================================================

class DDPOTrainer:
    """
    Denoising Diffusion Policy Optimization trainer with decoupled PPO.
    
    Key stability features:
    - Separate PPO ratios for continuous/discrete actions
    - Stricter clipping for discrete actions
    - KL divergence anchor against frozen reference model
    - Entropy regularization for discrete exploration
    - NaN protection via probability clamping
    """
    
    def __init__(
        self,
        actor: MatterGenActor,
        critic: ValueNetwork,
        reward_fn: Callable[[ChemGraph], float],
        config: DDPOConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.reward_fn = reward_fn
        self.config = config or DDPOConfig()
        self.device = torch.device(device)
        
        # Frozen reference model for KL anchor
        self.ref_actor = copy.deepcopy(actor)
        self.ref_actor.eval()
        for param in self.ref_actor.parameters():
            param.requires_grad = False
        self.ref_actor = self.ref_actor.to(device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=self.config.actor_lr,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.config.critic_lr,
        )
        
        # Metrics tracking
        self.metrics_history: list[dict] = []

        # Mixed-precision scaler for PPO backward passes
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        # torch.compile: fuses GemNet kernels, ~10-30% faster per forward pass.
        # dynamic=True avoids recompilation when atom counts change between crystals.
        if torch.cuda.is_available():
            self.actor = torch.compile(self.actor, dynamic=True)
            self.critic = torch.compile(self.critic, dynamic=True)
            self.ref_actor = torch.compile(self.ref_actor, dynamic=True)

    @torch.no_grad()
    def collect_trajectories(
        self,
        sampler,
        condition_loader,
        num_diffusion_batches: int = 1,
    ) -> list[Trajectory]:
        """
        Collect trajectories by running the diffusion sampling process using the Actor policy.
        
        Args:
            sampler: PredictorCorrector sampler
            condition_loader: Data loader for conditioning
            num_trajectories: Number of trajectories to collect
            
        Returns:
            List of Trajectory objects with steps and terminal rewards
        """
        self.actor.eval()
        
        # We don't want to track gradients during rollout to save memory.
        # Trajectories are fully detached.
        import time as _time
        trajectories = []
        _t_diffusion = 0.0
        _t_reward = 0.0
        with torch.no_grad():
            for _ in range(num_diffusion_batches):
                conditioning_data, mask = next(iter(condition_loader))
                conditioning_data = conditioning_data.to(self.device)

                try:
                    # Get prior initial state based on conditioning data
                    from mattergen.diffusion.sampling.pc_sampler import _sample_prior
                    state = _sample_prior(sampler._multi_corruption, conditioning_data, mask=None)

                    timesteps = torch.linspace(sampler._max_t, sampler._eps_t, sampler.N, device=self.device)
                    dt_scalar = -torch.tensor((sampler._max_t - sampler._eps_t) / (sampler.N - 1)).to(self.device)

                    steps = []
                    _t0 = _time.time()
                    # Rollout from T to 0
                    for i in range(sampler.N):
                        t_val = timesteps[i]
                        # Create states/tensors on CPU for the Actor
                        t = torch.full((state.get_batch_size(),), t_val, device=self.device)
                        dt = torch.full((state.get_batch_size(),), dt_scalar, device=self.device)
                        
                        # Apply Langevin Correctors exactly like pc_sampler does to stabilize generations
                        if hasattr(sampler, "_correctors") and sampler._correctors:
                            for _ in range(getattr(sampler, "_n_steps_corrector", 1)):
                                score_val = sampler._score_fn(state, t)
                                
                                # Process each corrector using the exact same mapping function as pc_sampler
                                from mattergen.diffusion.corruption.multi_corruption import apply
                                fns = {k: corrector.step_given_score for k, corrector in sampler._correctors.items()}
                                
                                samples_means = apply(
                                    fns=fns,
                                    broadcast={"t": t, "dt": dt},
                                    x=state,
                                    score=score_val,
                                    batch_idx=sampler._multi_corruption._get_batch_indices(state),
                                )
                                
                                # Samples_means returns a dictionary of Tuple[sample, mean]. 
                                # We update the physical state with the new stochastic sample before the next step.
                                state_updates = {k: v[0] for k, v in samples_means.items()}
                                state = state.replace(**state_updates)
                                    
                        # Record the pre-action state so (state, timestep) is a matched pair.
                        # The critic predicts V(x_t, t); storing next_state here would give
                        # V(x_{t-1}, t) — a less-noisy structure at the wrong noise level.
                        prev_state = state.clone()

                        # Sample action from the current actor policy
                        next_state, action_pos, action_cell, action_atoms, lp_cont, lp_disc, mean_state = self.actor.sample_action(
                            state, t, dt, self.config
                        )

                        # Advance the rollout state
                        state = next_state

                        # Record the step
                        steps.append(TrajectoryStep(
                            state=prev_state,
                            timestep=t.clone(),
                            action_pos=action_pos.clone(),
                            action_cell=action_cell.clone(),
                            action_atoms=action_atoms.clone(),
                            log_prob_cont=lp_cont.clone(),
                            log_prob_disc=lp_disc.clone(),
                            dt=dt.clone(),
                        ))
                        
                    _t_diffusion += _time.time() - _t0

                    # Final sample x_0
                    sample = mean_state

                    # Compute terminal reward using x_0 and t=0 inside reward_fn
                    _t1 = _time.time()
                    reward = self.reward_fn(sample)
                    _t_reward += _time.time() - _t1

                    # Create trajectory
                    traj = Trajectory(
                        steps=steps,
                        final_sample=sample,
                        reward=reward,
                    )
                    trajectories.append(traj)

                except Exception as e:
                    print(f"Trajectory collection failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        print(f"  [timing] diffusion: {_t_diffusion:.1f}s | reward: {_t_reward:.1f}s")
        return trajectories
    
    def compute_advantages(
        self,
        trajectories: list[Trajectory],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        
        """
        Compute advantages: A_t = R(x_0) - V(s_t)
        
        For terminal reward RL: reward only at final step.
        """
        advantages_list = []
        returns_list = []
        
        for traj in trajectories:
            if traj.final_sample is None:
                continue
                
            # If reward is already a tensor (from neural net), this keeps it.
            # If it's a float, it becomes a 0-D tensor.
            # In our case it will be a (batch_size,) tensor.
            reward = torch.as_tensor(traj.reward, dtype=torch.float32, device=self.device)
            
            # For each step in trajectory, advantage = R - V(s)
            # Since reward is terminal, all steps share the same return
            for step in traj.steps:
                value = self.critic(step.state, step.timestep)
                advantage = reward - value.detach()
                advantages_list.append(advantage)
                returns_list.append(reward)
        
        return advantages_list, returns_list
    
    def compute_decoupled_ppo_loss(
        self,
        states: list[ChemGraph],
        timesteps: torch.Tensor,
        dts: torch.Tensor,
        actions_pos: torch.Tensor,
        actions_cell: torch.Tensor,
        actions_atoms: torch.Tensor,
        old_log_probs_cont: torch.Tensor,
        old_log_probs_disc: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the decoupled PPO loss with separate handling for continuous/discrete.
        
        Key components:
        1. Separate PPO ratios r_cont and r_disc (NOT multiplied together)
        2. Separate clipping: ε=0.2 for continuous, ε=0.1 for discrete
        3. KL divergence anchor against frozen reference model
        """
        cfg = self.config
        
        # Ensure advantages and returns are passed directly. 
        # Previously they were incorrectly averaged with .mean(dim=-1), obliterating per-crystal optimization.
        # Advantages are [batch_size, num_crystals_in_batch] and should remain that way.
        
        # Get new log probabilities and entropy
        new_log_probs_cont, new_log_probs_disc, entropy_disc = self.actor.evaluate_actions(
            states, timesteps, dts, actions_pos, actions_cell, actions_atoms, cfg
        )
        
        # ==== Separate PPO Ratios (THE KEY INSIGHT) ====
        # Do NOT multiply these together - that causes gradient instability
        ratio_cont = torch.exp(new_log_probs_cont - old_log_probs_cont)
        ratio_disc = torch.exp(new_log_probs_disc - old_log_probs_disc)
        
        # ==== Continuous Loss (standard PPO clip ε=0.2) ====
        surr1_cont = ratio_cont * advantages
        surr2_cont = torch.clamp(
            ratio_cont,
            1.0 - cfg.clip_eps_cont,
            1.0 + cfg.clip_eps_cont,
        ) * advantages
        loss_cont = -torch.min(surr1_cont, surr2_cont).mean()
        
        # ==== Discrete Loss (stricter PPO clip ε=0.1) ====
        surr1_disc = ratio_disc * advantages
        surr2_disc = torch.clamp(
            ratio_disc,
            1.0 - cfg.clip_eps_disc,
            1.0 + cfg.clip_eps_disc,
        ) * advantages
        loss_disc = -torch.min(surr1_disc, surr2_disc).mean()
        
        # ==== KL Divergence Anchor (prevents chemistry collapse) ====
        # Compare current discrete policy to frozen reference
        with torch.no_grad():
            ref_log_probs_cont, ref_log_probs_disc, _ = self.ref_actor.evaluate_actions(
                states, timesteps, dts, actions_pos, actions_cell, actions_atoms, cfg
            )
        
        # KL(ref || new) for discrete actions
        # Approximation: mean of (ref_log_prob - new_log_prob)
        kl_disc = (ref_log_probs_disc - new_log_probs_disc).mean()
        kl_disc = kl_disc.clamp(min=0)  # KL should be non-negative
        
        # ==== Entropy Bonus (exploration) ====
        entropy_bonus = -cfg.entropy_coeff * entropy_disc.mean()
        
        # ==== Value Loss ====
        # Concatenate all minibatch states into a single mega-batch so GemNet
        # runs once instead of ppo_mb_size times sequentially.
        mega_state = _cat_chemgraph_states(states)
        mega_t = timesteps.reshape(-1)  # (ppo_mb_size * batch_size,)
        values_flat = self.critic(mega_state, mega_t)
        values = values_flat.view(len(states), -1)  # (ppo_mb_size, batch_size)
        value_loss = F.mse_loss(values, returns)
        
        # ==== Total Loss ====
        total_loss = (
            loss_cont +
            loss_disc +
            cfg.kl_coeff * kl_disc +
            entropy_bonus +
            cfg.value_coeff * value_loss
        )
        
        return {
            "total_loss": total_loss,
            "loss_cont": loss_cont,
            "loss_disc": loss_disc,
            "kl_disc": kl_disc,
            "entropy": entropy_disc.mean(),
            "value_loss": value_loss,
            "ratio_cont_mean": ratio_cont.mean(),
            "ratio_disc_mean": ratio_disc.mean(),
        }
    
    def update_step(
        self,
        trajectories: list[Trajectory],
    ) -> dict[str, float]:
        """
        Run K epochs of PPO updates on collected trajectories.
        """
        self.actor.train()
        self.critic.train()
        
        # Extract states — fully detached, no grad retention needed.
        states = [step.state for traj in trajectories for step in traj.steps]
        timesteps = torch.stack([step.timestep for traj in trajectories for step in traj.steps]).detach()
        dts = torch.stack([step.dt for traj in trajectories for step in traj.steps]).detach()
        actions_pos = torch.stack([step.action_pos for traj in trajectories for step in traj.steps]).detach()
        actions_cell = torch.stack([step.action_cell for traj in trajectories for step in traj.steps]).detach()
        actions_atoms = torch.stack([step.action_atoms for traj in trajectories for step in traj.steps]).detach()
        old_log_probs_cont = torch.stack([step.log_prob_cont for traj in trajectories for step in traj.steps]).detach()
        old_log_probs_disc = torch.stack([step.log_prob_disc for traj in trajectories for step in traj.steps]).detach()

        if len(states) == 0:
            return {"error": "No valid trajectory steps"}

        # Build returns WITHOUT critic: terminal-reward RL means return = final
        # reward for every step in the trajectory. No GemNet call needed here.
        returns_list = []
        for traj in trajectories:
            if traj.final_sample is not None:
                reward = torch.as_tensor(traj.reward, dtype=torch.float32, device=self.device)
                for _ in traj.steps:
                    returns_list.append(reward)
        if not returns_list:
            return {"error": "No valid trajectory steps"}
        returns = torch.stack(returns_list).to(self.device).detach()  # (num_steps, batch_size)

        num_samples = len(states)

        # Subsample BEFORE computing advantages so the critic only runs on the
        # steps we actually use — 5x fewer critic calls vs. sampling after.
        subsample_frac = self.config.timestep_subsample_frac
        subsample_n = max(self.config.ppo_mb_size, int(num_samples * subsample_frac))
        if subsample_n < num_samples:
            sub_idx = torch.randperm(num_samples, device=self.device)[:subsample_n]
            sub_idx_list = sub_idx.tolist()
            states = [states[j] for j in sub_idx_list]
            timesteps = timesteps[sub_idx]
            dts = dts[sub_idx]
            actions_pos = actions_pos[sub_idx]
            actions_cell = actions_cell[sub_idx]
            actions_atoms = actions_atoms[sub_idx]
            old_log_probs_cont = old_log_probs_cont[sub_idx]
            old_log_probs_disc = old_log_probs_disc[sub_idx]
            returns = returns[sub_idx]
            num_samples = len(states)

        # Compute advantages with a SINGLE batched critic call (no_grad, so no
        # activation storage — concatenating all subsampled states is cheap).
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                mega_state = _cat_chemgraph_states(states)
                mega_t = timesteps.reshape(-1)
                values_flat = self.critic(mega_state, mega_t)
            values = values_flat.view(num_samples, -1).detach()  # (subsample_n, batch_size)

        advantages = (returns - values).detach()

        # Normalize advantages (variance reduction)
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        minibatch_size = self.config.ppo_mb_size

        # K epochs of PPO updates
        epoch_metrics = []
        for _ in range(self.config.ppo_epochs):
            indices = torch.randperm(num_samples)
            batch_losses = []

            for start_idx in range(0, num_samples, minibatch_size):
                end_idx = min(start_idx + minibatch_size, num_samples)
                mb_indices = indices[start_idx:end_idx].tolist()

                mb_states = [states[j] for j in mb_indices]
                mb_timesteps = timesteps[mb_indices]
                mb_dts = dts[mb_indices]
                mb_actions_pos = actions_pos[mb_indices]
                mb_actions_cell = actions_cell[mb_indices]
                mb_actions_atoms = actions_atoms[mb_indices]
                mb_old_log_probs_cont = old_log_probs_cont[mb_indices]
                mb_old_log_probs_disc = old_log_probs_disc[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # AMP: forward pass in fp16 where safe, backward handled by scaler
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    losses = self.compute_decoupled_ppo_loss(
                        states=mb_states,
                        timesteps=mb_timesteps,
                        dts=mb_dts,
                        actions_pos=mb_actions_pos,
                        actions_cell=mb_actions_cell,
                        actions_atoms=mb_actions_atoms,
                        old_log_probs_cont=mb_old_log_probs_cont,
                        old_log_probs_disc=mb_old_log_probs_disc,
                        advantages=mb_advantages,
                        returns=mb_returns,
                    )

                # Check for NaN/Inf (scaler will also skip steps with Inf grads)
                if torch.isnan(losses["total_loss"]) or torch.isinf(losses["total_loss"]):
                    print("Warning: NaN/Inf loss detected, skipping update")
                    continue

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss = self.config.value_coeff * losses["value_loss"]
                self.scaler.scale(critic_loss).backward()
                self.scaler.unscale_(self.critic_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.config.grad_clip_norm,
                )
                self.scaler.step(self.critic_optimizer)

                # Update actor (LoRA weights only)
                self.actor_optimizer.zero_grad()
                actor_loss = (losses["loss_cont"] + losses["loss_disc"] +
                              self.config.kl_coeff * losses["kl_disc"] +
                              (-self.config.entropy_coeff * losses["entropy"]))
                self.scaler.scale(actor_loss).backward()
                self.scaler.unscale_(self.actor_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.config.grad_clip_norm,
                )
                self.scaler.step(self.actor_optimizer)

                # One scaler update per minibatch (covers both optimizer steps)
                self.scaler.update()

                batch_losses.append({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()})
                
            # Average minibatch losses for this epoch
            if batch_losses:
                avg_epoch_loss = {k: sum(m[k] for m in batch_losses) / len(batch_losses) for k in batch_losses[0].keys()}
                epoch_metrics.append(avg_epoch_loss)
        
        # Average metrics across epochs
        if epoch_metrics:
            avg_metrics = {
                k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics)
                for k in epoch_metrics[0].keys()
            }
        else:
            avg_metrics = {"error": "All epochs skipped due to NaN"}
        
        return avg_metrics
    
    def train(
        self,
        sampler,
        condition_loader,
        num_epochs: int = 100,
        num_diffusion_batches: int = 1,
        save_path: Path | None = None,
        save_every: int = 10,
    ) -> list[dict]:
        # Main training loop.
        #
        # Args:
        #     sampler: PredictorCorrector sampler for trajectory collection
        #     condition_loader: Data loader for conditioning data
        #     num_epochs: Number of training epochs
        #     num_diffusion_batches: Number of independent diffusion generation passes
        #     save_path: Directory to save checkpoints
        #     save_every: Save checkpoint every N epochs
        #
        # Returns:
        #     List of metrics for each epoch
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
        
        best_reward = float("-inf")
        global_best_sample_reward = float("-inf")
        
        import time
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            # Collect trajectories
            trajectories = self.collect_trajectories(
                sampler=sampler,
                condition_loader=condition_loader,
                num_diffusion_batches=num_diffusion_batches,
            )
            
            if len(trajectories) == 0:
                print(f"Epoch {epoch}: No valid trajectories collected")
                continue
            
            # Compute mean reward and find the best sample
            rewards = []
            best_sample = None
            best_sample_reward = float("-inf")
            best_sample_idx = 0
            for t in trajectories:
                if t.final_sample is not None:
                    # Handle reward regardless if it's a tensor or float
                    if isinstance(t.reward, torch.Tensor):
                        # Track per-crystal rewards for accurate mean and best
                        crystal_rewards = t.reward.tolist()
                        rewards.extend(crystal_rewards)
                        best_idx = int(t.reward.argmax().item())
                        rew_val = t.reward[best_idx].item()
                    else:
                        rewards.append(float(t.reward))
                        best_idx = 0
                        rew_val = float(t.reward)
                    if rew_val > best_sample_reward:
                        best_sample_reward = rew_val
                        best_sample = t.final_sample
                        best_sample_idx = best_idx
                    
            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            
            global_best_sample_reward = max(global_best_sample_reward, best_sample_reward)

            # PPO update
            _t_ppo = time.time()
            update_metrics = self.update_step(trajectories)
            _ppo_time = time.time() - _t_ppo

            epoch_time = time.time() - epoch_start_time
            
            # Log metrics
            epoch_metrics = {
                "epoch": epoch,
                "epoch_time_seconds": epoch_time,
                "mean_reward": mean_reward,
                "best_epoch_reward": best_sample_reward,
                "global_best_reward": global_best_sample_reward,
                "num_trajectories": len(trajectories),
                **update_metrics,
            }
            self.metrics_history.append(epoch_metrics)
            
            recent = self.metrics_history[-10:]
            rolling_mean = sum(m["mean_reward"] for m in recent) / len(recent)
            # Plateau: global best hasn't improved in the last 30 epochs
            window = self.metrics_history[-30:]
            plateau_flag = (
                len(window) == 30
                and window[-1]["global_best_reward"] <= window[0]["global_best_reward"]
            )
            print(
                f"Epoch {epoch:4d} | "
                f"Reward: {mean_reward:7.2f} (avg10: {rolling_mean:7.2f}) | "
                f"Best: {global_best_sample_reward:7.2f} | "
                f"KL: {update_metrics.get('kl_disc', 0):.4f} | "
                f"Time: {epoch_time:.0f}s (ppo: {_ppo_time:.0f}s)"
                + (" [PLATEAU]" if plateau_flag else "")
            )
            
            # Save metrics to disk per epoch for plotting
            if save_path:
                import json
                import csv
                with open(save_path / "metrics.json", "w") as f:
                    json.dump(self.metrics_history, f, indent=2)
                
                csv_path = save_path / "metrics.csv"
                file_exists = csv_path.exists()
                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=epoch_metrics.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(epoch_metrics)
            
            # Save best model
            if mean_reward > best_reward and save_path:
                best_reward = mean_reward
                self.save_checkpoint(save_path / "best_model.pt", epoch, epoch_metrics)
                
            # Extract and save the best generated material for this epoch
            if best_sample is not None and save_path:
                try:
                    from ase import Atoms
                    import numpy as np
                    
                    # Convert best_sample (ChemGraph) to ASE Atoms
                    # Handling the first structure in the batch if best_sample is batched
                    with torch.no_grad():
                        pos = best_sample["pos"][best_sample.get_batch_idx("pos") == best_sample_idx].detach().cpu().numpy()
                        cell = best_sample["cell"][best_sample_idx].detach().cpu().numpy()
                        atomic_numbers = best_sample["atomic_numbers"][best_sample.get_batch_idx("pos") == best_sample_idx].detach().cpu().numpy()
                        
                        # MatterGen outputs fractional coordinates, convert to Cartesian
                        positions_cart = pos @ cell
                        
                        best_atoms = Atoms(
                            numbers=atomic_numbers.astype(int),
                            positions=positions_cart,
                            cell=cell,
                            pbc=True,
                        )
                        
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            from ase.io import write
                            cif_path = save_path / f"best_material_epoch_{epoch}.cif"
                            write(str(cif_path), best_atoms)
                            print(f"Saved best material (Reward: {best_sample_reward:.2f}) to {cif_path.name}")
                            
                except Exception as e:
                    print(f"Failed to save best material: {e}")
            
            # Periodic save
            if save_path and epoch % save_every == 0 and epoch > 0:
                self.save_checkpoint(
                    save_path / f"checkpoint_epoch_{epoch}.pt",
                    epoch,
                    epoch_metrics,
                )
        
        # Final save
        if save_path:
            self.save_checkpoint(save_path / "final_model.pt", num_epochs, {})
        
        return self.metrics_history
    
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: dict,
    ) -> None:
        # Save model checkpoint.
        torch.save({
            "epoch": epoch,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "config": self.config.__dict__,
            "metrics": metrics,
        }, path)
        
    def load_checkpoint(self, path: Path) -> int:
        # Load model checkpoint. Returns the epoch number.
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer_state_dict"])
        return ckpt["epoch"]
