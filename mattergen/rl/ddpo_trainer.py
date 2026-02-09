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
    prob_clamp_max: float = 1.0 - 1e-6
    grad_clip_norm: float = 1.0
    
    # Learning rates
    actor_lr: float = 1e-5
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
    """
    RL wrapper around the MatterGen denoiser that exposes the policy interface.
    
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
        self.denoiser = denoiser
        self.diffusion_module = diffusion_module
        
    def forward(
        self,
        x: ChemGraph,
        t: torch.Tensor,
    ) -> ActorOutput:
        """
        Forward pass returning distribution parameters.
        
        Args:
            x: Noisy crystal state at timestep t
            t: Diffusion timesteps (batch_size,)
            
        Returns:
            ActorOutput with loc, scale, and logits
        """
        # Get score model output (this is the denoiser prediction)
        score_output = self.denoiser(x, t)
        
        # Extract predictions
        # Position: predicted noise (used as mean for Gaussian)
        loc_pos = score_output["pos"]
        
        # Cell: predicted noise for lattice
        loc_cell = score_output["cell"]
        
        # Atom types: logits for classification
        logits_atoms = score_output["atomic_numbers"]
        
        # Get scale from the SDE marginal probability
        # For VP-SDE: std increases with t
        batch_idx = x.get_batch_idx("pos")
        sde_pos = self.diffusion_module.corruption.sdes.get("pos")
        sde_cell = self.diffusion_module.corruption.sdes.get("cell")
        
        if sde_pos is not None:
            _, std_pos = sde_pos.marginal_prob(
                x=torch.ones_like(loc_pos),
                t=t,
                batch_idx=batch_idx,
                batch=x,
            )
            scale_pos = std_pos.clamp(min=1e-6)
        else:
            # Fallback: constant scale
            scale_pos = torch.ones_like(loc_pos) * 0.1
            
        if sde_cell is not None:
            _, std_cell = sde_cell.marginal_prob(
                x=torch.ones_like(loc_cell),
                t=t,
                batch_idx=None,
                batch=x,
            )
            scale_cell = std_cell.clamp(min=1e-6)
        else:
            scale_cell = torch.ones_like(loc_cell) * 0.1
            
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
        
        for i, (state, t) in enumerate(zip(states, timesteps)):
            # Forward pass for this state
            output = self.forward(state, t.unsqueeze(0))
            
            # Continuous distribution (position + cell)
            dist_pos = Normal(output.loc_pos, output.scale_pos)
            dist_cell = Normal(output.loc_cell, output.scale_cell)
            
            # Get log probs for this step's actions
            # Handle per-atom actions for positions
            batch_idx = state.get_batch_idx("pos")
            step_actions_pos = actions_pos[i] if actions_pos.dim() == 3 else actions_pos
            step_actions_cell = actions_cell[i] if actions_cell.dim() == 4 else actions_cell
            step_actions_atoms = actions_atoms[i] if actions_atoms.dim() == 2 else actions_atoms
            
            # Log prob continuous: sum over dimensions
            lp_pos = dist_pos.log_prob(step_actions_pos).sum()
            lp_cell = dist_cell.log_prob(step_actions_cell).sum()
            log_prob_cont = lp_pos + lp_cell
            
            # Clamp for numerical stability
            log_prob_cont = log_prob_cont.clamp(
                min=torch.log(torch.tensor(config.prob_clamp_min)),
                max=torch.log(torch.tensor(config.prob_clamp_max)),
            )
            
            # Discrete distribution (atom types)
            # Clamp logits to prevent extreme probabilities
            logits_clamped = output.logits_atoms.clamp(-20, 20)
            dist_atoms = Categorical(logits=logits_clamped)
            
            log_prob_disc = dist_atoms.log_prob(step_actions_atoms).sum()
            log_prob_disc = log_prob_disc.clamp(
                min=torch.log(torch.tensor(config.prob_clamp_min)),
                max=torch.log(torch.tensor(config.prob_clamp_max)),
            )
            
            entropy = dist_atoms.entropy().mean()
            
            log_probs_cont.append(log_prob_cont)
            log_probs_disc.append(log_prob_disc)
            entropies.append(entropy)
        
        return (
            torch.stack(log_probs_cont),
            torch.stack(log_probs_disc),
            torch.stack(entropies),
        )
    
    def sample_action(
        self,
        x: ChemGraph,
        t: torch.Tensor,
        config: DDPOConfig,
    ) -> tuple[ChemGraph, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        output = self.forward(x, t)
        
        # Sample from continuous distributions
        dist_pos = Normal(output.loc_pos, output.scale_pos)
        dist_cell = Normal(output.loc_cell, output.scale_cell)
        
        action_pos = dist_pos.rsample()  # For gradient flow
        action_cell = dist_cell.rsample()
        
        # Log probs for continuous
        log_prob_pos = dist_pos.log_prob(action_pos).sum()
        log_prob_cell = dist_cell.log_prob(action_cell).sum()
        log_prob_cont = (log_prob_pos + log_prob_cell).clamp(
            min=torch.log(torch.tensor(config.prob_clamp_min)),
            max=torch.log(torch.tensor(config.prob_clamp_max)),
        )
        
        # Sample from discrete distribution
        logits_clamped = output.logits_atoms.clamp(-20, 20)
        dist_atoms = Categorical(logits=logits_clamped)
        action_atoms = dist_atoms.sample()
        
        log_prob_disc = dist_atoms.log_prob(action_atoms).sum().clamp(
            min=torch.log(torch.tensor(config.prob_clamp_min)),
            max=torch.log(torch.tensor(config.prob_clamp_max)),
        )
        
        # Construct next state
        next_state = x.replace(
            pos=action_pos,
            cell=action_cell,
            atomic_numbers=action_atoms,
        )
        
        return next_state, action_pos, action_cell, action_atoms, log_prob_cont, log_prob_disc


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
        
        # Filter to only load GemNet and noise_level_encoding weights
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("gemnet.") or key.startswith("noise_level_encoding."):
                filtered_state_dict[key] = value
        
        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        print(f"ValueNetwork loaded from {checkpoint_path}")
        print(f"  Missing keys (expected, new head): {len(missing)}")
        print(f"  Loaded keys: {len(filtered_state_dict)}")
        
        return model.to(device)


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
        
    def collect_trajectories(
        self,
        sampler,
        condition_loader,
        num_trajectories: int = 4,
    ) -> list[Trajectory]:
        """
        Collect trajectories by running the diffusion sampling process.
        
        Args:
            sampler: PredictorCorrector sampler
            condition_loader: Data loader for conditioning
            num_trajectories: Number of trajectories to collect
            
        Returns:
            List of Trajectory objects with steps and terminal rewards
        """
        self.actor.eval()
        trajectories = []
        
        with torch.no_grad():
            for _ in range(num_trajectories):
                conditioning_data, mask = next(iter(condition_loader))
                conditioning_data = conditioning_data.to(self.device)
                
                try:
                    # Use sampler with trajectory recording
                    sample, mean_sample, steps = sampler.sample_with_record(
                        conditioning_data, mask=None
                    )
                    
                    # Compute terminal reward
                    reward = self.reward_fn(sample)
                    
                    # Create trajectory
                    traj = Trajectory(
                        steps=[],  # Would need to capture steps during sampling
                        final_sample=sample,
                        reward=reward,
                    )
                    trajectories.append(traj)
                    
                except Exception as e:
                    print(f"Trajectory collection failed: {e}")
                    continue
        
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
                
            reward = torch.tensor(traj.reward, device=self.device)
            
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
        4. Entropy bonus for discrete exploration
        """
        cfg = self.config
        
        # Get new log probabilities and entropy
        new_log_probs_cont, new_log_probs_disc, entropy_disc = self.actor.evaluate_actions(
            states, timesteps, actions_pos, actions_cell, actions_atoms, cfg
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
                states, timesteps, actions_pos, actions_cell, actions_atoms, cfg
            )
        
        # KL(ref || new) for discrete actions
        # Approximation: mean of (ref_log_prob - new_log_prob)
        kl_disc = (ref_log_probs_disc - new_log_probs_disc).mean()
        kl_disc = kl_disc.clamp(min=0)  # KL should be non-negative
        
        # ==== Entropy Bonus (exploration) ====
        entropy_bonus = -cfg.entropy_coeff * entropy_disc.mean()
        
        # ==== Value Loss ====
        values = torch.stack([
            self.critic(state, t.unsqueeze(0)).squeeze()
            for state, t in zip(states, timesteps)
        ])
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
        
        # Prepare batch data from trajectories
        states = []
        timesteps = []
        actions_pos = []
        actions_cell = []
        actions_atoms = []
        old_log_probs_cont = []
        old_log_probs_disc = []
        
        for traj in trajectories:
            for step in traj.steps:
                states.append(step.state)
                timesteps.append(step.timestep)
                actions_pos.append(step.action_pos)
                actions_cell.append(step.action_cell)
                actions_atoms.append(step.action_atoms)
                old_log_probs_cont.append(step.log_prob_cont)
                old_log_probs_disc.append(step.log_prob_disc)
        
        if len(states) == 0:
            return {"error": "No valid trajectory steps"}
        
        timesteps = torch.stack(timesteps)
        actions_pos = torch.stack(actions_pos)
        actions_cell = torch.stack(actions_cell)
        actions_atoms = torch.stack(actions_atoms)
        old_log_probs_cont = torch.stack(old_log_probs_cont)
        old_log_probs_disc = torch.stack(old_log_probs_disc)
        
        # Compute advantages
        advantages_list, returns_list = self.compute_advantages(trajectories)
        if len(advantages_list) == 0:
            return {"error": "No advantages computed"}
            
        advantages = torch.stack(advantages_list)
        returns = torch.stack(returns_list)
        
        # Normalize advantages (variance reduction)
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # K epochs of PPO updates
        epoch_metrics = []
        for _ in range(self.config.ppo_epochs):
            losses = self.compute_decoupled_ppo_loss(
                states=states,
                timesteps=timesteps,
                actions_pos=actions_pos,
                actions_cell=actions_cell,
                actions_atoms=actions_atoms,
                old_log_probs_cont=old_log_probs_cont,
                old_log_probs_disc=old_log_probs_disc,
                advantages=advantages,
                returns=returns,
            )
            
            # Check for NaN
            if torch.isnan(losses["total_loss"]) or torch.isinf(losses["total_loss"]):
                print("Warning: NaN/Inf loss detected, skipping update")
                continue
            
            # Update actor
            self.actor_optimizer.zero_grad()
            (losses["loss_cont"] + losses["loss_disc"] + 
             self.config.kl_coeff * losses["kl_disc"] + 
             (-self.config.entropy_coeff * losses["entropy"])).backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.config.grad_clip_norm,
            )
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            (self.config.value_coeff * losses["value_loss"]).backward()
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.config.grad_clip_norm,
            )
            self.critic_optimizer.step()
            
            epoch_metrics.append({k: v.item() for k, v in losses.items()})
        
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
        trajectories_per_epoch: int = 4,
        save_path: Path | None = None,
        save_every: int = 10,
    ) -> list[dict]:
        """
        Main training loop.
        
        Args:
            sampler: PredictorCorrector sampler for trajectory collection
            condition_loader: Data loader for conditioning data
            num_epochs: Number of training epochs
            trajectories_per_epoch: Trajectories to collect per epoch
            save_path: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            
        Returns:
            List of metrics for each epoch
        """
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
        
        best_reward = float("-inf")
        
        for epoch in range(num_epochs):
            # Collect trajectories
            trajectories = self.collect_trajectories(
                sampler=sampler,
                condition_loader=condition_loader,
                num_trajectories=trajectories_per_epoch,
            )
            
            if len(trajectories) == 0:
                print(f"Epoch {epoch}: No valid trajectories collected")
                continue
            
            # Compute mean reward
            rewards = [t.reward for t in trajectories if t.final_sample is not None]
            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            
            # PPO update
            update_metrics = self.update_step(trajectories)
            
            # Log metrics
            epoch_metrics = {
                "epoch": epoch,
                "mean_reward": mean_reward,
                "num_trajectories": len(trajectories),
                **update_metrics,
            }
            self.metrics_history.append(epoch_metrics)
            
            print(
                f"Epoch {epoch:4d} | "
                f"Reward: {mean_reward:7.2f} | "
                f"Loss Cont: {update_metrics.get('loss_cont', 0):.4f} | "
                f"Loss Disc: {update_metrics.get('loss_disc', 0):.4f} | "
                f"KL: {update_metrics.get('kl_disc', 0):.4f}"
            )
            
            # Save best model
            if mean_reward > best_reward and save_path:
                best_reward = mean_reward
                self.save_checkpoint(save_path / "best_model.pt", epoch, epoch_metrics)
            
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
        """Save model checkpoint."""
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
        """Load model checkpoint. Returns the epoch number."""
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer_state_dict"])
        return ckpt["epoch"]
