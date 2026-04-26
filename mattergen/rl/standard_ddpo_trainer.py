"""
Standard DDPO Trainer for MatterGen — ablation with a single combined PPO ratio.

Key differences from DDPOTrainer (decoupled):
- Combines cont + disc log-probs into one importance ratio
- Single clip epsilon (no separate ε_cont / ε_disc)
- Keeps the actor-critic structure and value network
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.rl.ddpo_trainer import (
    DDPOConfig,
    DDPOTrainer,
    MatterGenActor,
    Trajectory,
    ValueNetwork,
)


@dataclass
class StandardDDPOConfig:
    """Single combined ratio PPO — no decoupling of cont / disc actions."""
    clip_eps: float = 0.2
    ppo_epochs: int = 3
    kl_coeff: float = 0.1
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    prob_clamp_min: float = 1e-6
    grad_clip_norm: float = 1.0
    timestep_subsample_frac: float = 0.2
    ppo_mb_size: int = 4
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4


class StandardDDPOTrainer(DDPOTrainer):
    """
    DDPO with a single combined importance ratio for all action types.
    Inherits trajectory collection and the training loop from DDPOTrainer;
    overrides update_step with a standard (non-decoupled) PPO objective.
    """

    def __init__(
        self,
        actor: MatterGenActor,
        critic: ValueNetwork,
        reward_fn: Callable[[ChemGraph], torch.Tensor],
        config: StandardDDPOConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.reward_fn = reward_fn
        self.config = config or StandardDDPOConfig()
        self.device = torch.device(device)

        self.ref_actor = copy.deepcopy(actor)
        self.ref_actor.eval()
        for p in self.ref_actor.parameters():
            p.requires_grad = False
        self.ref_actor = self.ref_actor.to(device)

        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.config.critic_lr)

        self.metrics_history: list[dict] = []

    def compute_advantages(
        self, trajectories: list[Trajectory]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        advantages_list, returns_list = [], []
        for traj in trajectories:
            if traj.final_sample is None:
                continue
            reward = torch.as_tensor(traj.reward, dtype=torch.float32, device=self.device)
            for step in traj.steps:
                value = self.critic(step.state, step.timestep)
                advantages_list.append(reward - value.detach())
                returns_list.append(reward)
        return advantages_list, returns_list

    def _compute_standard_ppo_loss(
        self,
        states: list[ChemGraph],
        timesteps: torch.Tensor,
        dts: torch.Tensor,
        actions_pos: torch.Tensor,
        actions_cell: torch.Tensor,
        actions_atoms: torch.Tensor,
        old_lp_cont: torch.Tensor,
        old_lp_disc: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        eval_cfg = DDPOConfig(prob_clamp_min=cfg.prob_clamp_min)

        new_lp_cont, new_lp_disc, entropy_disc = self.actor.evaluate_actions(
            states, timesteps, dts, actions_pos, actions_cell, actions_atoms, eval_cfg
        )

        # Single combined ratio (the key difference from decoupled)
        old_lp = old_lp_cont + old_lp_disc
        new_lp = new_lp_cont + new_lp_disc
        ratio = torch.exp(new_lp - old_lp)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL anchor on discrete actions against frozen reference
        with torch.no_grad():
            _, ref_lp_disc, _ = self.ref_actor.evaluate_actions(
                states, timesteps, dts, actions_pos, actions_cell, actions_atoms, eval_cfg
            )
        kl = (ref_lp_disc - new_lp_disc).mean().clamp(min=0)

        entropy_bonus = -cfg.entropy_coeff * entropy_disc.mean()

        # Value loss (evaluated sequentially to avoid OOM from batching GemNet)
        values_list = []
        for state, t in zip(states, timesteps):
            t_exp = t.unsqueeze(0) if t.dim() == 0 else t
            val = self.critic(state, t_exp)
            val = val.squeeze(-1) if val.dim() > 1 else val
            values_list.append(val)
        values = torch.stack(values_list)
        value_loss = F.mse_loss(values, returns)

        return {
            "policy_loss": policy_loss,
            "kl": kl,
            "entropy": entropy_disc.mean(),
            "value_loss": value_loss,
            "ratio_mean": ratio.mean(),
        }

    def update_step(self, trajectories: list[Trajectory]) -> dict[str, float]:
        self.actor.train()
        self.critic.train()

        states = [step.state for traj in trajectories for step in traj.steps]
        timesteps = torch.stack([step.timestep for traj in trajectories for step in traj.steps]).detach()
        dts = torch.stack([step.dt for traj in trajectories for step in traj.steps]).detach()
        actions_pos = torch.stack([step.action_pos for traj in trajectories for step in traj.steps]).detach()
        actions_cell = torch.stack([step.action_cell for traj in trajectories for step in traj.steps]).detach()
        actions_atoms = torch.stack([step.action_atoms for traj in trajectories for step in traj.steps]).detach()
        old_lp_cont = torch.stack([step.log_prob_cont for traj in trajectories for step in traj.steps]).detach()
        old_lp_disc = torch.stack([step.log_prob_disc for traj in trajectories for step in traj.steps]).detach()

        if not states:
            return {"error": "No valid trajectory steps"}

        with torch.no_grad():
            advantages_list, returns_list = self.compute_advantages(trajectories)
            if not advantages_list:
                return {"error": "No advantages computed"}
            advantages = torch.stack(advantages_list).to(self.device).detach()
            returns = torch.stack(returns_list).to(self.device).detach()

        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = len(states)
        subsample_n = max(self.config.ppo_mb_size, int(num_samples * self.config.timestep_subsample_frac))
        if subsample_n < num_samples:
            idx = torch.randperm(num_samples, device=self.device)[:subsample_n].tolist()
            states = [states[j] for j in idx]
            timesteps = timesteps[idx]
            dts = dts[idx]
            actions_pos = actions_pos[idx]
            actions_cell = actions_cell[idx]
            actions_atoms = actions_atoms[idx]
            old_lp_cont = old_lp_cont[idx]
            old_lp_disc = old_lp_disc[idx]
            advantages = advantages[idx]
            returns = returns[idx]
            num_samples = len(states)

        minibatch_size = self.config.ppo_mb_size
        epoch_metrics = []

        for _ in range(self.config.ppo_epochs):
            indices = torch.randperm(num_samples)
            batch_losses = []
            for start in range(0, num_samples, minibatch_size):
                mb = indices[start:start + minibatch_size].tolist()

                losses = self._compute_standard_ppo_loss(
                    [states[j] for j in mb],
                    timesteps[mb], dts[mb],
                    actions_pos[mb], actions_cell[mb], actions_atoms[mb],
                    old_lp_cont[mb], old_lp_disc[mb],
                    advantages[mb], returns[mb],
                )

                # Critic update (value_loss depends only on critic params)
                self.critic_optimizer.zero_grad()
                critic_loss = self.config.value_coeff * losses["value_loss"]
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip_norm)
                self.critic_optimizer.step()

                # Actor update (policy_loss / kl / entropy depend only on actor params)
                self.actor_optimizer.zero_grad()
                actor_loss = (
                    losses["policy_loss"]
                    + self.config.kl_coeff * losses["kl"]
                    - self.config.entropy_coeff * losses["entropy"]
                )
                if torch.isnan(actor_loss) or torch.isinf(actor_loss):
                    print("Warning: NaN/Inf actor loss, skipping")
                    continue
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
                self.actor_optimizer.step()

                batch_losses.append({
                    k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in losses.items()
                })

            if batch_losses:
                epoch_metrics.append({
                    k: sum(m[k] for m in batch_losses) / len(batch_losses)
                    for k in batch_losses[0]
                })

        if not epoch_metrics:
            return {"error": "All epochs skipped due to NaN"}
        return {k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics) for k in epoch_metrics[0]}

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict) -> None:
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
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer_state_dict"])
        return ckpt["epoch"]
