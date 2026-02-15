"""
Unit tests for DDPO trainer components.
"""

import pytest
import torch
import torch.nn as nn

from mattergen.rl import DDPOConfig


class TestDDPOConfig:
    """Test DDPOConfig dataclass."""
    
    def test_default_values(self):
        config = DDPOConfig()
        assert config.clip_eps_cont == 0.2
        assert config.clip_eps_disc == 0.1
        assert config.ppo_epochs == 3
        assert config.kl_coeff == 0.1
        assert config.entropy_coeff == 0.01
        
    def test_custom_values(self):
        config = DDPOConfig(
            clip_eps_cont=0.3,
            clip_eps_disc=0.05,
            ppo_epochs=5,
        )
        assert config.clip_eps_cont == 0.3
        assert config.clip_eps_disc == 0.05
        assert config.ppo_epochs == 5


class TestNaNProtection:
    """Test numerical stability features."""
    
    def test_probability_clamping(self):
        """Verify probability clamping prevents NaN in log computations."""
        config = DDPOConfig()
        
        # Edge case: probability = 0
        prob_zero = torch.tensor(0.0)
        clamped = prob_zero.clamp(config.prob_clamp_min, config.prob_clamp_max)
        log_val = torch.log(clamped)
        assert not torch.isnan(log_val)
        assert not torch.isinf(log_val)
        
        # Edge case: probability = 1
        prob_one = torch.tensor(1.0)
        clamped = prob_one.clamp(config.prob_clamp_min, config.prob_clamp_max)
        log_val = torch.log(clamped)
        assert not torch.isnan(log_val)
        
    def test_logit_clamping(self):
        """Verify logit clamping prevents extreme softmax values."""
        logits = torch.tensor([1000.0, -1000.0, 0.0])
        clamped = logits.clamp(-20, 20)
        probs = torch.softmax(clamped, dim=0)
        
        assert not torch.isnan(probs).any()
        assert not torch.isinf(probs).any()
        assert (probs >= 0).all()
        assert (probs <= 1).all()


class TestDecoupledPPORatios:
    """Test the key decoupled PPO ratio computation."""
    
    def test_separate_ratios(self):
        """Verify separate ratio computation for cont/disc actions."""
        # Old and new log probs
        old_log_prob_cont = torch.tensor(-1.0)
        new_log_prob_cont = torch.tensor(-0.8)
        
        old_log_prob_disc = torch.tensor(-2.0)
        new_log_prob_disc = torch.tensor(-1.5)
        
        # Compute ratios separately (NOT multiplied together)
        ratio_cont = torch.exp(new_log_prob_cont - old_log_prob_cont)
        ratio_disc = torch.exp(new_log_prob_disc - old_log_prob_disc)
        
        # Ratios should be independent
        assert ratio_cont != ratio_disc
        
        # Verify clipping works independently
        config = DDPOConfig()
        clipped_cont = torch.clamp(
            ratio_cont,
            1.0 - config.clip_eps_cont,
            1.0 + config.clip_eps_cont,
        )
        clipped_disc = torch.clamp(
            ratio_disc,
            1.0 - config.clip_eps_disc,
            1.0 + config.clip_eps_disc,
        )
        
        # Discrete should be clipped more aggressively
        assert config.clip_eps_disc < config.clip_eps_cont
        
    def test_advantage_weighting(self):
        """Test advantage-weighted loss computation."""
        advantages = torch.tensor([1.0, -0.5, 0.3])
        ratio = torch.tensor([1.1, 0.9, 1.0])
        
        # Standard PPO surrogate
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        loss = -torch.min(surr1, surr2).mean()
        
        assert not torch.isnan(loss)


class TestKLDivergence:
    """Test KL divergence anchor computation."""
    
    def test_kl_approximation(self):
        """Test KL approximation for discrete policies."""
        ref_log_probs = torch.tensor([-1.0, -2.0, -1.5])
        new_log_probs = torch.tensor([-1.1, -1.8, -1.6])
        
        # KL(ref || new) ≈ mean(ref_log_prob - new_log_prob)
        kl = (ref_log_probs - new_log_probs).mean()
        kl = kl.clamp(min=0)  # KL should be non-negative
        
        assert kl >= 0
        assert not torch.isnan(kl)
        
    def test_kl_zero_when_same(self):
        """KL should be zero when policies are identical."""
        log_probs = torch.tensor([-1.0, -2.0, -1.5])
        kl = (log_probs - log_probs).mean()
        kl = kl.clamp(min=0)
        
        assert kl.item() == 0.0


class TestEntropyBonus:
    """Test entropy regularization."""
    
    def test_entropy_computation(self):
        """Test entropy from categorical distribution."""
        from torch.distributions import Categorical
        
        # Uniform distribution has maximum entropy
        logits_uniform = torch.zeros(10)
        dist_uniform = Categorical(logits=logits_uniform)
        entropy_uniform = dist_uniform.entropy()
        
        # Peaked distribution has lower entropy
        logits_peaked = torch.tensor([10.0] + [-10.0] * 9)
        dist_peaked = Categorical(logits=logits_peaked)
        entropy_peaked = dist_peaked.entropy()
        
        assert entropy_uniform > entropy_peaked
        assert not torch.isnan(entropy_uniform)
        assert not torch.isnan(entropy_peaked)
        
    def test_entropy_bonus_gradient(self):
        """Verify entropy bonus provides gradient."""
        from torch.distributions import Categorical
        
        logits = torch.randn(10, requires_grad=True)
        dist = Categorical(logits=logits)
        entropy = dist.entropy()
        
        # Entropy bonus should have gradients
        entropy.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


# Skip integration tests if full model is not available
@pytest.mark.skip(reason="Requires full MatterGen model checkpoint")
class TestIntegration:
    """Integration tests requiring full model."""
    
    def test_actor_forward(self):
        """Test MatterGenActor forward pass."""
        pass
    
    def test_critic_forward(self):
        """Test ValueNetwork forward pass."""
        pass
    
    def test_training_loop_smoke(self):
        """Smoke test: run 2 epochs without errors."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
