import os
os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from networks.networks import ActorCritic



class RolloutBuffer:
    """Buffer zum Sammeln von Rollout-Daten"""
    
    def __init__(self, num_envs, num_steps, num_obs, num_actions, device):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device
        
        # Buffers
        self.obs = torch.zeros((num_steps, num_envs, num_obs), device=device)
        self.actions = torch.zeros((num_steps, num_envs, num_actions), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)
        
        # GAE buffers
        self.advantages = torch.zeros((num_steps, num_envs), device=device)
        self.returns = torch.zeros((num_steps, num_envs), device=device)
        
        self.step = 0
    
    def add(self, obs, actions, rewards, dones, values, log_probs):
        """Add transition to buffer"""
        self.obs[self.step] = obs
        self.actions[self.step] = actions
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        self.log_probs[self.step] = log_probs
        self.step += 1
    
    def compute_returns_and_advantages(self, last_values, gamma=0.99, gae_lambda=0.95):
        """Compute GAE advantages and returns"""
        with torch.no_grad():
            advantages = torch.zeros_like(self.rewards)
            last_gae_lam = 0
            
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    next_values = last_values
                else:
                    next_values = self.values[t + 1]
                
                delta = self.rewards[t] + gamma * next_values * (1 - self.dones[t]) - self.values[t]
                advantages[t] = last_gae_lam = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae_lam
            
            self.advantages = advantages
            self.returns = advantages + self.values
    
    def get_batch(self, batch_size):
        """Get random mini-batch for training"""
        total_samples = self.num_steps * self.num_envs
        indices = torch.randperm(total_samples, device=self.device)[:batch_size]
        
        # Flatten buffers
        obs = self.obs.reshape(-1, self.obs.shape[-1])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        log_probs = self.log_probs.reshape(-1)
        advantages = self.advantages.reshape(-1)
        returns = self.returns.reshape(-1)
        values = self.values.reshape(-1)
        
        return (
            obs[indices],
            actions[indices],
            log_probs[indices],
            advantages[indices],
            returns[indices],
            values[indices]
        )
    
    def clear(self):
        """Reset buffer"""
        self.step = 0


class PPO:
    """Proximal Policy Optimization Algorithm"""
    
    def __init__(
        self,
        num_envs,
        num_obs,
        num_actions,
        reward_fn: callable = None,
        device='cuda',
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=1.0,
        num_steps_per_update=24,
        num_epochs=5,
        batch_size=512,
    ):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.num_actions = num_actions
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_steps_per_update = num_steps_per_update
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        # Networks
        self.actor_critic = ActorCritic(num_obs, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            num_envs, num_steps_per_update, num_obs, num_actions, self.device
        )
        
        # Training statistics
        self.total_steps = 0

        # Reward function
        self.reward_fn = reward_fn
    
    def act(self, obs, deterministic=False):
        """Select action given observation"""
        with torch.no_grad():
            action, log_prob, value = self.actor_critic.act(obs, deterministic)
        return action, log_prob, value
    
    def store_transition(self, obs, actions, rewards, dones, values, log_probs):
        """Store transition in rollout buffer"""
        self.rollout_buffer.add(obs, actions, rewards, dones, values, log_probs)
    
    def ppo_update(self, last_values):
        """
        Perform PPO update using collected rollouts
        
        Args:
            last_values: Value estimates for the last observation (for bootstrapping)
        
        Returns:
            dict: Training statistics
        """
        # Compute advantages and returns
        self.rollout_buffer.compute_returns_and_advantages(
            last_values, self.gamma, self.gae_lambda
        )
        
        # Training statistics
        policy_losses = []
        value_losses = []
        entropies = []
        clip_fractions = []
        
        # Multiple epochs of updates
        for epoch in range(self.num_epochs):
            # Get mini-batches
            num_batches = (self.num_steps_per_update * self.num_envs) // self.batch_size
            
            for _ in range(num_batches):
                batch = self.rollout_buffer.get_batch(self.batch_size)
                obs_batch, actions_batch, old_log_probs_batch, advantages_batch, returns_batch, old_values_batch = batch
                
                # Normalize advantages
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                
                # Evaluate actions under current policy
                log_probs, values, entropy = self.actor_critic.evaluate(obs_batch, actions_batch)
                values = values.squeeze(-1)
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                value_pred_clipped = old_values_batch + torch.clamp(
                    values - old_values_batch, -self.clip_epsilon, self.clip_epsilon
                )
                value_loss_unclipped = (values - returns_batch).pow(2)
                value_loss_clipped = (value_pred_clipped - returns_batch).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Statistics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                    clip_fractions.append(clip_fraction.item())
        
        # Clear buffer
        self.rollout_buffer.clear()
        
        # Return training statistics
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'clip_fraction': np.mean(clip_fractions),
        }
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        print(f"Model loaded from {path}")
