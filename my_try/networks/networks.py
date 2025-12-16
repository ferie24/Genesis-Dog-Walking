import torch.nn as nn
import torch
import torch.optim as optim
from torch.distributions import Normal
import numpy as np



class ActorCritic(nn.Module):
    """Actor-Critic Netzwerk für PPO"""
    
    def __init__(self, num_obs, num_actions, hidden_dims=[256, 256, 256]):
        super().__init__()
        
        # Actor (Policy) Network
        actor_layers = []
        input_dim = num_obs
        for hidden_dim in hidden_dims:
            actor_layers.append(nn.Linear(input_dim, hidden_dim))
            actor_layers.append(nn.ELU())
            input_dim = hidden_dim
        self.actor_backbone = nn.Sequential(*actor_layers)
        self.actor_mean = nn.Linear(input_dim, num_actions)
        self.actor_logstd = nn.Parameter(torch.zeros(num_actions))
        
        # Critic (Value) Network
        critic_layers = []
        input_dim = num_obs
        for hidden_dim in hidden_dims:
            critic_layers.append(nn.Linear(input_dim, hidden_dim))
            critic_layers.append(nn.ELU())
            input_dim = hidden_dim
        critic_layers.append(nn.Linear(input_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, obs):
        # Actor
        actor_features = self.actor_backbone(obs)
        action_mean = self.actor_mean(actor_features)
        action_std = torch.exp(self.actor_logstd)
        
        # Critic
        value = self.critic(obs)
        
        return action_mean, action_std, value
    
    def act(self, obs, deterministic=False):
        """Sample action from policy"""
        action_mean, action_std, value = self.forward(obs)
        
        if deterministic:
            return action_mean, value
        
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, action_log_prob, value
    
    def evaluate(self, obs, actions):
        """Evaluate actions under current policy"""
        action_mean, action_std, value = self.forward(obs)
        
        dist = Normal(action_mean, action_std)
        action_log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action_log_prob, value, entropy