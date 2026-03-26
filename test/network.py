from math import gamma

import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, num_inputs,
                 num_outputs,
                 #gamma,
                 #lmbda,
                 epsilon):
        """
        Network constructor.
        policy network gives scores for states -> probs -> actions
        """
        super().__init__()
        num_states = num_inputs
        num_actions = num_outputs
        self.shared = nn.Sequential(
            nn.Linear(num_states, 512),
            nn.ELU(),  # Or ReLU; ELU common in locomotion
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU()
        )

        # Actor head (continuous actions, e.g., torques)
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),# Normalize actions to [-1,1]; scale in forward if needed
        )
        self.actor_mean = nn.Linear(32, num_actions)
        self.actor_logstd = nn.Parameter(torch.zeros(num_actions))

        # Critic head (value estimate)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

        #self.gamma = gamma
        #self.lmbda = lmbda
        self.epsilon = epsilon

    def forward(self, state):
        combined = self.shared(state)
        action_feats = self.actor(combined)
        action_mean = self.actor_mean(action_feats)
        #action_std = torch.exp(self.actor_logstd)
        action_std = torch.exp(torch.clamp(self.actor_logstd, -5.0, 2.0))
        value = self.critic(combined)

        return action_mean, action_std, value

    def get_actions(self, state):
        action_mean, action_std, value = self.forward(state)

        dist = torch.distributions.Normal(action_mean, action_std)
        actions = dist.sample()
        return actions, value

    def get_value(self, state):

        action_mean, action_std, value = self.forward(state)
        return value

    def compute_log_probs(self, states, actions):
        action_mean, action_std, value = self.forward(states)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, value, entropy

    def compute_advantages(self, states, rewards, next_states, dones):
        raise NotImplementedError

    def compute_loss(self, states, actions, advantages, critic_targets, log_probs_old, returns):

        log_probs_new, values_new, entropy = self.compute_log_probs(states, actions)

        critic_loss = F.mse_loss(values_new.squeeze(-1), returns.squeeze(-1))
        ratios = torch.exp(log_probs_new - log_probs_old.squeeze(-1))

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        return critic_loss, actor_loss, entropy.mean()













