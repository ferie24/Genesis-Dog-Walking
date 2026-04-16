import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, num_inputs,
                 num_outputs,
                 epsilon):
        """
        Network constructor.
        policy network gives scores for states -> probs -> actions
        """
        super().__init__()
        num_states = num_inputs
        num_actions = num_outputs

        # Actor head (continuous actions, e.g., torques)
        self.actor = nn.Sequential(
            nn.Linear(num_states, 512),
            nn.ELU(),  # Or ReLU; ELU common in locomotion
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),

        )
        self.actor_mean = nn.Linear(128, num_actions)
        self.log_std = nn.Parameter(torch.zeros(num_actions))

        # Critic head (value estimate)
        self.critic = nn.Sequential(
            nn.Linear(num_states, 512),
            nn.ELU(),  # Or ReLU; ELU common in locomotion
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

        self.epsilon = epsilon
        self._init_weights()

    def _init_weights(self):
        # Actor-Mean-Output normal initialisieren
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        # Critic-Output normal initialisieren
        for net in [self.actor, self.critic]:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=1.0)
                    nn.init.zeros_(layer.bias)

    def _distribution(self, state):
        mean = self.actor_mean(self.actor(state))
        log_std = torch.clamp(self.log_std, min=-4.0, max=1.0)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mean, std), mean, std
    
    def forward(self, state):
        dist, mean, std = self._distribution(state)
        value = self.critic(state)
        return mean, std, value

    def get_actions(self, state):
        dist, mean, std = self._distribution(state)
        value = self.critic(state)
        actions = dist.sample()
        return actions, value

    def get_value(self, state):
        return self.critic(state)

    def compute_log_probs(self, states, actions):
        dist, mean, std = self._distribution(states)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy   = dist.entropy().sum(dim=-1)
        value     = self.critic(states)
        return log_probs, value, entropy

    def compute_loss(self, states, actions, advantages, log_probs_old,
                 returns, old_mu, old_sigma, old_values=None):
        dist, mu, sigma = self._distribution(states)
        log_probs_new = dist.log_prob(actions).sum(dim=-1)
        
        entropy = dist.entropy().sum(dim=-1)
        values_new = self.critic(states)

        kl = torch.sum(
            (torch.log(sigma + 1e-5) - torch.log(old_sigma + 1e-5))
            + (old_sigma**2 + (old_mu - mu)**2) / (2.0 * sigma**2)
            - 0.5,
            dim=-1
        ).mean()

        log_ratio = torch.clamp(log_probs_new - log_probs_old.squeeze(-1), min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        surr1 = ratio * advantages.squeeze(-1)
        surr2 = ratio.clamp(1 - self.epsilon, 1 + self.epsilon) * advantages.squeeze(-1)
        actor_loss = -torch.min(surr1, surr2).mean()

        # CORRECTED: Value clipping relativ zu OLD values, nicht zu returns!
        #values_new = self.critic(states)
        returns_sq = returns.squeeze(-1)
        values_sq = values_new.squeeze(-1)
        
        if old_values is not None:
            # CORRECT: Clip relative to OLD value estimate, not returns
            old_values_sq = old_values.squeeze(-1)
            val_clipped = old_values_sq + (values_sq - old_values_sq).clamp(
                -self.epsilon, self.epsilon
            )
            # Compute loss as max of clipped and unclipped
            critic_loss = torch.max(
                (values_sq - returns_sq).pow(2),
                (val_clipped - returns_sq).pow(2)
            ).mean()
        else:
            # Fallback if old_values not available
            critic_loss = (values_sq - returns_sq).pow(2).mean()
        
        return critic_loss, actor_loss, entropy.mean(), kl













