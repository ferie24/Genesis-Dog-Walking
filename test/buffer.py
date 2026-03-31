import torch

class Buffer:
    """
    Buffer to gather Episodes for improving the policy
    """
    def __init__(self, num_envs, obs_dim, act_dim, max_length, device):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.max_length = max_length

        self.obs = torch.zeros(self.max_length+1, self.num_envs, self.obs_dim, device=device)
        self.actions = torch.zeros(self.max_length, self.num_envs, self.act_dim, device=device)

        self.rewards = torch.zeros((max_length, num_envs, 1), device=device)
        self.log_probs = torch.zeros((max_length, num_envs, 1), device=device)
        self.values = torch.zeros((max_length + 1, num_envs, 1), device=device)
        self.dones = torch.zeros((max_length, num_envs, 1), device=device)
        self.advantages = torch.zeros((max_length, num_envs, 1), device=device)
        self.returns = torch.zeros((max_length, num_envs, 1), device=device)

        self.lin_vel_rewards = torch.zeros((max_length, num_envs, 1), device=device)

        self.steps = 0

        self.obs_mean  = torch.zeros(obs_dim, device=device)
        self.obs_var   = torch.ones(obs_dim, device=device)
        self.obs_count = torch.tensor(1e-4, device=device)
        # for network changes 

        self.mu    = torch.zeros(max_length, num_envs, act_dim, device=device)
        self.sigma = torch.zeros(max_length, num_envs, act_dim, device=device)

        self.time_outs = torch.zeros((max_length, num_envs, 1), device=device)

    def reset(self):
        self.steps = 0

    def add_step(self, obs, actions, log_probs, rewards, dones, values, lin_vel_rewards, mu, sigma, time_outs):
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"obs dim {obs.shape} != {self.obs_dim}")
        if actions.shape[-1] != self.act_dim:
            raise ValueError(f"actions dim {actions.shape} != {self.act_dim}")
        if rewards.dim() == 1: rewards = rewards.unsqueeze(-1)  # (N,) -> (N,1)
        if log_probs.dim() == 1: log_probs = log_probs.unsqueeze(-1)
        if dones.dim() == 1: dones = dones.unsqueeze(-1).float()
        if values.dim() == 1: values = values.unsqueeze(-1)
        if lin_vel_rewards.dim() == 1: lin_vel_rewards = lin_vel_rewards.unsqueeze(-1)
        if time_outs.dim() == 1: time_outs = time_outs.unsqueeze(-1).float()

        t = self.steps
        self.steps += 1
        # In Buffer.add_step
        if torch.isnan(obs).any():
            nan_mask = torch.isnan(obs).any(dim=-1)  # (N,) bool
            obs = obs.clone()
            obs[nan_mask] = 0.0      # neutrale Observation
            dones[nan_mask] = True   # Episode als beendet markieren
            rewards[nan_mask] = 0.0  # kein falsches Reward-Signal

        self.obs[t + 1].copy_(self.normalize_obs(obs.detach()))
        self.actions[t].copy_(actions.detach())
        self.rewards[t].copy_(rewards)
        self.log_probs[t].copy_(log_probs.detach())
        self.values[t].copy_(values.detach())
        self.dones[t].copy_(dones.detach())
        self.lin_vel_rewards[t].copy_(lin_vel_rewards.detach())
        self.mu[t].copy_(mu.detach())
        self.sigma[t].copy_(sigma.detach())
        self.time_outs[t].copy_(time_outs.detach())

        #self.update_obs_stats(obs.detach())

    def init_obs(self, obs0, values0):
        self.obs[0] = obs0
        self.values[0] = values0
    
    def update_obs_stats(self, obs):
        """Aufruf einmal pro Step im Rollout"""
        batch_mean  = obs.mean(dim=0)
        batch_var   = obs.var(dim=0)
        batch_count = obs.shape[0]
        total = self.obs_count + batch_count
        self.obs_mean = (self.obs_count * self.obs_mean + batch_count * batch_mean) / total
        self.obs_var  = (self.obs_count * self.obs_var  + batch_count * batch_var)  / total
        self.obs_count = total

    def normalize_obs(self, obs):
        return ((obs - self.obs_mean) / (self.obs_var.sqrt() + 1e-8)).clamp(-5, 5)

    def compute_returns_and_advantages(self, gamma, lmbda):
        advantages = torch.zeros((self.num_envs, 1), device=self.device)

        for t in reversed(range(self.steps)):
            terminated = self.dones[t].float() * (1 - self.time_outs[t].float()) # (N,1)
            mask = 1 - terminated  # (N,1) 1 für nicht-terminierte und timeouts, 0 für terminierte Schritte
            delta = self.rewards[t] + gamma * self.values[t + 1] * mask - self.values[t]
            advantages = delta + gamma * lmbda * advantages * mask

            self.advantages[t].copy_(advantages)
            self.returns[t].copy_(advantages + self.values[t])
        # Normalize advantages and returns otherwise explode. 
        self.advantages[:self.steps] = ((
                    self.advantages[:self.steps] - self.advantages[:self.steps].mean()) /
                    (self.advantages[:self.steps].std() + 1e-8)
                    ) 
        #self.returns[:self.steps] = (
        #            (self.returns[:self.steps] - self.returns[:self.steps].mean()) /
        #            (self.returns[:self.steps].std() + 1e-8)
        #            )

    def get_batch(self, batch_size):
        indices = torch.randperm(self.steps * self.num_envs)[:batch_size]
        batch = dict(
            obs=self.obs[:-1].reshape(-1, self.obs_dim)[indices].clone().detach(),
            actions=self.actions.reshape(-1, self.act_dim)[indices].clone().detach(),
            log_probs=self.log_probs.reshape(-1, 1)[indices].clone().detach(),
            advantages=self.advantages.reshape(-1, 1)[indices].clone().detach(),
            returns=self.returns.reshape(-1, 1)[indices].clone().detach(),
            values=self.values.reshape(-1, 1)[indices].clone().detach(),
            mu=self.mu.reshape(-1, self.act_dim)[indices].clone().detach(),
            sigma=self.sigma.reshape(-1, self.act_dim)[indices].clone().detach(),
            time_outs=self.time_outs.reshape(-1, 1)[indices].clone().detach(),
        )
        return batch

    def get_minibatches(self, num_minibatches):
        total_size = self.steps * self.num_envs

        # Einmal reshapen, einmal permutieren
        obs = self.obs[:-1].reshape(-1, self.obs_dim).detach()
        actions = self.actions.reshape(-1, self.act_dim).detach()
        log_probs = self.log_probs.reshape(-1, 1).detach()
        advantages = self.advantages.reshape(-1, 1).detach()
        returns = self.returns.reshape(-1, 1).detach()
        values = self.values.reshape(-1, 1).detach()
        mu = self.mu.reshape(-1, self.act_dim).detach()
        sigma = self.sigma.reshape(-1, self.act_dim).detach()
        time_outs = self.time_outs.reshape(-1, 1).detach()

        indices = torch.randperm(total_size, device=self.device)
        batch_size = total_size // num_minibatches  

        for start in range(0, num_minibatches * batch_size, batch_size):
            idx = indices[start : start + batch_size]
            yield dict(
                obs=obs[idx], actions=actions[idx],
                log_probs=log_probs[idx], advantages=advantages[idx],
                returns=returns[idx], values=values[idx],
                mu=mu[idx], sigma=sigma[idx],
                time_outs=time_outs[idx]

            )




