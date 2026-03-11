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



        self.steps = 0

    def reset(self):
        self.steps = 0

    def add_step(self, obs, actions, log_probs, rewards, dones, values):
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"obs dim {obs.shape} != {self.obs_dim}")
        if actions.shape[-1] != self.act_dim:
            raise ValueError(f"actions dim {actions.shape} != {self.act_dim}")
        #if rewards.dim() == 1: rewards = rewards.unsqueeze(-1)  # (N,) -> (N,1)
        #if log_probs.dim() == 1: log_probs = log_probs.unsqueeze(-1)
        #if dones.dim() == 1: dones = dones.unsqueeze(-1).float()
        #if values.dim() == 1: values = values.unsqueeze(-1)

        t = self.steps
        self.steps += 1
        self.obs[t + 1].copy_(obs)
        self.actions[t].copy_(actions)
        self.rewards[t].copy_(rewards)
        self.log_probs[t].copy_(log_probs)
        self.values[t].copy_(values)
        self.dones[t].copy_(dones)

    def init_obs(self, obs0, values0):
        self.obs[0] = obs0
        self.values[0] = values0

    def compute_returns_and_advantages(self, gamma, lmbda):
        advantages = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        for t in reversed(range(self.max_length)):
            if t == self.max_length:
                next_value = torch.zeros_like(self.values[0])
            else:
                next_value = self.values[t + 1]
            delta = self.rewards[t].squeeze(-1) + gamma * next_value.squeeze(-1) * (1 - self.dones[t].squeeze(-1)) - \
                    self.values[t].squeeze(-1)
            advantages = delta + gamma * lmbda * advantages * (1 - self.dones[t].squeeze(-1))
            self.advantages[t] = advantages.unsqueeze(-1)
            self.returns[t] = advantages + self.values[t].squeeze(-1)
            self.returns[t] = self.returns[t].unsqueeze(-1)
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batch(self, batch_size):
        # Flatten for PPO updates: (T*N, D)
        indices = torch.randperm(self.ptr * self.num_envs)[:batch_size]
        batch = dict(
            obs=self.obs[:-1].reshape(-1, self.obs_dim)[indices],
            actions=self.actions.reshape(-1, self.act_dim)[indices],
            log_probs=self.log_probs.reshape(-1, 1)[indices],
            advantages=self.advantages.reshape(-1, 1)[indices],
            returns=self.returns.reshape(-1, 1)[indices],
        )
        return batch



