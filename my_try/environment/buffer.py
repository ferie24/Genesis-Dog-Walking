import torch 

class RolloutBuffer:
    def __init__(self, horizon, num_envs, obs_dim, act_dim, device):
        self.horizon = horizon
        self.num_envs = num_envs
        self.device = device

        self.obs = torch.zeros(horizon + 1, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(horizon, num_envs, act_dim, device=device)
        self.log_probs = torch.zeros(horizon, num_envs, device=device)
        self.rewards = torch.zeros(horizon, num_envs, device=device)
        self.dones = torch.zeros(horizon, num_envs, device=device, dtype=torch.bool)
        self.values = torch.zeros(horizon + 1, num_envs, device=device)

        self.advantages = torch.zeros(horizon, num_envs, device=device)
        self.returns = torch.zeros(horizon, num_envs, device=device)

        self.step = 0

    def reset(self):
        self.step = 0

    def insert(self, obs, actions, log_probs, rewards, dones, values_next):
        """
        obs: next obs [N, obs_dim]
        values_next: value at next obs [N]
        """
        t = self.step
        self.obs[t + 1].copy_(obs)
        self.actions[t].copy_(actions)
        self.log_probs[t].copy_(log_probs)
        self.rewards[t].copy_(rewards)
        self.dones[t].copy_(dones)
        self.values[t + 1].copy_(values_next)
        self.step += 1

    def set_initial_obs_and_value(self, obs0, value0):
        self.obs[0].copy_(obs0)
        self.values[0].copy_(value0)

    def compute_returns_and_advantages(self, gamma, lam):
        """
        GAE-Lambda.
        """
        gae = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.horizon)):
            mask = 1.0 - self.dones[t].float()
            delta = self.rewards[t] + gamma * self.values[t + 1] * mask - self.values[t]
            gae = delta + gamma * lam * mask * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def get_flat(self):
        # flatten (T, N) → (T*N)
        T, N = self.horizon, self.num_envs
        flat_obs = self.obs[:-1].reshape(T * N, -1)
        flat_actions = self.actions.reshape(T * N, -1)
        flat_log_probs = self.log_probs.reshape(T * N)
        flat_advantages = self.advantages.reshape(T * N)
        flat_returns = self.returns.reshape(T * N)
        return flat_obs, flat_actions, flat_log_probs, flat_advantages, flat_returns
