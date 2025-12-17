import torch
import numpy as np 

from environment.make_environment import Go2WalkingEnv
from policy.ppo import PPO

from tqdm.auto import tqdm


def _fmt(x, default="—", prec=4):
    if x is None:
        return default
    try:
        if torch.is_tensor(x):
            x = x.item()
        return f"{float(x):.{prec}f}"
    except Exception:
        return default

class Runner: 
    def __init__(self, env: Go2WalkingEnv, policy: PPO):
        self.env = env
        self.policy = policy
        self.obs = self.env.reset()
        self.done = False
        self.reward = 0.0
        self.global_step = 0

    def training_history(self, update_stats, rollout_reward_sum, update_idx):
        
        avg_rollout_reward = rollout_reward_sum / self.policy.num_steps_per_update

        # book-keeping
        update_stats = dict(update_stats)  # ensure normal dict
        update_stats["update"] = update_idx + 1
        update_stats["avg_rollout_reward"] = avg_rollout_reward
        update_stats["global_step"] = int(self.global_step)
        return update_stats, avg_rollout_reward

    def store_temp_version(self, update_idx=None): 
        if update_idx % 10 == 0: 
            """Store a temporary copy of the current policy weights to file."""
            self.temp_state_dict = {
                k: v.cpu().clone() for k, v in self.policy.actor_critic.state_dict().items()
            }
            torch.save(self.temp_state_dict, f"checkpoints/temp_policy_{update_idx}.pth")

    def load_temp_version(self, latest):
        """Load a temporary copy of the policy weights from file."""
        temp_state_dict = torch.load(f"checkpoints/temp_policy_{latest}.pth")
        self.policy.actor_critic.load_state_dict(temp_state_dict)

    def learn(self, num_updates=10, latest=None, show_progress=True):
        """Collect rollouts and update the PPO policy."""
        self.reset()
        training_history = []
        if latest is not None:
            self.load_temp_version(latest)
        
        self.policy.actor_critic.train()

        # progress over updates
        pbar = tqdm(
            range(num_updates),
            desc="PPO",
            leave=True,
            dynamic_ncols=True,
            disable=not show_progress,
        )

        for update_idx in pbar:
            rollout_reward_sum = 0.0
            self.store_temp_version(update_idx)

            for _ in range(self.policy.num_steps_per_update):

                obs_tensor = self.obs.detach()  # clone usually not needed
                action, log_prob, value = self.policy.act(obs_tensor)

                _, reward, done, _ = self.step(action)

                self.policy.store_transition(
                    obs_tensor,
                    action,
                    reward,
                    done.float(),
                    value.squeeze(-1),
                    log_prob,
                )

                rollout_reward_sum += reward.mean().item()
                self.global_step += self.env.num_envs

            with torch.no_grad():
                _, _, last_values = self.policy.actor_critic(self.obs)

            update_stats = self.policy.ppo_update(last_values.squeeze(-1))
            update_stats, avg_rollout_reward = self.training_history(update_stats, rollout_reward_sum, update_idx)
            training_history.append(
                update_stats
            )
            
            pbar.set_postfix({
                "R": _fmt(avg_rollout_reward, prec=3),
                "pi_loss": _fmt(update_stats.get("policy_loss"), prec=3),
                "vf_loss": _fmt(update_stats.get("value_loss"), prec=3),
                "steps": update_stats["global_step"],
            })

        return training_history
    
    def make_video(self, max_steps=100, video_path="simulation.mp4"):
        """Generate a video of the agent's behavior."""
        camera = self.env.get_camera()
        camera.start_recording()
        rewards = []
        for i in range(max_steps):
            with torch.no_grad(): 
                obs_tensor = self.obs.detach().clone()
                action, log_prob, value = self.policy.act(obs_tensor)

                _, reward, done, _ = self.step(action)
                camera.render()
                rewards.append(reward)
            
        camera.stop_recording(save_to_filename=video_path, fps=60)
       

    def reset(self):
        self.obs = self.env.reset()
        self.done = False
        self.reward = 0.0
    
    def step(self, action):
        self.obs, reward, done, info = self.env.step(action)
        self.reward = reward
        self.done = done
        return self.obs, reward, done, info
    
    def get_observation(self):
        return self.obs
    
    def is_done(self):
        return self.done
