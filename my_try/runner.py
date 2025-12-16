import torch
import numpy as np 

from environment.make_environment import Go2WalkingEnv
from policy.ppo import PPO

class Runner: 
    def __init__(self, env: Go2WalkingEnv, policy: PPO):
        self.env = env
        self.policy = policy
        self.obs = self.env.reset()
        self.done = False
        self.reward = 0.0
        self.global_step = 0

    def learn(self, num_updates=10, checkpoints=None):
        """Collect rollouts and update the PPO policy."""
        self.reset()
        training_history = []

        for update_idx in range(num_updates):
            rollout_reward = 0.0

            for _ in range(self.policy.num_steps_per_update):
                obs_tensor = self.obs.detach().clone()
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

                rollout_reward += reward.mean().item()
                self.global_step += self.env.num_envs

            with torch.no_grad():
                _, _, last_values = self.policy.actor_critic(self.obs)

            update_stats = self.policy.ppo_update(last_values.squeeze(-1))
            update_stats["update"] = update_idx + 1
            update_stats["avg_rollout_reward"] = rollout_reward / self.policy.num_steps_per_update
            training_history.append(update_stats)

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
