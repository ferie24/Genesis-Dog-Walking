import torch

class Rewards:
    def __init__(self, forward_weight):
        self.forward = forward_weight

    def __call__(self, obs, actions, info):

        lin_vel = info["base_vel"]
        commands = info["commands"]
        lin_vel_reward = self.calc_lin_vel_forward_reward(lin_vel, commands)
        return lin_vel_reward

    def calc_lin_vel_forward_reward(self, lin_vel, commands):
        base_vel = lin_vel[:, 0]
        target_vel = commands[:, 0]

        vel_error = torch.exp(-10 * torch.abs(base_vel - target_vel))
        return vel_error