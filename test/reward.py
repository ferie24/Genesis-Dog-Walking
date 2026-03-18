import torch

class Rewards:
    def __init__(self, tracking_sigma=0.25, scales=None):
        self.tracking_sigma = tracking_sigma
        self.scales = {
            "tracking_lin_vel_x": 1.5,
            "tracking_ang_vel": 0.5,
            "x_progress": 0.5,
            "lin_vel_z": 0.2,
            "lin_vel_y": 0.1,
            "action_rate": 0.01,
            "similar_to_default": 0.02,
            "termination": 1.0,
            "sideway_movement": 0.1,
            "base_height": 0.1,
        }
        if scales is not None:
            self.scales.update(scales)


    def __call__(self, obs, actions, info, tracking_sigma=0.25):
        base_vel = info["base_vel"]
        base_ang_vel = info["base_ang_vel"]
        base_pos = info["base_pos"]
        base_init_pos = info["base_init_pos"]
        dof_pos = info["dof_pos"]
        default_dof_pos = info["default_dof_pos"]
        commands = info["commands"]
        last_actions = info["last_actions"]
        reset_buf = info["reset_buf"]
        episode_length_buf = info["episode_length_buf"]
        max_episode_length = info["max_episode_length"]
        base_height = info["base_height"]

        tracking_lin_vel_x = torch.exp(
            -torch.square(commands[:, 0] - base_vel[:, 0]) / tracking_sigma
        )

        tracking_ang_vel = torch.exp(
            -torch.abs(commands[:, 2] - base_ang_vel[:, 2]) / tracking_sigma
        )

        lin_vel_z = torch.square(base_vel[:, 2])
        lin_vel_y = torch.square(base_vel[:, 1])

        action_rate = torch.sum(torch.square(last_actions - actions), dim=1)

        similar_to_default = torch.sum(
            torch.abs(dof_pos - default_dof_pos), dim=1
        )

        non_timeout_reset = (reset_buf == 1) & (episode_length_buf <= max_episode_length)
        termination = non_timeout_reset.float()

        sideway_movement = torch.clamp(
            torch.abs(base_pos[:, 1] - base_init_pos[1]), max=2.0
        )

        x_progress = torch.clamp(
            base_pos[:, 0] - base_init_pos[0], max=1.0
        )

        height = torch.clamp(
            base_height - 0.2, min=0.0, max=0.5
        )

        reward = (
                self.scales["tracking_lin_vel_x"] * tracking_lin_vel_x
                + self.scales["tracking_ang_vel"] * tracking_ang_vel
                #+ self.scales["x_progress"] * x_progress
                - self.scales["lin_vel_z"] * lin_vel_z
                - self.scales["lin_vel_y"] * lin_vel_y
                - self.scales["action_rate"] * action_rate
                - self.scales["similar_to_default"] * similar_to_default
                #- self.scales["termination"] * termination
                - self.scales["sideway_movement"] * sideway_movement
                #- self.scales["base_height"] * height
        )

        return reward