import torch


DEFAULT_SCALES = {
    "tracking_lin_vel_x": 1.0,
    "tracking_ang_vel": 1.0,
    "lin_vel_z": -1.0,
    "lin_vel_y": -5.0,
    "action_rate": -0.005,
    "similar_to_default": -0.1,
    "sideway_movement": -1.0,
}

class Rewards:
    def __init__(self, tracking_sigma=0.25, scales=None):
        self.tracking_sigma = float(tracking_sigma)
        self.inv_tracking_sigma = 1.0 / self.tracking_sigma
        resolved_scales = DEFAULT_SCALES.copy()
        if scales is not None:
            resolved_scales.update(scales)
        self.scales = resolved_scales
        # Cache active scales once to avoid repeated dict lookups in each env step.
        self.s_tracking_lin_vel_x = float(self.scales["tracking_lin_vel_x"])
        self.s_tracking_ang_vel = float(self.scales["tracking_ang_vel"])
        self.s_lin_vel_z = float(self.scales["lin_vel_z"])
        self.s_lin_vel_y = float(self.scales["lin_vel_y"])
        self.s_action_rate = float(self.scales["action_rate"])
        self.s_similar_to_default = float(self.scales["similar_to_default"])
        self.s_sideway_movement = float(self.scales["sideway_movement"])


    def __call__(self, obs, actions, info):
        base_vel = info["base_lin_vel_base"]
        base_ang_vel = info["base_ang_vel"]
        base_pos = info["base_pos"]
        base_init_pos = info["base_init_pos"]
        dof_pos = info["dof_pos"]
        default_dof_pos = info["default_dof_pos"]
        commands = info["commands"]
        last_actions = info["last_actions"]

        tracking_lin_vel_x = torch.exp(
            -torch.square(commands[:, 0] - base_vel[:, 0]) * self.inv_tracking_sigma
        )

        tracking_ang_vel = torch.exp(
            -torch.abs(commands[:, 2] - base_ang_vel[:, 2]) * self.inv_tracking_sigma
        )

        lin_vel_z = torch.square(base_vel[:, 2])
        lin_vel_y = torch.square(base_vel[:, 1])

        action_rate = torch.sum(torch.square(last_actions - actions), dim=1)

        similar_to_default = torch.sum(
            torch.abs(dof_pos - default_dof_pos), dim=1
        )

        sideway_movement = torch.clamp(
            torch.abs(base_pos[:, 1] - base_init_pos[1]), max=2.0
        )

        reward = (
                self.s_tracking_lin_vel_x * tracking_lin_vel_x
                + self.s_tracking_ang_vel * tracking_ang_vel
                + self.s_lin_vel_z * lin_vel_z
                + self.s_lin_vel_y * lin_vel_y
                + self.s_action_rate * action_rate
                + self.s_similar_to_default * similar_to_default
                + self.s_sideway_movement * sideway_movement
        )
        return reward, tracking_lin_vel_x