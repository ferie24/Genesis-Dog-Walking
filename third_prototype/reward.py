import torch


DEFAULT_SCALES = {
    "tracking_lin_vel_x": 1.0,
    "tracking_ang_vel": 1.0,
    "lin_vel_z": -1.0,
    "lin_vel_y": -5.0,
    "action_rate": -0.005,
    "similar_to_default": -0.1,
    "sideway_movement": -1.0,
    "x_progress": 1.0,
    "orientation": 0.0,
    "rear_legs_air": 0.0,
    "heading_error": -0.5,
}

class Rewards:
    def __init__(self, tracking_sigma=0.25, scales=None):
        self.tracking_sigma = float(tracking_sigma)
        #self.inv_tracking_sigma = 1.0 / self.tracking_sigma
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
        self.s_x_progress = float(self.scales["x_progress"])
        self.s_orientation = float(self.scales.get("orientation", 0.0))
        self.s_rear_legs_air = float(self.scales.get("rear_legs_air", 0.0))
        self.s_heading_error = float(self.scales.get("heading_error", 0.0))

    def __call__(self, obs, actions, info):
        # Nutze das LOKALE Koordinatensystem des Roboters für Geschwindigkeiten!
        base_lin_vel_base = info["base_lin_vel_base"]  # <-- LOKAL (Getauscht)
        base_ang_vel = info["base_ang_vel"]
        base_pos = info["base_pos"]
        base_init_pos = info["base_init_pos"]
        dof_pos = info["dof_pos"]
        default_dof_pos = info["default_dof_pos"]
        commands = info["commands"]
        last_actions = info["last_actions"]
        x_progress = info["x_progress"]
        projected_gravity = info["projected_gravity"]
        foot_contacts = info["foot_contacts"]
        heading_error = info["heading_error"]


        # Wir vergleichen jetzt das Kommando mit der LOKALEN X-Geschwindigkeit
        
        tracking_lin_vel_x = torch.exp(
            -torch.square(commands[:, 0] - base_lin_vel_base[:, 0]) / self.tracking_sigma
        )

        tracking_ang_vel = torch.exp(
            -torch.abs(commands[:, 2] - base_ang_vel[:, 2]) / self.tracking_sigma
        )

        # Die Strafen auf Z und Y sollten auch im lokalen Frame evaluiert werden
        lin_vel_z = torch.square(base_lin_vel_base[:, 2])
        lin_vel_y = torch.square(base_lin_vel_base[:, 1])

        action_rate = torch.sum(torch.square(last_actions - actions), dim=1)

        similar_to_default = torch.sum(
            torch.abs(dof_pos - default_dof_pos), dim=1
        )

        sideway_movement = torch.clamp(
            torch.abs(base_pos[:, 1] - base_init_pos[1]), max=2.0
        )       

        orientation = torch.sqrt(
            projected_gravity[:, 0] ** 2
            + projected_gravity[:, 1] ** 2
        )

        command_x = commands[:, 0].clamp(min=0.1)

        progress_ratio = torch.clamp(
            x_progress / command_x,
            min=0.0,
            max=1.0,
        )

        contacts = foot_contacts > 0.5
        rl = contacts[:, 2]
        rr = contacts[:, 3]
        both_rear_air = (~rl) & (~rr)
        moving = commands[:, 0].abs() > 0.1
        reward_rear_legs_air = both_rear_air.float() * moving.float()

        
        cos_heading_error = 1.0 - torch.cos(heading_error)

        reward = (
                self.s_tracking_lin_vel_x * tracking_lin_vel_x
                + self.s_tracking_ang_vel * tracking_ang_vel
                + self.s_lin_vel_z * lin_vel_z
                + self.s_lin_vel_y * lin_vel_y
                + self.s_action_rate * action_rate
                + self.s_similar_to_default * similar_to_default
                + self.s_sideway_movement * sideway_movement
                + self.s_x_progress * progress_ratio
                + self.s_orientation * orientation
                + self.s_heading_error * cos_heading_error
                + self.s_rear_legs_air * reward_rear_legs_air
        )
        return reward, tracking_lin_vel_x #* 0.1
