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
    "undesired_body_contact": -0.25,
}


class Rewards:
    def __init__(self, tracking_sigma=0.25, scales=None, tracking_sigma_ang_vel=None):
        self.tracking_sigma = float(tracking_sigma)
        self.tracking_sigma_ang_vel = (
            self.tracking_sigma
            if tracking_sigma_ang_vel is None
            else float(tracking_sigma_ang_vel)
        )
        # self.inv_tracking_sigma = 1.0 / self.tracking_sigma
        resolved_scales = DEFAULT_SCALES.copy()
        if scales is not None:
            resolved_scales.update(scales)
        self.scales = resolved_scales
        # Cache active scales once to avoid repeated dict lookups in each env step.
        self.s_tracking_lin_vel_x = float(self.scales["tracking_lin_vel_x"])
        self.s_tracking_ang_vel = float(self.scales["tracking_ang_vel"])
        self.s_tracking_lin_vel_y = float(self.scales["tracking_lin_vel_y"])
        self.s_lin_vel_z = float(self.scales["lin_vel_z"])
        self.s_lin_vel_y = float(self.scales["lin_vel_y"])
        self.s_action_rate = float(self.scales["action_rate"])
        self.s_similar_to_default = float(self.scales["similar_to_default"])
        self.s_sideway_movement = float(self.scales["sideway_movement"])
        self.s_x_progress = float(self.scales["x_progress"])
        self.s_orientation = float(self.scales.get("orientation", 0.0))
        self.s_rear_legs_air = float(self.scales.get("rear_legs_air", 0.0))
        self.s_heading_error = float(self.scales.get("heading_error", 0.0))
        self.s_undesired_body_contact = float(
            self.scales.get("undesired_body_contact", 0.0)
        )

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
        projected_gravity = info["projected_gravity"]
        foot_contacts = info["foot_contacts"]
        heading_error = info["heading_error"]
        undesired_body_contact = info["undesired_body_contact"]
        base_ang_vel_base = info["base_ang_vel_base"]  # <-- LOKAL (Getauscht)

        tracking_lin_vel_x = torch.exp(
            -torch.square(commands[:, 0] - base_lin_vel_base[:, 0])
            / self.tracking_sigma
        )

        yaw_rate_actual = info["base_ang_vel_base"][:, 2]
        yaw_rate_target = info["commands"][:, 2]

        yaw_rate_error = yaw_rate_target - yaw_rate_actual

        tracking_ang_vel = torch.exp(
            -yaw_rate_error.square() / self.tracking_sigma_ang_vel
        )

        tracking_lin_vel_y = torch.exp(
            -torch.square(commands[:, 1] - base_lin_vel_base[:, 1])
            / self.tracking_sigma
        )

        action_rate = torch.sum(torch.square(last_actions - actions), dim=1)

        similar_to_default = (
            torch.sum(torch.abs(dof_pos - default_dof_pos), dim=1)
            if self.s_similar_to_default != 0.0 else 0.0
        )

        orientation = (
            torch.sqrt(projected_gravity[:, 0] ** 2 + projected_gravity[:, 1] ** 2)
            if self.s_orientation != 0.0 else 0.0
        )

        # Signed horizontal speed along the commanded direction, in m/s.
        # Keep the existing x_progress weight for configuration compatibility.
        progress_velocity = self.xy_progress(
            commands=commands,
            yaw=info["yaw"],
            base_vel_world=info["base_vel"],
        )

        contacts = foot_contacts > 0.5
        rl = contacts[:, 2]
        rr = contacts[:, 3]
        both_rear_air = (~rl) & (~rr)
        moving = commands[:, 0].abs() > 0.1
        reward_rear_legs_air = both_rear_air.float() * moving.float()

        cos_heading_error = (
            1.0 - torch.cos(heading_error) if self.s_heading_error != 0.0 else 0.0
        )

        reward = (
            self.s_tracking_lin_vel_x * tracking_lin_vel_x
            + self.s_tracking_ang_vel * tracking_ang_vel
            + self.s_tracking_lin_vel_y * tracking_lin_vel_y
            # + self.s_tracking_lin_vel_z * tracking_lin_vel_z
            # + self.s_lin_vel_z * lin_vel_z
            # + self.s_lin_vel_y * lin_vel_y
            + self.s_action_rate * action_rate
            + self.s_similar_to_default * similar_to_default
            # + self.s_sideway_movement * sideway_movement
            + self.s_x_progress * progress_velocity
            + self.s_orientation * orientation
            + self.s_heading_error * cos_heading_error
            + self.s_rear_legs_air * reward_rear_legs_air
            + self.s_undesired_body_contact * undesired_body_contact
        )
        return reward, tracking_lin_vel_x  # * 0.1

    def xy_progress(self, commands, yaw, base_vel_world):
        """Project world XY velocity onto the yaw-rotated linear command.

        Returns signed speed in m/s, with zero progress when the commanded
        linear speed is at most 0.1 m/s. Yaw-only commands earn no progress.
        """

        command_xy = commands[:, :2]
        target_speed = torch.linalg.vector_norm(command_xy, dim=1)

        direction_local = command_xy / target_speed.clamp_min(1e-6).unsqueeze(1)

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        direction_world = torch.stack(
            (
                cos_yaw * direction_local[:, 0] - sin_yaw * direction_local[:, 1],
                sin_yaw * direction_local[:, 0] + cos_yaw * direction_local[:, 1],
            ),
            dim=1,
        )

        progress_velocity = (base_vel_world[:, :2] * direction_world).sum(dim=1)

        progress_velocity = torch.where(
            target_speed > 0.1,
            progress_velocity,
            torch.zeros_like(progress_velocity),
        )
        return progress_velocity
