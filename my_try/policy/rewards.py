import torch

class Rewards:
    def __init__(
        self,
        v_target=0.6,
        w_vel=10.0,
        w_lat=0.5,
        w_yaw=0.2,
        w_upright=1.0,
        w_height=0.5,
        w_energy=0.002,
        w_smooth=0.05,
        w_zvel=0.2,
        w_slip=0.5,
        w_alive=0.01,
        w_rear_knee=1.0,
        w_clear=1.0,
        w_back=0.8,
    ):
        self.v_target = v_target
        self.w_vel = w_vel
        self.w_lat = w_lat
        self.w_yaw = w_yaw
        self.w_upright = w_upright
        self.w_height = w_height
        self.w_energy = w_energy
        self.w_smooth = w_smooth
        self.w_zvel = w_zvel
        self.w_slip = w_slip
        self.w_alive = w_alive
        self.prev_actions = None
        self.w_rear_knee = w_rear_knee
        self.w_clear = w_clear
        self.w_back = w_back 
    
    def __call__(self, obs, actions, info):
        v = info["base_vel"]               # (N,3)
        w = info.get("base_ang_vel", None) # (N,3) if available

        # 1) velocity tracking (smooth, bounded)
        v_des = info["commands"][:, 0]                 # desired forward vel
        v_err = v[:, 0] - v_des
        r_vel = torch.exp(-2.0 * v_err**2)


        p_back = torch.relu(-v[:, 0])  # penalize negative forward velocity


        # 2) don’t drift sideways / yaw wildly
        p_lat = v[:, 1]**2
        p_yaw = (w[:, 2]**2) if w is not None else 0.0

        # 3) stay upright (orientation_error should be small)
        # if orientation_error is already a magnitude, use exp(-k*err^2)
        upright_err = info["orientation_error"]
        r_upright = torch.exp(-3.0 * upright_err**2)

        # indices: 0 FL_hip,1 FL_thigh,2 FL_calf,3 FR_hip,4 FR_thigh,5 FR_calf,
        #          6 RL_hip,7 RL_thigh,8 RL_calf,9 RR_hip,10 RR_thigh,11 RR_calf
        rear_calf = info["dof_pos"][:, [8, 11]]          # RL_calf, RR_calf
        # penalize when calves fold past a limit (more negative = tucked under)
        over_fold = torch.relu((-rear_calf) - 1.7)       # threshold tunes how upright you want
        p_rear_knee = over_fold.sum(dim=-1)

        # 4) reasonable height (centered around nominal), no hard clamp
        h = info["base_height"]
        h0 = 0.28
        r_height = torch.exp(-50.0 * (h - h0)**2)

        # 5) energy + smooth actions
        p_energy = (actions**2).sum(dim=-1)
        if self.prev_actions is None:
            p_smooth = torch.zeros_like(p_energy)
        else:
            p_smooth = ((actions - self.prev_actions)**2).sum(dim=-1)
        self.prev_actions = actions.detach()

        # 6) penalize vertical bobbing (prevents hopping)
        p_zvel = v[:, 2]**2

        # 7) foot slip penalty (ONLY if you have foot velocities)
        p_slip = 0.0
        if "foot_vel" in info:
            contacts = info["foot_contacts"].unsqueeze(-1)  # (N,4,1)
            foot_v_xy = info["foot_vel"][..., :2]           # (N,4,2)
            slip = (foot_v_xy**2).sum(dim=-1)               # (N,4)
            p_slip = (slip * info["foot_contacts"]).sum(dim=-1)

        alive = torch.ones_like(p_energy)
        foot_v = info["foot_vel"][0]                 # (4,3)
        contacts = info["foot_contacts"][0].float()  # (4,)
        slip = (foot_v[:, :2]**2).sum(dim=-1)

        contacts = info["foot_contacts"].float()
        num_contact = contacts.sum(dim=1)
        p_airborne = (num_contact < 1.0).float()
        p_all_stance = (num_contact > 3.5).float()  # all four down

        foot_h = info["foot_height"]           # shape (N,4)
        clear = torch.clamp(foot_h - 0.03, min=0.0)  # height above 3 cm
        r_clear = clear.sum(dim=1)
        """
        print("r_vel:", r_vel * self.w_vel,
                "r_upright:", r_upright * self.w_upright,
                "r_height:", r_height * self.w_height,
                "p_lat:", p_lat * self.w_lat,
                "p_yaw:", p_yaw * self.w_yaw,
                "p_energy:", p_energy * self.w_energy,
                "p_smooth:", p_smooth * self.w_smooth,
                "p_zvel:", p_zvel * self.w_zvel,
                "p_slip:", p_slip * self.w_slip,
                "p_airborne:", p_airborne * self.w_slip,
                "p_all_stance:", p_all_stance * self.w_slip,
                "p_rear_knee:", p_rear_knee * self.w_rear_knee,
                "r_clear:", r_clear * self.w_clear,
                "p_back:", p_back * self.w_back,
            )"""

        return (
            self.w_vel * r_vel
            + self.w_upright * r_upright
            + self.w_height * r_height
            + self.w_alive * alive
            - self.w_lat * p_lat
            - self.w_yaw * p_yaw
            - self.w_energy * p_energy
            - self.w_smooth * p_smooth
            - self.w_zvel * p_zvel
            - self.w_slip * p_slip
            - self.w_slip * p_airborne
            - self.w_slip * p_all_stance
            - self.w_rear_knee * p_rear_knee
            + self.w_clear * r_clear
            - self.w_back * p_back
        )
    
    def adjust_all(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
