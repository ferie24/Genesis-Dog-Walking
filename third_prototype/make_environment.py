import torch
import math
import numpy as np
from pathlib import Path
import genesis as gs
from tensordict import TensorDict
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.entities.rigid_entity import RigidEntity
from genesis.engine.scene import Scene

class Go2WalkingEnv:
    def __init__(
        self,
        num_envs=1,
        device="cuda",
        show_viewer=True,
        use_terrain=False,
        episode_length_s=20.0,
        min_base_height=0.15,
        min_up_dot=0.2,
        reward_fn=None,
    ):
        """
        Args:
            num_envs: Number of parallel environments
            device: 'cuda' or 'cpu'
            show_viewer: Whether to show the visual viewer
            use_terrain: If True, uses complex terrain; if False, uses flat plane
            episode_length_s: Maximum episode length in seconds
            min_base_height: Height below which the robot is considered fallen
            min_up_dot: Minimum dot(base_up, world_up) before considering the robot tipped
        """
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.show_viewer = show_viewer
        self.use_terrain = use_terrain
        self.cfg = {
            "env": {
                "num_envs": num_envs,
                "use_terrain": use_terrain,
                "episode_length_s": episode_length_s,
            }
        }
        # Termination tolerances; loosen to avoid instant resets when touching down
        self.min_base_height = min_base_height
        self.min_up_dot = min_up_dot
        
        # Time and episode settings
        self.dt = 0.02  # 50Hz control frequency
        self.max_episode_length = math.ceil(episode_length_s / self.dt)
        
        # Robot configuration
        self.num_dof = 12  # 12 actuated joints (3 per leg)
        self.num_actions = 12
        # 48 = 3 base lin vel + 3 base ang vel + 3 projected gravity + 3 commands + 12 (dof pos) + 12 (dof vel) + 12 (actions) + 2 Headings sin, cos
        self.num_obs = 50 #48  # Robot state observations
        
        # Action and observation buffers
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.target_dof_pos = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self._gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        self._up_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)

        self._forward_vec = torch.zeros(
            (self.num_envs, 3),
            device=self.device,
        )

        self._forward_vec[:, 0] = 1.0
        
        # Episode tracking
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        
        # Reward tracking
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.episode_sums = {}
        
        # Command targets (linear and angular velocities)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)
        self.commands[:, 0] = 1.0  # Forward velocity target (m/s)
        
        # Default joint positions (standing pose)
        self.default_dof_pos = torch.tensor([
            0.0, 0.8, -1.5,  # FL: hip, thigh, calf
            0.0, 0.8, -1.5,  # FR
            0.0, 1.0, -1.5,  # RL
            0.0, 1.0, -1.5,  # RR
        ], device=self.device)
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0).repeat(self.num_envs, 1)

        # PD controller gains
        self.kp = 20.0
        self.kd = 1.0#0.5
        
        self.reward_fn = (
            reward_fn
            if reward_fn is not None
            else lambda obs, actions, info: torch.zeros(self.num_envs, device=self.device)
        )

        # Camera follow offsets relative to the tracked robot body.
        self._camera_follow_offset = np.array([3.0, 3.0, 2.0], dtype=np.float32)
        self._camera_lookat_offset = np.array([0.0, 0.0, 0.5], dtype=np.float32)

        # Initialize the simulation
        self._create_scene()
        self._add_terrain()
        self._add_robot()

        self.camera = self.scene.add_camera(
            pos=(3.0, 3.0, 2.0),
            lookat=(0.0, 0.0, 0.5),
            fov=45,
            GUI=False,
            res=(720, 720),
        )
        

        self.scene.build(n_envs=num_envs)
        self._setup_robot()
        self._initialize_buffers()

    def get_camera(self): 
        self._update_follow_camera()
        rgb, depth, segmentation, normal = self.camera.render(depth=True, segmentation=True, normal=True)
        return self.camera

    def _update_follow_camera(self):
        """Keep the camera centered on env 0 robot with fixed relative offsets."""
        if not hasattr(self, "camera") or not hasattr(self, "base_pos"):
            return

        robot_pos = self.base_pos[0].detach().cpu().numpy()
        self.camera.set_pose(
            pos=robot_pos + self._camera_follow_offset,
            lookat=robot_pos + self._camera_lookat_offset,
        )

        
    def _create_scene(self):
        """Create the Genesis simulation scene"""
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / self.dt),
                camera_pos=(3.0, 3.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=False,
            ),
            show_viewer=self.show_viewer,
        )

        
        # Get rigid solver reference
        #for solver in self.scene.sim.solvers:
        #    if isinstance(solver, RigidSolver):
        #        self.rigid_solver = solver
        #        break
    
    def _add_terrain(self):
        """Add terrain (flat plane or complex terrain)"""
        if self.use_terrain:
            # Complex terrain with height variations - smoothed out
            self.terrain = self.scene.add_entity(
                gs.morphs.Terrain(
                    n_subterrains=(3, 1),
                    # 1. Streckt das Gelände horizontal (Standard ist oft 0.25 oder 0.1).
                    # Wenn wir es von 0.1 auf 0.2 oder 0.25 erhöhen, werden Steigungen flacher.
                    horizontal_scale=0.25, 
                    
                    # 2. Staucht das Gelände vertikal. Wir halbieren die vertikale
                    # Skalierung (von 0.005 auf 0.0025 oder 0.002), wodurch alle
                    # Fraktale und Stufen nur noch halb so hoch ausfallen.
                    vertical_scale=0.005, 
                    
                    subterrain_size = (25, 15),
                    
                    # fractal_terrain ist gut, aber du könntest hier auch "wave_terrain"
                    # beimischen für sanfte Sinus-Hügel anstatt rauer Fraktale.
                    subterrain_types="fractal_terrain", 
                    randomize=False,
                ),
            )
            # Da das Terrain nun flacher ist, können wir den Roboter etwas niedriger spawnen
            self.base_init_pos = torch.tensor([5.0, 5.0, 1.0], device=self.device)
        else:
            # Simple flat plane
            self.plane = self.scene.add_entity(gs.morphs.Plane())
            # Slightly lower spawn height to reduce nose-first resets on flat ground.
            self.base_init_pos = torch.tensor([0.0, 0.0, 0.36], device=self.device)

    
    def _add_robot(self):
        """Add the Go2 robot to the scene"""
        self.base_init_quat = torch.tensor([
            1.0,
            0.0,
            0.0,
            0.0
        ], device=self.device)

        self.robot: RigidEntity = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                links_to_keep=[
                                "FL_foot",
                                "FR_foot",
                                "RL_foot",
                                "RR_foot"
                                ],
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )
    
    def _setup_robot(self):
        """Configure robot motors and properties"""
        # Get all dof names
        self.dof_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        
        # Map dof names to local indices expected by Genesis
        self.dof_indices = []
        for name in self.dof_names:
            joint = self.robot.get_joint(name)
            idx_local = joint.dofs_idx_local
            if isinstance(idx_local, (list, tuple)) and len(idx_local) == 1:
                self.dof_indices.append(int(idx_local[0]))
            else:
                raise RuntimeError(f"Unexpected dof index format for joint {name}: {idx_local}")
        
        # Configure motors for each joint
        for dof_idx in self.dof_indices:
            self.robot.set_dofs_kp([self.kp], [dof_idx])
            self.robot.set_dofs_kv([self.kd], [dof_idx])
            # Force range expects tensors sized like the provided indices
            self.robot.set_dofs_force_range([-23.7], [23.7], [dof_idx])
        
        # Get foot link names
        self.foot_links = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        # Cache local indices for foot links (contact force tensor is ordered by local link index)
        self.foot_link_indices = []
        for name in self.foot_links:
            link = self.robot.get_link(name)
            if link is None:
                raise RuntimeError(f"Foot link {name} not found in URDF.")
            self.foot_link_indices.append(int(link.idx - self.robot.link_start))
        
    def _initialize_buffers(self):
        """Initialize state buffers after scene is built"""
        # Joint state buffers
        self.dof_pos = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        
        # Base state buffers
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Contact sensors
        self.foot_contacts = torch.zeros((self.num_envs, 4), device=self.device)

        #X_progress tracking
        self.prev_base_pos_x = self.base_pos[:, 0].clone()
        self.x_progress = torch.zeros(self.num_envs, device=self.device)


    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        
        # Vectorized reset (Fixes the teleport bug)
        pos_batch = self.base_init_pos.expand(len(env_ids), -1)
        quat_batch = self.base_init_quat.expand(len(env_ids), -1)
        
        self.robot.set_pos(pos_batch, zero_velocity=True, envs_idx=env_ids)
        self.robot.set_quat(quat_batch, zero_velocity=True, envs_idx=env_ids)
        
        dof_noise = torch.empty_like(self.default_dof_pos[env_ids]).uniform_(-0.1, 0.1)
        self.robot.set_dofs_position(
            self.default_dof_pos[env_ids] + dof_noise,
            self.dof_indices, 
            zero_velocity=True, 
            envs_idx=env_ids
        )
        
        self.actions[env_ids].zero_()
        self.last_actions[env_ids].zero_()
        self.prev_base_pos_x[env_ids] = pos_batch[:, 0]
        self.x_progress[env_ids].zero_()
        
        self._update_state()
        self._update_follow_camera()
        self._compute_observations()
        
        return self.obs_buf

    def get_observations(self):
        """Return observations in rsl_rl TensorDict format."""
        return TensorDict(
            {
                "policy": self.obs_buf.clone(),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )


    def step(self, actions):
        """
        Step the environment forward.
        
        Args:
            actions: Tensor of shape (num_envs, num_actions) with target joint positions
        
        Returns:
            obs: Observations
            rewards: Rewards
            dones: Done flags
            info: Additional info dictionary
        """
        # Clip actions
        # second without clamp, third with 12, first with normal -1 
        torch.clamp(actions, -1.0, 1.0, out=self.actions)
        #self.actions = actions
        # Apply actions to robot
        self.target_dof_pos.copy_(self.default_dof_pos)
        self.target_dof_pos.add_(self.actions, alpha=0.25)

        self.robot.control_dofs_position(self.target_dof_pos, self.dof_indices)
        # Step simulation
        self.scene.step()
        
        # Update state
        self._update_state()
        self._update_follow_camera()
        
        # Compute observations and rewards
        self._compute_observations()
        rewards, lin_vel_x_rew = self._compute_rewards()
        
        # Update episode length
        self.episode_length_buf += 1
        time_outs = self.episode_length_buf >= self.max_episode_length
        terminated, reasons = self._check_termination()
        self.reset_buf = time_outs | terminated
        done_buf = self.reset_buf.clone()

        self.last_actions[:] = self.actions[:]
        
        # Vor dem Reset eine Kopie ziehen
        time_outs_out = time_outs.clone()
        heading_error = self._compute_heading_error()
        
        heading_diag = {
            "heading_error_abs_mean":
                heading_error.abs().mean(),

            "heading_error_signed_mean":
                heading_error.mean(),

            "heading_error_abs_max":
                heading_error.abs().max(),
        }
        foot_diag = self._compute_foot_diagnostics()
        foot_diag.update(self._compute_undesired_body_contacts())
        foot_diag.update(reasons)
        foot_diag.update(heading_diag)
        
        if done_buf.any():
            self.reset(done_buf.nonzero(as_tuple=False).flatten())
        
        extras = {
            "time_outs": time_outs_out.float(),
            "lin_vel_x_rew": lin_vel_x_rew,
            "foot_diag": foot_diag,
            #"undesired_contacts": undesired_contacts,

        }
        return self.get_observations(), rewards, done_buf, extras

        ##info = {
        #    "time_outs" : (self.episode_length_buf >= self.max_episode_length).sum().item(),}

        #return self.obs_buf, rewards, done_buf, time_outs  

    
    def _update_state(self):
        """Refresh state buffers without device/alloc thrash."""
        base_pos = self.robot.get_pos(envs_idx=None)
        base_quat = self.robot.get_quat(envs_idx=None)
        base_vel = self.robot.get_vel(envs_idx=None)
        dof_pos = self.robot.get_dofs_position(self.dof_indices, envs_idx=None)
        dof_vel = self.robot.get_dofs_velocity(self.dof_indices, envs_idx=None)
        contact_forces = self.robot.get_links_net_contact_force(envs_idx=None)

        # Helper to normalize array/tensor to torch on self.device
        def to_torch(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.device, non_blocking=True)
            return torch.as_tensor(x, device=self.device)

        base_pos_t = to_torch(base_pos)
        base_quat_t = to_torch(base_quat)

        # Handle base_vel: ensure shape (...,6)
        if isinstance(base_vel, torch.Tensor):
            bv = base_vel
            if bv.shape[-1] < 6:
                pad = torch.zeros((*bv.shape[:-1], 6 - bv.shape[-1]), device=bv.device, dtype=bv.dtype)
                bv = torch.cat([bv, pad], dim=-1)
            base_vel_t = bv.to(self.device, non_blocking=True)
        else:
            base_vel_np = np.asarray(base_vel)
            if base_vel_np.shape[-1] < 6:
                pad = np.zeros((base_vel_np.shape[0], 6 - base_vel_np.shape[-1]), dtype=base_vel_np.dtype)
                base_vel_np = np.concatenate([base_vel_np, pad], axis=-1)
            base_vel_t = torch.as_tensor(base_vel_np, device=self.device)

        dof_pos_t = to_torch(dof_pos)
        dof_vel_t = to_torch(dof_vel)

        self.base_pos.copy_(base_pos_t)
        self.x_progress = (self.base_pos[:, 0] - self.prev_base_pos_x ) / self.dt
        self.prev_base_pos_x.copy_(self.base_pos[:, 0])
        self.base_quat.copy_(base_quat_t)
        quat_norm = self.base_quat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self.base_quat = self.base_quat / quat_norm
        self.base_lin_vel.copy_(base_vel_t[:, :3])
        self.base_ang_vel.copy_(base_vel_t[:, 3:6])
        self.dof_pos.copy_(dof_pos_t)
        self.dof_vel.copy_(dof_vel_t)

        if contact_forces is None or (hasattr(contact_forces, "numel") and contact_forces.numel() == 0):
            self.foot_contacts.zero_()
        else:
            cf = to_torch(contact_forces)
            for j, link_idx in enumerate(self.foot_link_indices):
                if link_idx >= cf.shape[1]:
                    self.foot_contacts[:, j].zero_()
                else:
                    self.foot_contacts[:, j].copy_((cf[:, link_idx].norm(dim=-1) > 1.0).float())

    
    def _compute_observations(self):
        """Compute observations from current state"""
        # Get projected gravity (orientation information)
        base_quat_inv = inv_quat(self.base_quat)
        proj_gravity = transform_by_quat(self._gravity_vec, base_quat_inv)
        
        # Compute velocity in base frame
        base_lin_vel_base = transform_by_quat(self.base_lin_vel, base_quat_inv)
        base_ang_vel_base = transform_by_quat(self.base_ang_vel, base_quat_inv)

        # Heading Error: sin and cos of heading error
        heading_error = self._compute_heading_error()
        # Assemble observations in-place to avoid temporary cat allocations each step.
        self.obs_buf[:, 0:3] = base_lin_vel_base * 2.0
        self.obs_buf[:, 3:6] = base_ang_vel_base * 0.25
        self.obs_buf[:, 6:9] = proj_gravity
        self.obs_buf[:, 9:12] = self.commands #* 2.0
        self.obs_buf[:, 12:24] = self.dof_pos - self.default_dof_pos
        self.obs_buf[:, 24:36] = self.dof_vel #* 0.05
        self.obs_buf[:, 36:48] = self.actions
        self.obs_buf[:, 48] = torch.sin(heading_error)
        self.obs_buf[:, 49] = torch.cos(heading_error)
    
    def _compute_rewards(self):
        obs = self.obs_buf
        actions = self.actions
        base_quat_inv = inv_quat(self.base_quat)
        base_lin_vel_base = transform_by_quat(self.base_lin_vel, base_quat_inv)
        orientation_error = 1.0 - transform_by_quat(self._up_vec, self.base_quat)[:, 2]
        heading_error = self._compute_heading_error()
        info = { # TODO: clean up info dict
            "base_lin_vel_base": base_lin_vel_base,
            "base_vel": self.base_lin_vel,
            "base_ang_vel": self.base_ang_vel,
            "base_pos": self.base_pos,
            "base_init_pos": self.base_init_pos,
            "orientation_error": orientation_error,
            "foot_contacts": self.foot_contacts,
            "dof_pos": self.dof_pos,
            "dof_vel": self.dof_vel,
            "default_dof_pos": self.default_dof_pos,
            "commands": self.commands,
            "base_height": self.base_pos[:, 2],
            "last_actions": self.last_actions,
            "reset_buf": self.reset_buf,
            "episode_length_buf": self.episode_length_buf,
            "max_episode_length": self.max_episode_length,
            "x_progress": self.x_progress, 
            "projected_gravity": transform_by_quat(self._gravity_vec, base_quat_inv),
            "foot_contacts": self.foot_contacts,
            "commands": self.commands,
            "heading_error": heading_error,
        }
        reward_out = self.reward_fn(obs, actions, info)
        lin_vel_x_rew = None
        if isinstance(reward_out, (tuple, list)):
            rewards = reward_out[0]
            if len(reward_out) > 1:
                lin_vel_x_rew = reward_out[1]
        else:
            rewards = reward_out

        if rewards.dim() > 1:
            rewards = rewards.squeeze(-1)

        if lin_vel_x_rew is None:
            lin_vel_x_rew = torch.zeros_like(rewards)
        elif isinstance(lin_vel_x_rew, torch.Tensor) and lin_vel_x_rew.dim() > 1:
            lin_vel_x_rew = lin_vel_x_rew.squeeze(-1)

        return rewards, lin_vel_x_rew
    
    # make_environment.py – _check_termination():
    def _check_termination(self):
        base_quat_inv = inv_quat(self.base_quat)
        proj_gravity = transform_by_quat(
            self._gravity_vec,
            base_quat_inv
        )
        roll_termination = (
            torch.abs(proj_gravity[:, 1]) > 0.342
        )
        pitch_termination = (
            torch.abs(proj_gravity[:, 0]) > 0.522
        )
        fall_termination = (
            self.base_pos[:, 2] < self.min_base_height
        )
        # Grace period
        grace_mask = self.episode_length_buf < 40

        # Nur Terminations zählen, die tatsächlich wirksam sind
        effective_roll = roll_termination & ~grace_mask
        effective_pitch = pitch_termination & ~grace_mask
        effective_fall = fall_termination & ~grace_mask

        termination = (
            effective_roll
            | effective_pitch
            | effective_fall
        )
        reasons = {
            "roll_termination_fraction":
                effective_roll.float().mean(),

            "pitch_termination_fraction":
                effective_pitch.float().mean(),

            "fall_termination_fraction":
                effective_fall.float().mean(),
        }

        return termination, reasons





    
    def set_commands(self, lin_vel_x, lin_vel_y, ang_vel_yaw):
        """
        Set target velocities for all environments.
        
        Args:
            lin_vel_x: Forward velocity (m/s)
            lin_vel_y: Lateral velocity (m/s)
            ang_vel_yaw: Yaw angular velocity (rad/s)
        """
        self.commands[:, 0] = lin_vel_x
        self.commands[:, 1] = lin_vel_y
        self.commands[:, 2] = ang_vel_yaw
        
    def _set_camera(self):
        '''Set camera positions and directions for recording'''
        # Elevated behind view (original)
        self._floating_camera_behind = self.scene.add_camera(
            pos=np.array([-1.5, 0.0, 5.0]),  # Behind and elevated
            lookat=np.array([0, 0, 0.1]),    # Looking at the robot
            fov=45,                          
            GUI=False,
            res=(720, 720),               
        )
        
        # Side view for feet
        if self.eval:
            self._floating_camera_side = self.scene.add_camera(
                pos=np.array([0.0, -2.5, 1.5]),     # Side view: to the right and lower
                lookat=np.array([0, 0, 0.3]),       # Looking at robot's center/legs
                fov=45,                              
                GUI=False,
                res=(720, 720),                      
            )
    def _compute_foot_diagnostics(self):
        contacts = self.foot_contacts > 0.5

        fl = contacts[:, 0]
        fr = contacts[:, 1]
        rl = contacts[:, 2]
        rr = contacts[:, 3]

        front_both_air = ~fl & ~fr
        rear_both_air = ~rl & ~rr

        num_contacts = contacts.sum(dim=1)

        flight = num_contacts == 0
        all_four_contact = num_contacts == 4

        diagonal_support = (
            (fl & rr & ~fr & ~rl)
            |
            (fr & rl & ~fl & ~rr)
        )

        return {
            "front_both_air": front_both_air.float().mean(),
            "rear_both_air": rear_both_air.float().mean(),
            "flight": flight.float().mean(),
            "all_four_contact": all_four_contact.float().mean(),
            "diagonal_support": diagonal_support.float().mean(),
        }
    
    def _compute_undesired_body_contacts(self):
        """Compute logging diagnostics for contacts of non-foot links."""

        contact_forces = self.robot.get_links_net_contact_force(envs_idx=None)
        contact_mask = contact_forces.norm(dim=-1) > 1.0

        foot_mask = torch.zeros(
            contact_mask.shape[1],
            device=contact_mask.device,
            dtype=torch.bool,
        )

        valid_foot_indices = [
            idx
            for idx in self.foot_link_indices
            if 0 <= idx < contact_mask.shape[1]
        ]

        if valid_foot_indices:
            foot_mask[valid_foot_indices] = True

        # [num_envs, num_links]
        undesired_mask = (
            contact_mask
            & ~foot_mask.unsqueeze(0)
        )

        return {
            # Durchschnittliche Anzahl kontaktierender Nicht-Fuß-Links
            "undesired_contact_count":
                undesired_mask.sum(dim=1).float().mean(),

            # Anteil der Environments mit mindestens einem
            # Nicht-Fuß-Kontakt
            "undesired_contact_fraction":
                undesired_mask.any(dim=1).float().mean(),
        }
    def _compute_heading_error(self):
        # Lokale +X-Achse des Roboters in Weltkoordinaten transformieren
        forward_world = transform_by_quat(
            self._forward_vec,
            self.base_quat,
        )

        # Aktuelle Yaw-/Heading-Richtung des Roboters
        current_heading = torch.atan2(
            forward_world[:, 1],
            forward_world[:, 0],
        )

        # Gewünschte Richtung: Welt +X = 0 rad
        desired_heading = torch.zeros_like(current_heading)

        # Signed angular error, sauber auf [-pi, pi] gewrappt
        heading_diff = current_heading - desired_heading

        heading_error = torch.atan2(
            torch.sin(heading_diff),
            torch.cos(heading_diff),
        )

        return heading_error