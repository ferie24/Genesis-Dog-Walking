import torch
import math
import numpy as np
from pathlib import Path
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.entities.rigid_entity import RigidEntity
from genesis.engine.scene import Scene
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver


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
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_envs = num_envs
        self.show_viewer = show_viewer
        self.use_terrain = use_terrain
        # Termination tolerances; loosen to avoid instant resets when touching down
        self.min_base_height = min_base_height
        self.min_up_dot = min_up_dot
        
        # Time and episode settings
        self.dt = 0.02  # 50Hz control frequency
        self.max_episode_length = math.ceil(episode_length_s / self.dt)
        
        # Robot configuration
        self.num_dof = 12  # 12 actuated joints (3 per leg)
        self.num_actions = 12
        # 48 = 3 base lin vel + 3 base ang vel + 3 projected gravity + 3 commands + 12 (dof pos) + 12 (dof vel) + 12 (actions)
        self.num_obs = 48  # Robot state observations
        
        # Action and observation buffers
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        
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
        
        # PD controller gains
        self.kp = 20.0
        self.kd = 0.5
        
        self.reward_fn = reward_fn if reward_fn is not None else lambda obs, actions, info: torch.zeros(self.num_envs, device=self.device)

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
        rgb, depth, segmentation, normal = self.camera.render(depth=True, segmentation=True, normal=True)
        return self.camera

        
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
                n_rendered_envs=1,
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
        for solver in self.scene.sim.solvers:
            if isinstance(solver, RigidSolver):
                self.rigid_solver = solver
                break
    
    def _add_terrain(self):
        """Add terrain (flat plane or complex terrain)"""
        if self.use_terrain:
            # Complex terrain with height variations
            self.terrain = self.scene.add_entity(
                gs.morphs.Terrain(
                    n_subterrains=(5, 5),
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                    subterrain_size=(5.0, 5.0),
                    subterrain_types=['flat', 'random_uniform'],
                    randomize=False,
                ),
            )
            self.base_init_pos = torch.tensor([2.5, 12.5, 0.42], device=self.device)
        else:
            # Simple flat plane
            self.plane = self.scene.add_entity(gs.morphs.Plane())
            self.base_init_pos = torch.tensor([0.0, 0.0, 0.42], device=self.device)
    
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
        
    def reset(self, env_ids=None):
        """
        Reset the environment.
        
        Args:
            env_ids: Optional list of environment indices to reset. If None, reset all.
        
        Returns:
            observations: Current observations after reset
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Reset episode lengths
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        
        # Reset robot to initial pose
        for env_id in env_ids:
            # Set base position and orientation
            self.robot.set_pos(self.base_init_pos.cpu().numpy(), zero_velocity=True, envs_idx=[env_id.item()])
            self.robot.set_quat(np.array([0, 0, 0, 1]), zero_velocity=True, envs_idx=[env_id.item()])
            
            # Set joint positions to default standing pose
            self.robot.set_dofs_position(
                self.default_dof_pos.cpu().numpy(),
                self.dof_indices,
                zero_velocity=True,
                envs_idx=[env_id.item()]
            )
        
        # Reset action buffers
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        
        # Get new observations
        self._update_state()
        self._compute_observations()
        
        return self.obs_buf
    
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
        self.actions = torch.clip(actions, -1.0, 1.0)
        
        # Apply actions to robot
        target_dof_pos = self.default_dof_pos + self.actions * 0.25  # Scale actions
        
        for i in range(self.num_envs):
            self.robot.control_dofs_position(
                target_dof_pos[i].cpu().numpy(),
                self.dof_indices,
                envs_idx=[i]
            )
        
        # Step simulation
        self.scene.step()
        
        # Update state
        self._update_state()
        
        # Compute observations and rewards
        self._compute_observations()
        rewards = self._compute_rewards()
        
        # Update episode length
        self.episode_length_buf += 1
        
        # Check for resets
        self.reset_buf = self.episode_length_buf >= self.max_episode_length
        self.reset_buf |= self._check_termination()
        
        # Auto-reset finished environments
        if self.reset_buf.any():
            self.reset(self.reset_buf.nonzero(as_tuple=False).flatten())
        
        self.last_actions[:] = self.actions[:]
        
        return self.obs_buf, rewards, self.reset_buf, {}
    
    def _update_state(self):
        """Update all state buffers from simulation"""
        for i in range(self.num_envs):
            # Get base state
            base_pos = self.robot.get_pos(envs_idx=[i])
            base_quat = self.robot.get_quat(envs_idx=[i])
            base_vel = self.robot.get_vel(envs_idx=[i])
            # Guard against empty velocity arrays coming from the backend
            if base_vel is None or len(base_vel) == 0 or len(base_vel[0]) < 6:
                base_vel_np = np.zeros(6, dtype=float)
            else:
                base_vel_np = np.array(base_vel[0], dtype=float)
                if base_vel_np.shape[-1] < 6:
                    base_vel_np = np.pad(base_vel_np, (0, 6 - base_vel_np.shape[-1]))
            
            self.base_pos[i] = torch.tensor(base_pos[0], device=self.device)
            self.base_quat[i] = torch.tensor(base_quat[0], device=self.device)
            self.base_lin_vel[i] = torch.tensor(base_vel_np[:3], device=self.device)
            self.base_ang_vel[i] = torch.tensor(base_vel_np[3:6], device=self.device)
            
            # Get joint states
            dof_pos = self.robot.get_dofs_position(self.dof_indices, envs_idx=[i])
            dof_vel = self.robot.get_dofs_velocity(self.dof_indices, envs_idx=[i])
            
            self.dof_pos[i] = torch.tensor(dof_pos, device=self.device)
            self.dof_vel[i] = torch.tensor(dof_vel, device=self.device)
            
            # Get foot contacts
            contact_forces = self.robot.get_links_net_contact_force(envs_idx=[i])
            if contact_forces is None or contact_forces.numel() == 0:
                contact_forces = torch.zeros((1, len(self.foot_link_indices), 3), device=self.device)
            for j, link_idx in enumerate(self.foot_link_indices):
                if link_idx >= contact_forces.shape[1]:
                    force_vec = torch.zeros(3, device=self.device)
                else:
                    force_vec = contact_forces[0, link_idx]
                self.foot_contacts[i, j] = (torch.norm(force_vec) > 1.0).float()
    
    def _compute_observations(self):
        """Compute observations from current state"""
        # Get projected gravity (orientation information)
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        proj_gravity = transform_by_quat(gravity_vec.repeat(self.num_envs, 1), inv_quat(self.base_quat))
        
        # Compute velocity in base frame
        base_lin_vel_base = transform_by_quat(self.base_lin_vel, inv_quat(self.base_quat))
        base_ang_vel_base = transform_by_quat(self.base_ang_vel, inv_quat(self.base_quat))
        
        # Assemble observations
        obs = torch.cat([
            base_lin_vel_base * 2.0,           # 3
            base_ang_vel_base * 0.25,          # 3
            proj_gravity,                      # 3
            self.commands * 2.0,               # 3
            self.dof_pos - self.default_dof_pos,  # 12
            self.dof_vel * 0.05,               # 12
            self.actions,                      # 12
        ], dim=-1)
        #print(obs)
        self.obs_buf[:] = obs
    
    def _compute_rewards(self):
        obs = self.obs_buf
        actions = self.actions
        up_vec = torch.tensor(
            [0.0, 0.0, 1.0],  # note the floats
            device=self.device,
            dtype=self.base_quat.dtype,
        ).repeat(self.num_envs, 1)

        info = {
            "base_vel": self.base_lin_vel,
            "orientation_error": 1.0 - transform_by_quat(up_vec, self.base_quat)[:, 2],
            "foot_contacts": self.foot_contacts,
            "dof_pos": self.dof_pos,
            "dof_vel": self.dof_vel,
            "commands": self.commands,
            "base_height": self.base_pos[:, 2],
            "foot_vel": self.robot.get_links_vel(self.foot_link_indices),
            "base_height": self.base_pos[:, 2],
            "commands": self.commands,
        }
        return self.reward_fn(obs, actions, info)
    
    def _check_termination(self):
        """Check if any environments should terminate"""
        # Terminate if robot falls over
        base_height = self.base_pos[:, 2]
        fall_termination = base_height < self.min_base_height
        
        # Terminate if robot tips over too much
        up_vec = transform_by_quat(
            torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1),
            self.base_quat
        )
        # Tilt termination (dot with world up); higher threshold is stricter
        tip_termination = up_vec[:, 2] < self.min_up_dot
        
        return fall_termination | tip_termination
    
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
    