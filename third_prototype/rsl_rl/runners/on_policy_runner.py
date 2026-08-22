# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import time
import torch
from pathlib import Path
import wandb

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.utils import check_nan, resolve_callable
from rsl_rl.utils.logger import Logger


class OnPolicyRunner:
    """On-policy runner for reinforcement learning algorithms."""

    alg: PPO
    """The actor-critic algorithm."""

    def __init__(self, env: VecEnv, train_cfg: dict, 
                 log_dir: str | None = None, 
                 device: str = "cpu", 
                 vid_interval: int = 200,
                 video_dir: Path = Path("video.mp4"),) -> None:
        """Construct the runner, algorithm, and logging stack."""
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self.vid_interval = vid_interval
        self.video_dir = video_dir
        self.video_dir.mkdir(parents=True, exist_ok=True)

        # Setup multi-GPU training if enabled
        self._configure_multi_gpu()

        # Query observations from the environment for algorithm construction
        obs = self.env.get_observations()

        # Create the algorithm
        alg_class: type[PPO] = resolve_callable(self.cfg["algorithm"]["class_name"])  # type: ignore
        self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device)

        # Create the logger
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0
        self.counter = 0
        self.current_lin_vel_x = self.env.commands[0][0].item()
        #self.current_lin_vel_x = torch.tensor(0.0, device=rewards.device)


    def learn(self, 
              num_learning_iterations: int, 
              init_at_random_ep_len: bool = False, 
              #curriculum: bool = False, 
              curriculum_cfg: dict | None = None,
              ) -> None:
        
        """Run the learning loop for the specified number of iterations."""
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initialize the logging writer
        self.logger.init_logging_writer()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations

        """ Einfach hier immer den overflow buffer mäßig machen, oder so ne art liste und dann den avg aus der liste berechen? 
        Aber lsite berechnen immer extra steps 
        alle 20 nen neuen ist ja uach falsch -> maybe einfach auslagern in extra funktion und da gucken, hier nur extras speichernv"""


        avg_rew_over_episodes = 0.0
        for it in range(start_it, total_it):
            start = time.time()
            # Rollout
            tracking_sum = torch.tensor(0.0, device=self.device)
            tracking_count = 0  
            if it % self.vid_interval == 0 and self.logger.writer is not None:
                print("Recording video for evaluation...")
                cam = self.env.camera
                cam.start_recording()
            with torch.inference_mode():
                foot_sums = {
                    "front_both_air": torch.zeros((), device=self.device),
                    "rear_both_air": torch.zeros((), device=self.device),
                    "flight": torch.zeros((), device=self.device),
                    "all_four_contact": torch.zeros((), device=self.device),
                    "diagonal_support": torch.zeros((), device=self.device),
                    "undesired_contact_count": torch.zeros((), device=self.device),
                    "undesired_contact_fraction": torch.zeros((), device=self.device),
                    "roll_termination_fraction": torch.zeros((), device=self.device),
                    "pitch_termination_fraction": torch.zeros((), device=self.device),
                    "fall_termination_fraction": torch.zeros((), device=self.device),
                    "heading_error_abs_mean": torch.zeros((), device=self.device),
                    "heading_error_signed_mean": torch.zeros((), device=self.device),
                    "heading_error_abs_max": torch.zeros((), device=self.device)
                }   
                
                for _ in range(self.cfg["num_steps_per_env"]):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    tracking_sum += extras["lin_vel_x_rew"].mean()
                    tracking_count += 1
                    for key, value in extras["foot_diag"].items():
                        foot_sums[key] += value
                    #contact_sums["undesired_contacts"] += extras["undesired_contacts"]
                    # Check for NaN values from the environment
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards if RND is used (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    # Book keeping
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
                    if it % self.vid_interval == 0 and self.logger.writer is not None:
                        cam.render()

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)
            if it % self.vid_interval == 0 and self.logger.writer is not None:
                video_path = self.video_dir / f"go2_eval_{it}.mp4"
                cam.stop_recording(save_to_filename=str(video_path), fps=50)
                print(f"Video gespeichert unter: {video_path}")
                if wandb.run is not None and video_path.exists():  # Prüfen, ob WandB aktiv ist
                        wandb.log({
                            "Eval_Video": wandb.Video(str(video_path), fps=50, format="mp4")
                        }, step=it)  # Der step ordnet das Video dem richtigen Zeitpunkt zu
            rollout_tracking = tracking_sum / tracking_count
            if curriculum_cfg is not None and curriculum_cfg["enabled"]:
                self.update_curriculum(
                        curriculum_cfg,
                        it,
                        rollout_tracking
                    )
            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            num_steps = self.cfg["num_steps_per_env"]

            foot_means = {    key: value / self.cfg["num_steps_per_env"]
                        for key, value in foot_sums.items()}

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
                diagnostics=foot_means,
            )

            # Save model
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        # Save the final model after training and stop the logging writer
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()
        return self.current_lin_vel_x

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the models and training state to a given path and upload them if external logging is used."""
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        # Upload model to external logging services
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        """Load the models and training state from a given path.

        Args:
            path (str): Path to load the model from.
            load_cfg (dict | None): Optional dictionary that defines what models and states to load. If None, all
                models and states are loaded.
            strict (bool): Whether state_dict loading should be strict.
            map_location (str | None): Device mapping for loading the model.
        """
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> MLPModel:
        """Return the policy on the requested device for inference."""
        self.alg.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        return self.alg.get_policy().to(device)  # type: ignore

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the model to a Torch JIT file."""
        jit_model = self.alg.get_policy().as_jit()
        jit_model.to("cpu")

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        # Trace and save the model
        traced_model = torch.jit.script(jit_model)
        traced_model.save(save_path)

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export the model into an ONNX file."""
        onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
        onnx_model.to("cpu")
        onnx_model.eval()

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        # Trace and save the model
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),  # type: ignore
            save_path,
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=onnx_model.input_names,  # type: ignore
            output_names=onnx_model.output_names,  # type: ignore
        )

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Register a repository path whose git status should be logged."""
        self.logger.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.cfg["multi_gpu"] = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.cfg["multi_gpu"] = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-GPU configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # Initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # Set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)

    def update_curriculum(
        self,
        curriculum_cfg: dict,
        it: int,
        rewards: torch.Tensor
    ) -> None:
        """Update the curriculum based on reward performance."""

        enabled = curriculum_cfg.get("enabled", False)
        if not enabled:
            return

        max_lin_vel_x = curriculum_cfg.get("max_lin_vel_x", 1.0)
        delta_lin_vel_x = curriculum_cfg.get("delta_lin_vel_x", 0.1)
        threshold = curriculum_cfg.get("curriculum_threshold", 0.85)
        increase_anyway_thresh = curriculum_cfg.get("increase_anyway_threshold", 500)
        threshold_size = curriculum_cfg.get("threshold_size", 20)

        increase_anyway = (
            it != 0
            and increase_anyway_thresh
            and it % increase_anyway_thresh == 0
        )

        tracking_lin_vel_x = rewards[:, 0].mean() if rewards.ndim == 2 else rewards.mean()

        if not hasattr(self, "curriculum_buffer") or self.curriculum_buffer is None:
            self.curriculum_buffer = torch.zeros(
                threshold_size, dtype=torch.float32, device=rewards.device
            )
            self.counter = 0
            self.buffer_full = False

        if self.curriculum_buffer.numel() != threshold_size:
            self.curriculum_buffer = torch.zeros(
                threshold_size, dtype=torch.float32, device=rewards.device
            )
            self.counter = 0
            self.buffer_full = False

        self.curriculum_buffer[self.counter] = tracking_lin_vel_x.detach()
        self.counter = (self.counter + 1) % threshold_size
        if self.counter == 0:
            self.buffer_full = True

        # --- NEU: immer die aktuelle Zielgeschwindigkeit + rohen Tracking-Wert loggen ---
        if self.logger.writer is not None:
            self.logger.writer.add_scalar("Curriculum/lin_vel_x_cmd", self.current_lin_vel_x, it)
            self.logger.writer.add_scalar("Curriculum/tracking_lin_vel_x_raw", tracking_lin_vel_x.item(), it)
            self.logger.writer.add_scalar("Curriculum/buffer_full", float(self.buffer_full), it)

        if not self.buffer_full:
            return

        avg_tracking_lin_vel_x = self.curriculum_buffer.mean()

        # --- NEU: geglätteten Buffer-Mittelwert immer loggen, sobald der Buffer voll ist ---
        if self.logger.writer is not None:
            self.logger.writer.add_scalar(
                "Curriculum/avg_tracking_lin_vel_x", avg_tracking_lin_vel_x.item(), it
            )
            self.logger.writer.add_scalar(
                "Curriculum/threshold", threshold, it
            )
            self.logger.writer.add_scalar(
                "Curriculum/increase_anyway_flag", float(increase_anyway), it
            )

        if avg_tracking_lin_vel_x >= threshold or increase_anyway:
            current_cmd = self.current_lin_vel_x
            new_lin_vel_x = min(current_cmd + delta_lin_vel_x, max_lin_vel_x)
            self.current_lin_vel_x = new_lin_vel_x
            self.env.set_commands(lin_vel_x=new_lin_vel_x, lin_vel_y=0.0, ang_vel_yaw=0.0)

            # --- NEU: expliziter Marker für den Curriculum-Sprung selbst ---
            if self.logger.writer is not None:
                self.logger.writer.add_scalar("Curriculum/step_event", 1.0, it)
                self.logger.writer.add_scalar(
                    "Curriculum/step_triggered_by_increase_anyway", float(increase_anyway), it
                )

            print(
                f"Updating curriculum at iteration {it}: "
                f"avg_tracking_lin_vel_x={avg_tracking_lin_vel_x.item():.4f} "
                f">= {threshold}, setting lin_vel_x to {new_lin_vel_x:.4f}"
            )
            # reset Buffer 
            self.counter = 0
            self.buffer_full = False
