from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import genesis as gs
import torch

project_root = Path(__file__).resolve().parent
local_rsl_rl = project_root / "rsl_rl"
if str(local_rsl_rl) not in sys.path:
    sys.path.insert(0, str(local_rsl_rl))

from make_environment import Go2WalkingEnv
from rsl_rl.runners import OnPolicyRunner


backend = gs.gpu if torch.cuda.is_available() else gs.cpu
try:
    gs.init(logging_level=logging.WARNING, backend=backend)
except Exception as exc:
    print(f"Genesis war bereits initialisiert: {exc}")


def go2_reward(obs, actions, info):
    target_vx = info["commands"][:, 0]
    current_vx = info["base_lin_vel_base"][:, 0]
    tracking = torch.exp(-((current_vx - target_vx) ** 2) / 0.25)

    upright = (1.0 - info["orientation_error"]).clamp(0.0, 1.0)
    action_penalty = 0.001 * (actions ** 2).sum(dim=-1)
    side_penalty = 0.05 * torch.abs(info["base_lin_vel_base"][:, 1])

    return tracking + 0.2 * upright - action_penalty - side_penalty


def build_env(device: str, num_envs: int, show_viewer: bool) -> Go2WalkingEnv:
    env = Go2WalkingEnv(
        num_envs=num_envs,
        device=device,
        show_viewer=show_viewer,
        use_terrain=False,
        episode_length_s=20.0,
        min_up_dot=0.05,
        reward_fn=go2_reward,
        min_base_height=0.15,
    )
    env.set_commands(lin_vel_x=0.4, lin_vel_y=0.0, ang_vel_yaw=0.0)
    return env


def build_train_cfg() -> dict:
    return {
        "run_name": "go2_genesis_rsl_rl",
        "logger": "tensorboard",
        "num_steps_per_env": 128,
        "save_interval": 25,
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "learning_rate": 5e-4,
            "max_grad_norm": 0.5,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "normalize_advantage_per_mini_batch": False,
            "optimizer": "adam",
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 256],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.8,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 256],
            "activation": "elu",
            "obs_normalization": True,
        },
    }


def _iter_from_name(path_obj: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path_obj.name)
    return int(match.group(1)) if match else -1


def load_latest_checkpoint(runner: OnPolicyRunner, log_dir: Path, device: str) -> Path:
    checkpoint_paths = list(log_dir.glob("model_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"Kein Checkpoint in {log_dir} gefunden.")

    latest_checkpoint = sorted(checkpoint_paths, key=_iter_from_name)[-1]
    print(f"Lade Checkpoint: {latest_checkpoint.name}")

    with torch.inference_mode():
        runner.load(str(latest_checkpoint), map_location=device)

    return latest_checkpoint


def make_eval_video(
    runner: OnPolicyRunner,
    env: Go2WalkingEnv,
    video_path: Path,
    device: str,
    eval_steps: int = 800,
) -> None:
    policy = runner.get_inference_policy(device=device)
    policy.eval()

    obs_td = env.get_observations()
    cam = env.camera
    cam.start_recording()

    with torch.no_grad():
        for _ in range(eval_steps):
            actions = policy(obs_td, stochastic_output=False)
            actions = torch.clamp(actions, -1.0, 1.0)
            obs_td, rewards, dones, extras = env.step(actions)
            cam.render()
            if bool(dones[0].item()):
                break

    cam.stop_recording(save_to_filename=str(video_path), fps=50)
    print(f"Video gespeichert unter: {video_path}")


def main(num_learning_iterations: int = 3000, make_video: bool = True) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 512

    env = build_env(device=device, num_envs=num_envs, show_viewer=False)
    obs = env.get_observations()
    print("Observation keys:", list(obs.keys()))
    print("Policy obs shape:", obs["policy"].shape)

    train_cfg = build_train_cfg()
    log_dir = project_root / "logs" / "rsl_rl_go2"
    log_dir.mkdir(parents=True, exist_ok=True)

    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=str(log_dir),
        device=device,
    )
    print("Runner ist bereit.")
    print(f"Logs: {log_dir}")

    runner.learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=True)

    if make_video:
        latest_checkpoint = load_latest_checkpoint(runner, log_dir=log_dir, device=device)
        eval_env = build_env(device=device, num_envs=1, show_viewer=False)
        video_dir = project_root / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"go2_eval_{latest_checkpoint.stem}.mp4"
        make_eval_video(runner, eval_env, video_path, device=device)


if __name__ == "__main__":
    main()
