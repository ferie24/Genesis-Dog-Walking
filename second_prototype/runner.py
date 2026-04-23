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


from rsl_rl.runners import OnPolicyRunner


backend = gs.gpu if torch.cuda.is_available() else gs.cpu
try:
    gs.init(logging_level=logging.WARNING, backend=backend)
except Exception as exc:
    print(f"Genesis war bereits initialisiert: {exc}")

from make_environment import Go2WalkingEnv
from reward import Rewards

REWARD_CFG = {
    "tracking_lin_vel_x": 1.0,
    "tracking_ang_vel": 1.0,
    "lin_vel_z": -1.0,
    "lin_vel_y": -5.0,
    "action_rate": -0.005,
    "similar_to_default": -0.1,
    "sideway_movement": -1.0,
    "tracking_sigma": 0.3,
}

START_LIN_VEL_X = 0.15
EVAL_LIN_VEL_X = 0.2

CURRICULUM_CFG = {
    "enabled": True,
    "start_lin_vel_x": 0.15,
    "max_lin_vel_x": 0.8,
    "delta_lin_vel_x": 0.05,
    "stage_iterations": 150,
}


def build_reward_fn() -> Rewards:
    scales = {k: v for k, v in REWARD_CFG.items() if k != "tracking_sigma"}
    return Rewards(tracking_sigma=REWARD_CFG["tracking_sigma"], scales=scales)


def build_env(device: str, num_envs: int, show_viewer: bool, lin_vel_x: float = START_LIN_VEL_X) -> Go2WalkingEnv:
    reward_fn = build_reward_fn()
    env = Go2WalkingEnv(
        num_envs=num_envs,
        device=device,
        show_viewer=show_viewer,
        use_terrain=False,
        episode_length_s=20.0,
        min_up_dot=0.1,
        reward_fn=reward_fn,
        min_base_height=0.18,
    )
    env.set_commands(lin_vel_x=lin_vel_x, lin_vel_y=0.0, ang_vel_yaw=0.0)
    return env


def train_with_speed_curriculum(
    runner: OnPolicyRunner,
    env: Go2WalkingEnv,
    total_iterations: int,
    cfg: dict,
) -> None:
    """Train in stages while gradually increasing target forward speed."""
    if not cfg.get("enabled", False):
        runner.learn(num_learning_iterations=total_iterations, init_at_random_ep_len=True)
        return

    speed = float(cfg["start_lin_vel_x"])
    speed_max = float(cfg["max_lin_vel_x"])
    speed_delta = float(cfg["delta_lin_vel_x"])
    stage_iterations = int(cfg["stage_iterations"])
    stage_iterations = max(1, stage_iterations)

    completed = 0
    stage_idx = 0
    while completed < total_iterations:
        remaining = total_iterations - completed
        stage_iters = min(stage_iterations, remaining)

        env.set_commands(lin_vel_x=speed, lin_vel_y=0.0, ang_vel_yaw=0.0)
        print(
            f"Curriculum Stage {stage_idx:02d} | lin_vel_x={speed:.2f} | "
            f"iterations={stage_iters}"
        )

        runner.learn(
            num_learning_iterations=stage_iters,
            init_at_random_ep_len=(stage_idx == 0),
        )

        completed += stage_iters
        speed = min(speed_max, speed + speed_delta)
        stage_idx += 1

def build_train_cfg(run_name: str) -> dict:
    return {
        "run_name": run_name,
        "logger": "tensorboard",
        "num_steps_per_env": 256,
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
            "learning_rate": 3e-4,
            "max_grad_norm": 0.5,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.005,
            "normalize_advantage_per_mini_batch": False,
            "optimizer": "adam",
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
        },
    }


def _iter_from_name(path_obj: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path_obj.name)
    return int(match.group(1)) if match else -1


def reserve_run_version(logs_root: Path, base_run_name: str) -> tuple[str, Path]:
    """Reserve a unique run name and directory by incrementing a numeric suffix."""
    logs_root.mkdir(parents=True, exist_ok=True)

    version = 1
    while True:
        run_name = f"{base_run_name}_v{version:03d}"
        run_dir = logs_root / run_name
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_name, run_dir
        version += 1


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

    env = build_env(device=device, num_envs=num_envs, show_viewer=False, lin_vel_x=START_LIN_VEL_X)
    obs = env.get_observations()
    print("Observation keys:", list(obs.keys()))
    print("Policy obs shape:", obs["policy"].shape)

    logs_root = project_root / "logs"
    run_name, log_dir = reserve_run_version(logs_root=logs_root, base_run_name="go2_genesis_rsl_rl")
    train_cfg = build_train_cfg(run_name=run_name)

    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=str(log_dir),
        device=device,
    )
    print("Runner ist bereit.")
    print(f"Run name: {run_name}")
    print(f"Logs: {log_dir}")
    print(f"Start command lin_vel_x: {START_LIN_VEL_X}")
    if CURRICULUM_CFG["enabled"]:
        print("Curriculum aktiv:", CURRICULUM_CFG)

    train_with_speed_curriculum(
        runner=runner,
        env=env,
        total_iterations=num_learning_iterations,
        cfg=CURRICULUM_CFG,
    )

    if make_video:
        latest_checkpoint = load_latest_checkpoint(runner, log_dir=log_dir, device=device)
        eval_env = build_env(device=device, num_envs=1, show_viewer=False, lin_vel_x=EVAL_LIN_VEL_X)
        video_dir = project_root / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"go2_eval_{latest_checkpoint.stem}.mp4"
        make_eval_video(runner, eval_env, video_path, device=device)


if __name__ == "__main__":
    main()
