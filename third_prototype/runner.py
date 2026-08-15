from __future__ import annotations
import argparse
import wandb
import logging
import re
import statistics
import sys
from pathlib import Path

import genesis as gs
import torch

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import rsl_rl
print(f"[DEBUG] rsl_rl geladen von: {rsl_rl.__file__}")

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
    "tracking_ang_vel": 1.0, #1.0,
    "lin_vel_z": -1.0,
    "lin_vel_y": -5.0,
    "action_rate": -0.005,
    "similar_to_default": -0.1,# -0.1,   # stärker bestrafen, wenn er in Default-Pose "einfriert"
    "sideway_movement":  -1.0,#-1.0,
    "tracking_sigma": 0.3,        # engerer Tracking-Bonus, belohnt genaueres Geschwindigkeitsmatching statt "gut genug" bei 0
    "x_progress": 0.0,             # stärkerer Anreiz für tatsächlichen Vorwärtsfortschritt
}


SEED = 1
USE_TERRAIN = True
EPISODE_LENGTH_S = 30.0
NUM_ENVS = 4096
DEFAULT_NUM_LEARNING_ITERATIONS = 4000

CURRICULUM_CFG = {
    "enabled": True,
    "start_lin_vel_x": 0.2,
    "max_lin_vel_x": 1.0,
    "delta_lin_vel_x": 0.05,
    "curriculum_threshold": 0.85,
    "increase_anyway_threshold": 5000,
    "threshold_size": 30
}


def build_reward_fn() -> Rewards:
    scales = {k: v for k, v in REWARD_CFG.items() if k != "tracking_sigma"}
    return Rewards(tracking_sigma=REWARD_CFG["tracking_sigma"], scales=scales)


def build_env(device: str, num_envs: int, show_viewer: bool, lin_vel_x: float = CURRICULUM_CFG["start_lin_vel_x"]) -> Go2WalkingEnv:
    reward_fn = build_reward_fn()
    env = Go2WalkingEnv(
        num_envs=num_envs,
        device=device,
        show_viewer=show_viewer,
        use_terrain=USE_TERRAIN,
        episode_length_s=EPISODE_LENGTH_S,
        min_up_dot=0.1,
        reward_fn=reward_fn,
        min_base_height=0.22,
    )
    env.set_commands(lin_vel_x=lin_vel_x, lin_vel_y=0.0, ang_vel_yaw=0.0)
    return env


def build_train_cfg(run_name: str) -> dict:
    return {
        "run_name": run_name,
        "logger": "tensorboard",
        "num_steps_per_env": 96,
        "save_interval": 200,
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
            "entropy_coef": 0.001,
            "learning_rate": 3e-4,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "normalize_advantage_per_mini_batch": False,
            "optimizer": "adamw",
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
                "learn_std": True,
                "std_range": [0.05, 1.0],
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
    iteration: int = 0
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

    if wandb.run is not None:  # Prüfen, ob WandB aktiv ist
        wandb.log({
            "Eval_Video": wandb.Video(str(video_path), fps=50, format="mp4")
        }, step=iteration)  # Der step ordnet das Video dem richtigen Zeitpunkt zu



def main(num_learning_iterations: int = DEFAULT_NUM_LEARNING_ITERATIONS, make_video: bool = True) -> None:
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = NUM_ENVS
    

    parser = argparse.ArgumentParser()
    #parser.add_argument("--run_name", type=str, default="go2_test")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--action_rate_penalty", type=float, default=-0.005)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    env = build_env(device=device, num_envs=num_envs, show_viewer=False, lin_vel_x=CURRICULUM_CFG["start_lin_vel_x"])
    obs = env.get_observations()
    print("Observation keys:", list(obs.keys()))
    print("Policy obs shape:", obs["policy"].shape)

    logs_root = project_root / "logs"
    run_name, log_dir = reserve_run_version(logs_root=logs_root, base_run_name="go2_genesis_rsl_rl")
    train_cfg = build_train_cfg(run_name=run_name)
    # Nutze args für deine Configs
    wandb.init(
            project="Genesis_Dog_Walking",   # Name deines Projekts im Dashboard
            name=train_cfg["run_name"] + "_Server",      # Name des aktuellen Laufs (z.B. v008)
            config={                         # Speichert alle Parameter fürs spätere Vergleichen!
                "Curriculum": CURRICULUM_CFG,
                "Reward": REWARD_CFG,
                "Train_cfg": train_cfg
            },
            sync_tensorboard=True,           # Der magische Trick! Zieht sich alle TB-Daten.
            dir=project_root / "logs"
        )
    #train_cfg["algorithm"]["learning_rate"] = args.lr
    #REWARD_CFG["action_rate"] = args.action_rate_penalty
    #video_dir = project_root / "video" / run_name
    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=str(log_dir),
        device=device,
        vid_interval=200,
        video_dir= log_dir / "videos"
    )
    print("Runner ist bereit.")
    print(f"Run name: {run_name}")
    print(f"Logs: {log_dir}")
    print(f"Start command lin_vel_x: {CURRICULUM_CFG['start_lin_vel_x']}")
    print(f"Seed: {SEED}")
    print(f"Num envs: {num_envs}")
    print(f"Total learning iterations: {num_learning_iterations}")
    if CURRICULUM_CFG["enabled"]:
        print("Curriculum aktiv:", CURRICULUM_CFG)
    print("Reward configuration:", REWARD_CFG)
    print("Train_cfg: ", train_cfg)

    runner.learn(num_learning_iterations=num_learning_iterations, curriculum=True, curriculum_cfg=CURRICULUM_CFG, init_at_random_ep_len=True)

    
    latest_checkpoint = load_latest_checkpoint(runner, log_dir=log_dir, device=device)
    eval_env = build_env(device=device, num_envs=1, show_viewer=False, lin_vel_x=CURRICULUM_CFG["start_lin_vel_x"])
    video_dir = project_root / "video" / run_name
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"go2_eval_{latest_checkpoint.stem}.mp4"
    make_eval_video(runner, eval_env, video_path, device=device)


if __name__ == "__main__":
    # Quick debug mode to diagnose reward values
    if "--debug" in sys.argv:
        print("Running in DEBUG mode with only 10 iterations...")
        main(num_learning_iterations=10, make_video=False)
    else:
        main()
