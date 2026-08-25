from __future__ import annotations

import argparse
import logging
import re
import statistics
import sys
from pathlib import Path

import genesis as gs
import torch
import wandb

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

from config import build_configs
from make_environment import Go2WalkingEnv
from reward import Rewards


def build_reward_fn(
        reward_cfg: dict = None
) -> Rewards:
    scales = {k: v for k, v in reward_cfg.items() if k != "tracking_sigma"}
    return Rewards(tracking_sigma=reward_cfg["tracking_sigma"], scales=scales)


def build_env(device: str, 
              num_envs: int, 
              show_viewer: bool, 
              lin_vel_x: float = None, 
              use_terrain: bool = None, 
              episode_length_s: float = None, 
              reward_cfg: dict = None,
              command_range: dict = None, 
              command_range_allowed: bool = False
              ) -> Go2WalkingEnv:
    reward_fn = build_reward_fn(reward_cfg)
    env = Go2WalkingEnv(
        num_envs=num_envs,
        device=device,
        show_viewer=show_viewer,
        use_terrain= use_terrain,
        episode_length_s=episode_length_s,
        min_up_dot=0.1,
        reward_fn=reward_fn,
        min_base_height=0.22,
        command_range_allowed=command_range_allowed
    )
    env.command_range = command_range
    return env

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
    runner,
    env,
    video_path,
    device,
    eval_steps=800,
    iteration=0,
    stochastic=False,
):
    policy = runner.get_inference_policy(device=device)
    policy.eval()

    # WICHTIG:
    # Robot auf dieselbe Initialpose setzen wie beim Training
    env.reset()

    # Erst NACH reset() die Observation holen
    obs_td = env.get_observations()

    cam = env.camera
    cam.start_recording()

    with torch.no_grad():
        for _ in range(eval_steps):

            actions = policy(
                obs_td,
                stochastic_output=stochastic,
            )

            obs_td, rewards, dones, extras = env.step(actions)

            cam.render()

            if bool(dones[0].item()):
                break

    cam.stop_recording(
        save_to_filename=str(video_path),
        fps=50,
    )

    if wandb.run is not None:  # Prüfen, ob WandB aktiv ist
        wandb.log({
            "Eval_Video": wandb.Video(str(video_path), fps=50, format="mp4")
        }, step=iteration)  # Der step ordnet das Video dem richtigen Zeitpunkt zu

def get_config(config_name: str) -> dict:
    configs = build_configs(config_name)
    if not configs:
        raise ValueError(f"Keine Konfiguration für '{config_name}' gefunden.")
    return configs.get("Training_Config", {}), configs.get("Reward_Config", {}), configs.get("Curriculum_Config", {}), configs.get("Environment_Config", {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="config_A", help="Name of the configuration file (without .py extension)")
    train_cfg, reward_cfg, curriculum_cfg, env_cfg = get_config(parser.parse_args().config_name)

    torch.manual_seed(env_cfg.get("seed", 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    env = build_env(device=device, 
                    num_envs=env_cfg.get("num_envs", 4096), 
                    show_viewer=False, 
                    lin_vel_x=curriculum_cfg["start_lin_vel_x"], 
                    use_terrain=env_cfg.get("use_terrain", True),
                    episode_length_s=env_cfg.get("episode_length_s", 30.0), 
                    reward_cfg=reward_cfg, 
                    command_range=env_cfg.get("command_range", {
                        "lin_vel_x": (0.0, 1.0),
                        "lin_vel_y": (0.0, 0.0),
                        "ang_vel_yaw": (0.0, 0.0)
                    }),
                    command_range_allowed=env_cfg.get("command_range_allowed", False)
                    )
    obs = env.get_observations()
    #print("Observation keys:", list(obs.keys()))
    #print("Policy obs shape:", obs["policy"].shape)

    logs_root = project_root / "logs"
    run_name, log_dir = reserve_run_version(logs_root=logs_root, base_run_name="go2_genesis_rsl_rl"+f"_{parser.parse_args().config_name}")
    
    
    wandb.init(
            project="Genesis_Dog_Walking",  
            name=run_name,      
            config={                         
                "Curriculum": curriculum_cfg,
                "Reward": reward_cfg,
                "Train_cfg": train_cfg,
                "Environement_cfg": env_cfg
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
    print(f"Start command lin_vel_x: {curriculum_cfg['start_lin_vel_x']}")
    #print(f"Num envs: {env_cfg.get("num_envs", 4096)}")
    print(f"Total learning iterations: {train_cfg.get('num_learning_iterations', 4000)}")
    if curriculum_cfg["enabled"]:
        print("Curriculum aktiv:", curriculum_cfg)
    print("Reward configuration:", reward_cfg)
    print("Train_cfg: ", train_cfg)
    
    final_vel = runner.learn(num_learning_iterations=train_cfg.get('num_learning_iterations', 4000), 
                 #curriculum=True, 
                 curriculum_cfg=curriculum_cfg, 
                 init_at_random_ep_len=True)

    
    latest_checkpoint = load_latest_checkpoint(runner, log_dir=log_dir, device=device)
    eval_env = build_env(device=device, 
                         num_envs=1, 
                         show_viewer=False, 
                         lin_vel_x=final_vel,
                         use_terrain=env_cfg.get("use_terrain", True),
                         episode_length_s=env_cfg.get("episode_length_s", 30.0), 
                         reward_cfg=reward_cfg,
                         command_range_allowed=False)
    video_dir = project_root / "video" / run_name
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"go2_eval_{latest_checkpoint.stem}.mp4"
    tmp = env_cfg.get("command_range", {}).get("lin_vel_x", [0.0, 1.0])
    current = tmp[0]
    while current <= tmp[1]: 
        eval_env.set_commands(lin_vel_x=current, lin_vel_y=0.0, ang_vel_yaw=0.0)
        print(f"Evaluating with command lin_vel_x: {current}")
        make_eval_video(runner, eval_env, video_path.with_name(f"go2_eval_{latest_checkpoint.stem}_lin_vel_x_{current:.2f}.mp4"), device=device)
        current += 0.1
    


if __name__ == "__main__":
    # Quick debug mode to diagnose reward values
    if "--debug" in sys.argv:
        print("Running in DEBUG mode with only 10 iterations...")
        main(num_learning_iterations=10, make_video=False)
    else:
        main()
