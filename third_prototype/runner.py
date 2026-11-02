from __future__ import annotations

import argparse
from copy import deepcopy
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import wandb

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import rsl_rl

print(f"[DEBUG] rsl_rl geladen von: {rsl_rl.__file__}")

from rsl_rl.runners import OnPolicyRunner

from config import build_configs
from reward import Rewards

if TYPE_CHECKING:
    from make_environment import Go2WalkingEnv


def initialize_simulation(seed: int):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import genesis as gs

    gs.init(
        seed=seed,
        logging_level=logging.WARNING,
        backend=gs.gpu if torch.cuda.is_available() else gs.cpu,
        performance_mode=True,
    )


def build_reward_fn(reward_cfg: dict = None) -> Rewards:
    scales = {k: v for k, v in reward_cfg.items() if k != "tracking_sigma"}
    return Rewards(
        tracking_sigma=reward_cfg["tracking_sigma"],
        scales=scales,
        tracking_sigma_ang_vel=reward_cfg.get("tracking_sigma_ang_vel", None),
    )


def build_env(
    device: str,
    num_envs: int,
    show_viewer: bool,
    lin_vel_x: float = None,
    use_terrain: bool = None,
    episode_length_s: float = None,
    reward_cfg: dict = None,
    command_range: dict = None,
    command_range_allowed: bool = False,
    terminate_on_torso_contact: bool = False,
) -> Go2WalkingEnv:
    from make_environment import Go2WalkingEnv

    reward_fn = build_reward_fn(reward_cfg)
    env = Go2WalkingEnv(
        num_envs=num_envs,
        device=device,
        show_viewer=show_viewer,
        use_terrain=use_terrain,
        episode_length_s=episode_length_s,
        min_up_dot=0.1,
        reward_fn=reward_fn,
        min_base_height=0.22,
        command_range_allowed=command_range_allowed,
        terminate_on_torso_contact=terminate_on_torso_contact,
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
    speed_x=0.0,
    speed_y=0.0,
    speed_yaw=0.0,
):
    env.set_commands(lin_vel_x=speed_x, lin_vel_y=speed_y, ang_vel_yaw=speed_yaw)
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

            env._update_follow_camera()
            cam.render()

            if bool(dones[0].item()):
                break

    cam.stop_recording(
        save_to_filename=str(video_path),
        fps=50,
    )

    if wandb.run is not None:  # Prüfen, ob WandB aktiv ist
        wandb.log(
            {"Eval_Video": wandb.Video(str(video_path), fps=50, format="mp4")},
            step=iteration,
        )  # Der step ordnet das Video dem richtigen Zeitpunkt zu


def get_config(config_name: str) -> dict:
    configs = build_configs(config_name)
    if not configs:
        raise ValueError(f"Keine Konfiguration für '{config_name}' gefunden.")
    return (
        configs.get("Training_Config", {}),
        configs.get("Reward_Config", {}),
        configs.get("Curriculum_Config", {}),
        configs.get("Environment_Config", {}),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Konfiguration für einen neuen Lauf (Standard config_A); bei Resume aus dem Run-Namen abgeleitet.",
    )
    parser.add_argument("--iterations", type=int, help="Anzahl auszuführender Iterationen; bei Resume ZUSÄTZLICHE Iterationen.")
    parser.add_argument("--resume-from", type=Path, help="Training mit allen Lernzuständen dieses Checkpoints fortsetzen.")
    parser.add_argument("--saved-config", type=Path, help="Gespeicherte config.yaml für Resume ausdrücklich auswählen.")
    parser.add_argument("--dry-run", action="store_true", help="Konfiguration und Iterationsbereich prüfen, ohne Simulator oder Training.")
    parser.add_argument("--video-interval", type=int, help="Video alle N Iterationen; 0 deaktiviert Trainingsvideos.")
    parser.add_argument("--skip-final-eval", action="store_true", help="Abschließende Eval-Videos überspringen.")
    args = parser.parse_args(argv)
    if args.iterations is not None and args.iterations <= 0:
        parser.error("--iterations muss positiv sein.")
    if args.video_interval is not None and args.video_interval < 0:
        parser.error("--video-interval muss nicht negativ sein.")
    if args.resume_from is not None:
        args.resume_from = args.resume_from.expanduser().resolve()
        if not args.resume_from.is_file():
            parser.error(f"Checkpoint fehlt: {args.resume_from}")
        if args.iterations is None:
            parser.error("Bei --resume-from die ZUSÄTZLICHEN Iterationen mit --iterations angeben.")
    if args.saved_config is not None:
        args.saved_config = args.saved_config.expanduser().resolve()
        if args.resume_from is None or not args.saved_config.is_file():
            parser.error("--saved-config benötigt --resume-from und eine vorhandene Konfigurationsdatei.")
    return args


def prepare_training(args):
    """Resolve the historical configuration before creating a scene or new run."""
    resume = None
    if args.resume_from is not None:
        from speed_eval import load_config, load_saved_config

        configs, source = (
            load_saved_config(args.saved_config) if args.saved_config is not None
            else load_config(args.resume_from)
        )
        if not Path(source).is_file():
            raise ValueError("Gespeicherte Run-Konfiguration fehlt. Mit --saved-config config.yaml angeben.")
        match = re.fullmatch(r"go2_genesis_rsl_rl_(config_\w+)_v\d+", args.resume_from.parent.name)
        inferred = match.group(1) if match else None
        if inferred and args.config_name and inferred != args.config_name:
            raise ValueError(f"Checkpoint gehört zu {inferred}, nicht zu {args.config_name}.")
        config_name = inferred or args.config_name
        if config_name is None:
            raise ValueError("Konfigurationsname nicht ableitbar; --config_name angeben.")
        train_cfg, reward_cfg, curriculum_cfg, env_cfg = deepcopy(configs)
        if curriculum_cfg.get("enabled", False):
            raise ValueError("Resume eines aktiven Curriculums ist nicht unterstützt: dessen Zustand fehlt im Checkpoint.")
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter", "infos"}
        if not required.issubset(checkpoint):
            raise ValueError(f"Checkpoint enthält nicht alle Trainingszustände: {sorted(required - checkpoint.keys())}")
        iteration = checkpoint["iter"]
        if not isinstance(iteration, int) or iteration < 0:
            raise ValueError("Checkpoint enthält keinen gültigen Iterationszähler.")
        resume = {
            "checkpoint": str(args.resume_from), "config_source": source,
            "parent_iteration": iteration, "start_iteration": iteration + 1,
            "additional_iterations": args.iterations,
            "final_iteration": iteration + args.iterations,
            "simulator_state_restored": False,
        }
    else:
        config_name = args.config_name or "config_A"
        train_cfg, reward_cfg, curriculum_cfg, env_cfg = deepcopy(get_config(config_name))
        source = f"aktuelle config.py: {config_name}"
    if args.iterations is not None:
        train_cfg["num_learning_iterations"] = args.iterations
    if args.video_interval is not None:
        train_cfg["video_interval"] = args.video_interval
    return config_name, (train_cfg, reward_cfg, curriculum_cfg, env_cfg), source, resume


def restore_training(runner, resume: dict, device: str):
    """Restore actor, critic, normalizers and optimizer, then advance past the saved update."""
    runner.load(resume["checkpoint"], map_location=device)
    if runner.current_learning_iteration != resume["parent_iteration"]:
        raise ValueError("Geladener Checkpoint stimmt nicht mit dem geprüften Iterationszähler überein.")
    runner.current_learning_iteration = resume["start_iteration"]
    runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]["lr"]


def main(argv=None) -> None:
    args = parse_args(argv)
    config_name, configs, config_source, resume = prepare_training(args)
    train_cfg, reward_cfg, curriculum_cfg, env_cfg = configs
    print(f"Konfiguration: {config_source}", flush=True)
    iterations = train_cfg.get("num_learning_iterations", 4000)
    start_iteration = resume["start_iteration"] if resume else 0
    print(f"Training: {iterations} Iterationen, {start_iteration} bis {start_iteration + iterations - 1}.", flush=True)
    if resume:
        print(f"Fortsetzung von: {resume['checkpoint']}", flush=True)
        print("Actor, Critic, Normalisierung und Optimizer werden geladen; Simulator startet neu.", flush=True)
    if args.dry_run:
        print("Dry-Run: kein Simulator, kein neuer Run und kein Training.")
        return

    initialize_simulation(env_cfg.get("seed", 1))
    torch.manual_seed(env_cfg.get("seed", 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = build_env(
        device=device,
        num_envs=env_cfg.get("num_envs", 4096),
        show_viewer=False,
        lin_vel_x=curriculum_cfg["start_lin_vel_x"],
        use_terrain=env_cfg.get("use_terrain", True),
        episode_length_s=env_cfg.get("episode_length_s", 30.0),
        reward_cfg=reward_cfg,
        command_range=env_cfg.get(
            "command_range",
            {
                "lin_vel_x": (0.0, 1.0),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_yaw": (0.0, 0.0),
            },
        ),
        command_range_allowed=env_cfg.get("command_range_allowed", False),
        terminate_on_torso_contact=env_cfg.get("terminate_on_torso_contact", False),
    )

    logs_root = project_root / "logs"
    run_name, log_dir = reserve_run_version(
        logs_root=logs_root,
        base_run_name="go2_genesis_rsl_rl" + f"_{config_name}",
    )

    if resume:
        (log_dir / "resume.json").write_text(json.dumps(resume, indent=2) + "\n")

    wandb.init(
        project="Genesis_Dog_Walking",
        name=run_name,
        config={
            "Curriculum": curriculum_cfg,
            "Reward": reward_cfg,
            "Train_cfg": train_cfg,
            "Environement_cfg": env_cfg,
            "Resume": resume,
        },
        sync_tensorboard=True,  # Der magische Trick! Zieht sich alle TB-Daten.
        dir=project_root / "logs",
    )

    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=str(log_dir),
        device=device,
        vid_interval=train_cfg.get("video_interval", 200),
        video_dir=log_dir / "videos",
    )
    if resume:
        restore_training(runner, resume, device)
    print("Runner ist bereit.")
    print(f"Run name: {run_name}")
    print(f"Seed: {env_cfg.get('seed', 1)}")
    print(f"Logs: {log_dir}")
    print(f"Start command lin_vel_x: {curriculum_cfg['start_lin_vel_x']}")
    # print(f"Num envs: {env_cfg.get("num_envs", 4096)}")
    print(
        f"Total learning iterations: {train_cfg.get('num_learning_iterations', 4000)}"
    )
    if curriculum_cfg["enabled"]:
        print("Curriculum aktiv:", curriculum_cfg)
    print("Reward configuration:", reward_cfg)
    print("Train_cfg: ", train_cfg)

    final_vel = runner.learn(
        num_learning_iterations=train_cfg.get("num_learning_iterations", 4000),
        # curriculum=True,
        curriculum_cfg=curriculum_cfg,
        init_at_random_ep_len=True,
    )

    if args.skip_final_eval:
        wandb.finish()
        return

    latest_checkpoint = load_latest_checkpoint(runner, log_dir=log_dir, device=device)
    eval_env = build_env(
        device=device,
        num_envs=1,
        show_viewer=False,
        lin_vel_x=final_vel,
        use_terrain=env_cfg.get("use_terrain", True),
        episode_length_s=env_cfg.get("episode_length_s", 30.0),
        reward_cfg=reward_cfg,
        command_range_allowed=False,
        terminate_on_torso_contact=env_cfg.get("terminate_on_torso_contact", False),
    )
    video_dir = project_root / "video" / run_name
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"go2_eval_{latest_checkpoint.stem}.mp4"
    tmp = env_cfg.get("command_range", {}).get("lin_vel_x", [0.0, 1.0])
    tmp_y = env_cfg.get("command_range", {}).get("lin_vel_y", [0.0, 0.0])
    tmp_yaw = env_cfg.get("command_range", {}).get("ang_vel_yaw", [0.0, 0.0])

    make_eval_video(
        runner,
        eval_env,
        video_path.with_name(
            f"go2_eval_{latest_checkpoint.stem}_lin_vel_x_{tmp[0]:.2f}_lin_vel_y_{tmp_y[0]:.2f}.mp4"
        ),
        device=device,
        speed_x=tmp[0],
        speed_y=tmp_y[0],
    )

    make_eval_video(
        runner,
        eval_env,
        video_path.with_name(
            f"go2_eval_{latest_checkpoint.stem}_lin_vel_x_{tmp[1]:.2f}_lin_vel_y_{tmp_y[1]:.2f}.mp4"
        ),
        device=device,
        speed_x=tmp[1],
        speed_y=tmp_y[1],
    )

    make_eval_video(
        runner,
        eval_env,
        video_path.with_name(
            f"go2_eval_{latest_checkpoint.stem}_lin_vel_x_{tmp[0]:.2f}_lin_vel_y_{tmp_y[1]:.2f}.mp4"
        ),
        device=device,
        speed_x=tmp[0],
        speed_y=tmp_y[1],
    )

    make_eval_video(
        runner,
        eval_env,
        video_path.with_name(
            f"go2_eval_{latest_checkpoint.stem}_lin_vel_x_{tmp[1]:.2f}_lin_vel_y_{tmp_y[0]:.2f}.mp4"
        ),
        device=device,
        speed_x=tmp[1],
        speed_y=tmp_y[0],
    )
    make_eval_video(
        runner,
        eval_env,
        video_path.with_name(
            f"go2_eval_{latest_checkpoint.stem}_lin_vel_x_{0.0:.2f}_lin_vel_y_{0.0:.2f}_yaw_{tmp_yaw[0]:.2f}.mp4"
        ),
        device=device,
        speed_x=0.0,
        speed_y=0.0,
        speed_yaw=tmp_yaw[0],
    )
    make_eval_video(
        runner,
        eval_env,
        video_path.with_name(
            f"go2_eval_{latest_checkpoint.stem}_lin_vel_x_{0.0:.2f}_lin_vel_y_{0.0:.2f}_yaw_{tmp_yaw[1]:.2f}.mp4"
        ),
        device=device,
        speed_x=0.0,
        speed_y=0.0,
        speed_yaw=tmp_yaw[1],
    )


if __name__ == "__main__":
    # Quick debug mode to diagnose reward values
    if "--debug" in sys.argv:
        print("Running in DEBUG mode with only 10 iterations...")
        main(num_learning_iterations=10, make_video=False)
    else:
        main()
