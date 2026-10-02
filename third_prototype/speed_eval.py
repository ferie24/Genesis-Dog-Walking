from __future__ import annotations

import argparse
import json
import math
import logging
import re
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def build_reward_fn(reward_cfg: dict | None = None) -> Rewards:
    from reward import Rewards

    scales = {k: v for k, v in reward_cfg.items() if k != "tracking_sigma"}
    return Rewards(tracking_sigma=reward_cfg["tracking_sigma"], scales=scales)


def build_env(
    device: str,
    num_envs: int,
    show_viewer: bool,
    lin_vel_x: float | None = None,
    use_terrain: bool | None = None,
    episode_length_s: float | None = None,
    reward_cfg: dict | None = None,
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
        collect_diagnostics=True,
    )
    env.set_commands(lin_vel_x=lin_vel_x, lin_vel_y=0.0, ang_vel_yaw=0.0)
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

    latest_checkpoint = max(checkpoint_paths, key=_iter_from_name)
    print(f"Lade Checkpoint: {latest_checkpoint.name}")

    with torch.inference_mode():
        runner.load(str(latest_checkpoint), map_location=device)

    return latest_checkpoint


def evaluate_speed_tracking(
    runner,
    env,
    device,
    target_vx,
    eval_steps=1000,
    warmup_steps=100,
):
    """Measure one deterministic episode; never include post-reset state."""
    if eval_steps <= 0 or warmup_steps < 0 or warmup_steps >= eval_steps:
        raise ValueError("Require 0 <= warmup_steps < eval_steps")
    if env.num_envs != 1:
        raise ValueError("This diagnostic evaluator requires num_envs=1")

    policy = runner.get_inference_policy(device=device)
    policy.eval()
    env.command_range_allowed = False
    env.set_commands(lin_vel_x=target_vx, lin_vel_y=0.0, ang_vel_yaw=0.0)
    env.reset()
    obs = env.get_observations()
    initial_pos = env.base_pos[0].clone()
    previous_x = initial_pos[0].item()

    forward_distance = 0.0
    backward_distance = 0.0
    sum_vx = sum_error = sum_error_abs = sum_error_sq = 0.0
    sum_vx_sq = sum_yaw = sum_yaw_abs = sum_yaw_score = 0.0
    sum_tracking = sum_height = sum_ground_clearance = sum_reward = 0.0
    tracking_success_count = sample_count = 0
    contact_counts = torch.zeros(len(env.link_names), device=env.device)
    steps = 0
    end_reason = "evaluation_horizon"
    final_pos = initial_pos

    with torch.no_grad():
        for step in range(eval_steps):
            actions = policy(obs, stochastic_output=False)
            obs, rewards, dones, extras = env.step(actions)
            state = extras["diagnostics"]  # Snapshot from before automatic reset.
            final_pos = state["base_pos"][0]
            x = final_pos[0].item()
            dx = x - previous_x
            forward_distance += max(dx, 0.0)
            backward_distance += max(-dx, 0.0)
            previous_x = x
            contact_counts += state["link_contacts"][0].float()
            sum_reward += rewards[0].item()
            steps += 1

            if step >= warmup_steps:
                vx = state["base_lin_vel_local"][0, 0].item()
                yaw = state["base_ang_vel"][0, 2].item()
                error = vx - target_vx
                sum_vx += vx
                sum_vx_sq += vx * vx
                sum_error += error
                sum_error_abs += abs(error)
                sum_error_sq += error * error
                sum_yaw += yaw
                sum_yaw_abs += abs(yaw)
                sum_yaw_score += math.exp(-abs(yaw) / env.reward_fn.tracking_sigma)
                sum_tracking += math.exp(-(error * error) / env.reward_fn.tracking_sigma)
                sum_height += final_pos[2].item()
                sum_ground_clearance += state["base_height_above_terrain"][0].item()
                tracking_success_count += abs(error) <= 0.1
                sample_count += 1

            if bool(dones[0].item()):
                if bool(extras["time_outs"][0].item()):
                    end_reason = "timeout"
                else:
                    names = [
                        name for name in ("torso", "roll", "pitch", "fall")
                        if extras["foot_diag"][f"{name}_termination_fraction"].item() > 0
                    ]
                    end_reason = "+".join(names) if names else "termination"
                break

    mean = lambda total: total / sample_count if sample_count else None
    mean_vx = mean(sum_vx)
    results = {
        "command_m_s": target_vx,
        "steps": steps,
        "duration_s": steps * env.dt,
        "end_reason": end_reason,
        "net_world_x_m": final_pos[0].item() - initial_pos[0].item(),
        "net_world_y_m": final_pos[1].item() - initial_pos[1].item(),
        "forward_world_x_m": forward_distance,
        "backward_world_x_m": backward_distance,
        "mean_local_vx_m_s": mean_vx,
        "velocity_std_m_s": math.sqrt(max(0.0, mean(sum_vx_sq) - mean_vx * mean_vx)) if sample_count else None,
        "bias_m_s": mean(sum_error),
        "mae_m_s": mean(sum_error_abs),
        "rmse_m_s": math.sqrt(mean(sum_error_sq)) if sample_count else None,
        "tracking_success_fraction": tracking_success_count / sample_count if sample_count else None,
        "mean_tracking_score": mean(sum_tracking),
        "mean_world_yaw_rate_rad_s": mean(sum_yaw),
        "mean_abs_world_yaw_rate_rad_s": mean(sum_yaw_abs),
        "mean_yaw_reward_term": mean(sum_yaw_score),
        "mean_base_height_world_m": mean(sum_height),
        "mean_base_height_above_ground_m": mean(sum_ground_clearance),
        "episode_reward": sum_reward,
        "contact_fraction_by_link": {
            name: count.item() / steps for name, count in zip(env.link_names, contact_counts) if count.item() > 0
        },
    }
    return results


def get_config(config_name: str) -> dict:
    from config import build_configs

    configs = build_configs(config_name)
    if not configs:
        raise ValueError(f"Keine Konfiguration für '{config_name}' gefunden.")
    return (
        configs.get("Training_Config", {}),
        configs.get("Reward_Config", {}),
        configs.get("Curriculum_Config", {}),
        configs.get("Environment_Config", {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="config_A",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="config_A",
        help="Name of the configuration to use",
    )
    parser.add_argument("--commands", nargs="+", type=float, help="Fixed forward speeds in m/s")
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--json-out", type=Path, help="Optional output file for measurements")
    args = parser.parse_args()

    import genesis as gs
    import rsl_rl
    from rsl_rl.runners import OnPolicyRunner

    print(f"[DEBUG] rsl_rl geladen von: {rsl_rl.__file__}")
    backend = gs.gpu if torch.cuda.is_available() else gs.cpu
    gs.init(logging_level=logging.WARNING, backend=backend)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint-Pfad '{model_path}' existiert nicht.")
    train_cfg, reward_cfg, _curriculum_cfg, env_cfg = get_config(
        args.config_name
    )
    print(f"Verwende Konfiguration: {args.config_name}")
    torch.manual_seed(env_cfg.get("seed", 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logs_root = project_root / "logs"
    _run_name, log_dir = reserve_run_version(
        logs_root=logs_root, base_run_name="EVAL_VIDEO"
    )

    eval_env = build_env(
        device=device,
        num_envs=1,
        show_viewer=False,
        lin_vel_x=0.5,
        use_terrain=env_cfg.get("use_terrain", True),
        episode_length_s=env_cfg.get("episode_length_s", 30.0),
        reward_cfg=reward_cfg,
    )
    runner = OnPolicyRunner(
        env=eval_env,
        train_cfg=train_cfg,
        log_dir=str(log_dir),
        device=device,
        vid_interval=200,
        video_dir=log_dir / "videos",
    )
    with torch.inference_mode():
        runner.load(str(model_path), map_location=device)
    if args.commands is not None:
        commands = args.commands
    else:
        low, high = env_cfg.get("command_range", {}).get("lin_vel_x", [0.0, 1.0])
        commands = [round(low + 0.1 * i, 2) for i in range(int(round((high - low) / 0.1)) + 1)]
    results = []
    for target_vx in commands:
        result = evaluate_speed_tracking(
            runner, eval_env, device=device, target_vx=target_vx,
            eval_steps=args.eval_steps, warmup_steps=args.warmup_steps,
        )
        results.append(result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    main()
