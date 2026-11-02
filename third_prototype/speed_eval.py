r"""Evaluate deterministic XY and yaw-rate tracking from a trained checkpoint.

Example (30 seconds of measurement AFTER 2 seconds of warmup):
    python third_prototype/speed_eval.py --model-path third_prototype/logs/RUN/model_4999.pt \
        --commands 0 0.4 0.8 --y-range 0 0 --yaw-commands -0.4 0 0.4 \
        --duration 30 --warmup 2 --tolerance 0.1 --json-out measurements.json

Commands and tracking metrics use the robot's local frame. XY accuracy means
Euclidean velocity error <= tolerance, in m/s; yaw accuracy uses yaw-tolerance,
in rad/s. The environment's diagnostic snapshot must include base_ang_vel_local.
A fall ends the trial; terminal state is measured before automatic reset.
Use --terrain flat to evaluate on a flat plane instead of the saved terrain.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from decimal import Decimal
import itertools
import json
import logging
import math
import os
from pathlib import Path
import re
import sys

import torch

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from eval_trajectory import pose_sample, save_trajectory


def command_grid(low: float, high: float, step: float) -> list[float]:
    """Generate an inclusive grid without accumulated floating-point drift."""
    if not all(math.isfinite(v) for v in (low, high, step)):
        raise ValueError("Bereich und Schrittweite müssen endlich sein.")
    if step <= 0 or low > high:
        raise ValueError("Erforderlich: Minimum <= Maximum und Schrittweite > 0.")
    start, stop, spacing = (Decimal(str(v)) for v in (low, high, step))
    count = int((stop - start) // spacing) + 1
    return [float(start + i * spacing) for i in range(count)]


def get_config(config_name: str) -> tuple[dict, dict, dict, dict]:
    from config import build_configs

    cfg = build_configs(config_name)
    return tuple(
        cfg[k]
        for k in (
            "Training_Config",
            "Reward_Config",
            "Curriculum_Config",
            "Environment_Config",
        )
    )


def load_config(model_path: Path, config_name: str | None = None):
    """Prefer the matching local W&B snapshot unless a current config is requested."""
    if config_name is None:
        marker = f"Run name: {model_path.parent.name}"
        wandb_root = model_path.parent.parent / "wandb"
        for output in sorted(wandb_root.glob("*/files/output.log"), reverse=True):
            with output.open(errors="replace") as stream:
                prefix = stream.read(16384)
            if marker not in prefix.splitlines():
                continue
            config_path = output.with_name("config.yaml")
            if not config_path.is_file():
                continue
            return load_saved_config(config_path)
        match = re.search(r"_(config_\w+)_v\d+$", model_path.parent.name)
        config_name = match.group(1) if match else "config_A"
    return get_config(config_name), f"aktuelle config.py: {config_name}"


def load_saved_config(config_path: Path):
    import yaml

    saved = yaml.safe_load(config_path.read_text())
    keys = ("Train_cfg", "Reward", "Curriculum", "Environement_cfg")
    return tuple(deepcopy(saved[k]["value"]) for k in keys), str(config_path)


def initialize_simulation(seed: int) -> str:
    # Must be set before importing Genesis/OpenGL on headless compute nodes.
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import genesis as gs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gs.init(
        seed=seed,
        logging_level=logging.WARNING,
        backend=gs.gpu if device == "cuda" else gs.cpu,
    )
    return device


def build_env(device: str, env_cfg: dict, reward_cfg: dict, episode_length_s: float):
    from make_environment import Go2WalkingEnv
    from reward import Rewards

    scales = {
        k: v
        for k, v in reward_cfg.items()
        if k not in {"tracking_sigma", "tracking_sigma_ang_vel"}
    }
    # Historical checkpoints predate the newly added Y tracking term.
    scales.setdefault("tracking_lin_vel_y", 0.0)
    reward = Rewards(
        tracking_sigma=reward_cfg.get("tracking_sigma", 0.1),
        tracking_sigma_ang_vel=reward_cfg.get("tracking_sigma_ang_vel"),
        scales=scales,
    )
    return Go2WalkingEnv(
        num_envs=1,
        device=device,
        show_viewer=False,
        use_terrain=env_cfg.get("use_terrain", True),
        episode_length_s=episode_length_s,
        min_up_dot=0.1,
        min_base_height=0.18,
        reward_fn=reward,
        collect_diagnostics=True,
        command_range_allowed=False,
        terminate_on_torso_contact=env_cfg.get("terminate_on_torso_contact", False),
    )


def load_policy(model_path: Path, env, train_cfg: dict, device: str):
    """Load only the actor, including its saved observation normalization."""
    from rsl_rl.utils import resolve_callable

    actor_cfg = deepcopy(train_cfg["actor"])
    actor_class = resolve_callable(actor_cfg.pop("class_name"))
    policy = actor_class(
        env.get_observations(),
        train_cfg["obs_groups"],
        "actor",
        env.num_actions,
        **actor_cfg,
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    policy.eval()
    return policy


def axis_metrics(
    values: list[float], target: float, tolerance: float, *, unit="m_s"
) -> dict:
    names = tuple(
        f"{name}_{unit}"
        for name in ("mean", "std", "bias", "mae", "rmse", "max_abs_error")
    ) + ("within_tolerance_fraction",)
    if not values:
        return dict.fromkeys(names)
    count = len(values)
    mean = math.fsum(values) / count
    errors = [v - target for v in values]
    return {
        f"mean_{unit}": mean,
        f"std_{unit}": math.sqrt(math.fsum((v - mean) ** 2 for v in values) / count),
        f"bias_{unit}": math.fsum(errors) / count,
        f"mae_{unit}": math.fsum(abs(e) for e in errors) / count,
        f"rmse_{unit}": math.sqrt(math.fsum(e * e for e in errors) / count),
        f"max_abs_error_{unit}": max(abs(e) for e in errors),
        "within_tolerance_fraction": sum(abs(e) <= tolerance for e in errors) / count,
    }


@contextmanager
def record_video(env, video_path: Path | None):
    """Record this same measurement rollout, including its terminal state."""
    if video_path is None:
        yield
        return
    camera = env.camera
    previous = getattr(env, "record_eval_video", False)
    camera.start_recording()
    env.record_eval_video = True
    try:
        env._update_follow_camera()
        camera.render()  # Include the reset pose, even if the first step fails.
        yield
    finally:
        env.record_eval_video = previous
        camera.stop_recording(save_to_filename=str(video_path), fps=round(1 / env.dt))


@torch.inference_mode()
def evaluate_speed_tracking(
    policy,
    env,
    target_vx: float,
    target_vy: float = 0.0,
    eval_steps: int = 1000,
    warmup_steps: int = 100,
    tolerance: float = 0.1,
    *,
    target_yaw_rate: float = 0.0,
    yaw_tolerance: float = 0.1,
    video_path: Path | None = None,
    trajectory_path: Path | None = None,
) -> dict:
    """Measure one episode, including a terminal sample but never reset state.

    Reset and rollout share inference mode: step() may replace persistent
    environment buffers with inference tensors reused by the next trial.
    """
    if eval_steps <= 0 or not 0 <= warmup_steps < eval_steps:
        raise ValueError("Erforderlich: 0 <= warmup_steps < eval_steps.")
    if (
        not all(
            math.isfinite(v)
            for v in (target_vx, target_vy, target_yaw_rate, tolerance, yaw_tolerance)
        )
        or min(tolerance, yaw_tolerance) <= 0
    ):
        raise ValueError("Kommandos müssen endlich und die Toleranz positiv sein.")
    if env.num_envs != 1:
        raise ValueError("Die Auswertung benötigt genau eine Umgebung.")
    policy.eval()
    policy.reset()
    env.command_range_allowed = False
    env.set_commands(
        lin_vel_x=target_vx, lin_vel_y=target_vy, ang_vel_yaw=target_yaw_rate
    )
    env.reset()
    obs = env.get_observations()
    initial_position = env.base_pos[0].clone()
    final_position = initial_position
    vx_values, vy_values, yaw_values, world_yaw_values = [], [], [], []
    contact_counts = torch.zeros(len(env.link_names), device=env.device)
    measurement_contacts = torch.zeros_like(contact_counts)
    trajectory = []
    if trajectory_path is not None:
        trajectory.append(pose_sample(
            {"base_pos": env.base_pos, "base_quat": env.base_quat},
            0.0, False, env.link_names,
        ))
    steps = 0
    end_reason = "evaluation_horizon"
    episode_reward = 0.0
    with record_video(env, video_path):
        for step in range(eval_steps):
            obs, rewards, dones, extras = env.step(policy(obs, stochastic_output=False))
            state = extras["diagnostics"]
            if "base_ang_vel_local" not in state:
                raise RuntimeError(
                    "Yaw evaluation requires diagnostics['base_ang_vel_local']. "
                    "In make_environment.py step(), add "
                    "'base_ang_vel_local': transform_by_quat(self.base_ang_vel, base_quat_inv) "
                    "to the diagnostic snapshot before automatic reset."
                )
            final_position = state["base_pos"][0].clone()
            contact_counts += state["link_contacts"][0].float()
            episode_reward += rewards[0].item()
            steps += 1
            if trajectory_path is not None:
                trajectory.append(pose_sample(
                    state, steps * env.dt, step >= warmup_steps, env.link_names,
                ))
            if step >= warmup_steps:
                measurement_contacts += state["link_contacts"][0].float()
                velocity = state["base_lin_vel_local"][0]
                vx_values.append(velocity[0].item())
                vy_values.append(velocity[1].item())
                yaw_values.append(state["base_ang_vel_local"][0, 2].item())
                world_yaw_values.append(state["base_ang_vel"][0, 2].item())
            if bool(dones[0].item()):
                if bool(extras["time_outs"][0].item()):
                    end_reason = "timeout"
                else:
                    reasons = [
                        name
                        for name in ("torso", "roll", "pitch", "fall")
                        if extras["foot_diag"].get(f"{name}_termination_fraction", 0)
                        > 0
                    ]
                    end_reason = "+".join(reasons) if reasons else "termination"
                break
    count = len(vx_values)
    requested_samples = eval_steps - warmup_steps
    errors_xy = [
        math.hypot(vx - target_vx, vy - target_vy)
        for vx, vy in zip(vx_values, vy_values)
    ]
    within = sum(e <= tolerance for e in errors_xy)
    yaw_errors = [abs(value - target_yaw_rate) for value in yaw_values]
    yaw_within = sum(e <= yaw_tolerance for e in yaw_errors)
    joint_within = sum(
        xy <= tolerance and yaw <= yaw_tolerance
        for xy, yaw in zip(errors_xy, yaw_errors)
    )
    yaw_metrics = axis_metrics(yaw_values, target_yaw_rate, yaw_tolerance, unit="rad_s")
    yaw_metrics.update(
        mean_abs_rad_s=math.fsum(abs(v) for v in yaw_values) / count if count else None,
        correct_direction_fraction=(
            sum(v * target_yaw_rate > 0 for v in yaw_values) / count
            if count and target_yaw_rate != 0
            else None
        ),
        within_tolerance_requested_fraction=yaw_within / requested_samples,
    )
    completed = steps == eval_steps and end_reason in ("evaluation_horizon", "timeout")
    result = {
        "command_vx_m_s": target_vx,
        "command_vy_m_s": target_vy,
        "command_yaw_rate_rad_s": target_yaw_rate,
        "requested_total_duration_s": eval_steps * env.dt,
        "requested_measurement_duration_s": requested_samples * env.dt,
        "warmup_s": warmup_steps * env.dt,
        "duration_s": steps * env.dt,
        "measurement_duration_s": count * env.dt,
        "steps": steps,
        "measurement_samples": count,
        "completed": completed,
        "end_reason": end_reason,
        "measurement_coverage_fraction": count / requested_samples,
        "x": axis_metrics(vx_values, target_vx, tolerance),
        "y": axis_metrics(vy_values, target_vy, tolerance),
        "yaw": yaw_metrics,
        "yaw_tolerance_rad_s": yaw_tolerance,
        "xy_mae_m_s": math.fsum(errors_xy) / count if count else None,
        "xy_rmse_m_s": math.sqrt(math.fsum(e * e for e in errors_xy) / count)
        if count
        else None,
        "xy_max_error_m_s": max(errors_xy) if count else None,
        "xy_tolerance_m_s": tolerance,
        "xy_within_tolerance_fraction": within / count if count else None,
        # Unmeasured time after an early termination does not count as success.
        "xy_within_tolerance_requested_fraction": within / requested_samples,
        "xy_yaw_within_tolerance_fraction": joint_within / count if count else None,
        "xy_yaw_within_tolerance_requested_fraction": joint_within / requested_samples,
        "mean_world_yaw_rate_rad_s": math.fsum(world_yaw_values) / count
        if count
        else None,
        "mean_abs_world_yaw_rate_rad_s": math.fsum(abs(v) for v in world_yaw_values)
        / count
        if count
        else None,
        "net_world_x_m": (final_position[0] - initial_position[0]).item(),
        "net_world_y_m": (final_position[1] - initial_position[1]).item(),
        "episode_reward": episode_reward,
        "contact_fraction_by_link": {
            name: value.item() / steps
            for name, value in zip(env.link_names, contact_counts)
            if value.item() > 0
        },
        "measurement_contact_fraction_by_link": {
            name: value.item() / count if count else None
            for name, value in zip(env.link_names, measurement_contacts)
        },
        "final_base_pos_world_m": final_position.tolist(),
        "final_base_height_above_terrain_m": state["base_height_above_terrain"][
            0
        ].item(),
        "final_contact_links": [
            name
            for name, active in zip(env.link_names, state["link_contacts"][0].tolist())
            if active
        ],
    }
    if trajectory_path is not None:
        result["trajectory_path"] = str(trajectory_path)
        result["trajectory_metrics"] = save_trajectory(
            trajectory_path, trajectory, warmup_steps * env.dt,
            (target_vx, target_vy, target_yaw_rate), end_reason,
        )
    return result


def build_parser(description=None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description or __doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path", "--model_path", dest="model_path", type=Path, required=True
    )
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "--config-name",
        "--config_name",
        dest="config_name",
        help="Aktuelle Konfiguration erzwingen; sonst gespeicherte Run-Konfiguration verwenden.",
    )
    config_group.add_argument(
        "--saved-config", type=Path,
        help="Gespeicherte W&B-config.yaml ausdrücklich auswählen.",
    )
    parser.add_argument(
        "--terrain",
        choices=("saved", "flat"),
        default="saved",
        help="Untergrund: gespeicherte Konfiguration (Standard) oder ebene Fläche nur für diese Evaluation.",
    )
    x_group = parser.add_mutually_exclusive_group()
    x_group.add_argument(
        "--x-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="X-Bereich in m/s.",
    )
    x_group.add_argument(
        "--commands", nargs="+", type=float, help="Alternativ einzelne X-Kommandos."
    )
    parser.add_argument(
        "--y-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Y-Bereich in m/s.",
    )
    parser.add_argument(
        "--yaw-commands",
        nargs="+",
        type=float,
        default=[0.0],
        help="Yaw-Kommandos in rad/s, Standard nur 0. Wird mit allen XY-Kommandos kombiniert.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="Schrittweite für beide Achsen, Standard 0.1 m/s.",
    )
    duration_group = parser.add_mutually_exclusive_group()
    duration_group.add_argument(
        "--duration", type=float, help="Messdauer NACH Anlaufzeit, Standard 30 s."
    )
    duration_group.add_argument(
        "--eval-steps",
        type=int,
        help="Alternativ gesamte Schrittzahl INKLUSIVE Anlaufzeit.",
    )
    warmup_group = parser.add_mutually_exclusive_group()
    warmup_group.add_argument("--warmup", type=float, help="Anlaufzeit, Standard 2 s.")
    warmup_group.add_argument(
        "--warmup-steps", type=int, help="Alternativ Anlaufzeit in Schritten."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="Maximaler XY-Vektorfehler, Standard 0.1 m/s.",
    )
    parser.add_argument(
        "--yaw-tolerance",
        type=float,
        default=0.1,
        help="Maximaler absoluter Yaw-Ratenfehler, Standard 0.1 rad/s.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Unabhängige Starts je Kommando, Standard 1.",
    )
    parser.add_argument("--seed", type=int, help="Seed; sonst Seed der Konfiguration.")
    parser.add_argument(
        "--json-out", type=Path, help="Alle Einzelmessungen als JSON speichern."
    )
    parser.add_argument(
        "--trajectory-dir", type=Path,
        help="Optional: Position, Orientierung und Kontakte je Schritt als CSV, inklusive Anlauf und Resetpose.",
    )
    return parser


def validate_args(parser, args):
    if not args.model_path.is_file():
        parser.error(f"Checkpoint existiert nicht: {args.model_path}")
    if args.saved_config is not None and not args.saved_config.is_file():
        parser.error(f"Gespeicherte Konfiguration existiert nicht: {args.saved_config}")
    for name, value, allow_zero in (
        ("step", args.step, False),
        ("duration", args.duration, False),
        ("warmup", args.warmup, True),
        ("tolerance", args.tolerance, False),
        ("yaw-tolerance", args.yaw_tolerance, False),
    ):
        if value is not None and (
            not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero)
        ):
            parser.error(
                f"--{name} muss endlich und {'nicht negativ' if allow_zero else 'positiv'} sein."
            )
    if args.eval_steps is not None and args.eval_steps <= 0:
        parser.error("--eval-steps muss positiv sein.")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        parser.error("--warmup-steps darf nicht negativ sein.")
    if (
        args.eval_steps is not None
        and args.warmup_steps is not None
        and args.warmup_steps >= args.eval_steps
    ):
        parser.error("Die Anlaufzeit muss kürzer als die gesamte Auswertung sein.")
    if args.repeats <= 0:
        parser.error("--repeats muss positiv sein.")
    if args.seed is not None and not 0 <= args.seed < 2**32 - args.repeats:
        parser.error("--seed liegt außerhalb des erlaubten Bereichs.")
    try:
        for interval in (args.x_range, args.y_range):
            if interval is not None:
                command_grid(*interval, args.step)
        if args.commands is not None and not all(
            math.isfinite(v) for v in args.commands
        ):
            raise ValueError("X-Kommandos müssen endlich sein.")
        if not all(math.isfinite(v) for v in args.yaw_commands):
            raise ValueError("Yaw-Kommandos müssen endlich sein.")
    except ValueError as exc:
        parser.error(str(exc))


def plan_steps(args, dt: float) -> tuple[int, int]:
    warmup = (
        args.warmup_steps
        if args.warmup_steps is not None
        else math.ceil((args.warmup if args.warmup is not None else 2.0) / dt - 1e-10)
    )
    total = (
        args.eval_steps
        if args.eval_steps is not None
        else warmup
        + math.ceil((args.duration if args.duration is not None else 30.0) / dt - 1e-10)
    )
    if total <= 0 or not 0 <= warmup < total:
        raise ValueError("Die Anlaufzeit muss kürzer als die gesamte Auswertung sein.")
    return total, warmup


def print_result(result: dict):
    def number(value):
        return f"{value:.3f}" if value is not None else "n/a"

    success = result["xy_within_tolerance_fraction"]
    percent = f"{100 * success:.1f}%" if success is not None else "n/a"
    print(
        f"{result['command_vx_m_s']:6.2f} {result['command_vy_m_s']:6.2f} "
        f"{result['command_yaw_rate_rad_s']:8.2f} {result['trial']:3d} {number(result['x']['mean_m_s']):>7} "
        f"{number(result['y']['mean_m_s']):>7} {number(result['x']['mae_m_s']):>7} "
        f"{number(result['y']['mae_m_s']):>7} {number(result['xy_rmse_m_s']):>8} "
        f"{number(result['yaw']['mean_rad_s']):>8} {number(result['yaw']['mae_rad_s']):>7} "
        f"{percent:>7} {result['measurement_duration_s']:6.2f}s "
        f"{100 * result['measurement_coverage_fraction']:5.1f}% {result['end_reason']}",
        flush=True,
    )


def reserve_video_directory(root: Path) -> Path:
    """Keep each evaluation in a new directory, preserving existing videos."""
    root.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        directory = root / f"EVAL_VIDEO_v{version:03d}"
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            version += 1


def main(argv=None, *, record_videos=False, description=None):
    parser = build_parser(description)
    if record_videos:
        parser.add_argument(
            "--video-dir",
            type=Path,
            default=project_root / "video",
            help="Video-Basisordner; legt einen neuen EVAL_VIDEO_vNNN-Unterordner an.",
        )
    args = parser.parse_args(argv)
    args.model_path = args.model_path.expanduser().resolve()
    if args.saved_config is not None:
        args.saved_config = args.saved_config.expanduser().resolve()
    validate_args(parser, args)
    configs, config_source = (
        load_saved_config(args.saved_config.expanduser().resolve())
        if args.saved_config is not None else load_config(args.model_path, args.config_name)
    )
    train_cfg, reward_cfg, _, env_cfg = configs
    env_cfg = deepcopy(env_cfg)
    if args.terrain == "flat":
        env_cfg["use_terrain"] = False
    use_terrain = env_cfg.get("use_terrain", True)
    ranges = env_cfg.get("command_range", {})
    try:
        xs = (
            args.commands
            if args.commands is not None
            else command_grid(
                *(args.x_range or ranges.get("lin_vel_x", [0.0, 1.0])), args.step
            )
        )
        ys = command_grid(
            *(args.y_range or ranges.get("lin_vel_y", [0.0, 0.0])), args.step
        )
        total_steps, warmup_steps = plan_steps(args, 0.02)
    except ValueError as exc:
        parser.error(str(exc))
    seed = args.seed if args.seed is not None else env_cfg.get("seed", 1)
    print(f"Checkpoint: {args.model_path}\nKonfiguration: {config_source}")
    terrain_name = "fraktales Terrain" if use_terrain else "ebene Fläche"
    terrain_source = "Eval-Override --terrain flat" if args.terrain == "flat" else "Konfiguration"
    print(f"Untergrund: {terrain_name} ({terrain_source})")
    yaws = args.yaw_commands
    print(
        f"{len(xs) * len(ys) * len(yaws)} XY/Yaw-Kommandos, {args.repeats} Start(s) je Kommando."
    )
    device = initialize_simulation(seed)
    env = build_env(device, env_cfg, reward_cfg, (total_steps + 1) * 0.02)
    total_steps, warmup_steps = plan_steps(args, env.dt)
    # The requested test horizon can exceed the training episode duration.
    env.max_episode_length = total_steps + 1
    policy = load_policy(args.model_path, env, train_cfg, device)
    video_dir = (
        reserve_video_directory(args.video_dir.expanduser().resolve())
        if record_videos
        else None
    )
    if video_dir is not None:
        print(f"Videos: {video_dir}")
        if args.json_out is None:
            args.json_out = video_dir / "metrics.json"
    print(
        f"Anlauf: {warmup_steps * env.dt:.2f}s; Messdauer: {(total_steps - warmup_steps) * env.dt:.2f}s."
    )
    print(
        f"XY in m/s, Yaw in rad/s; XY-Toleranz {args.tolerance:g} m/s, Yaw-Toleranz {args.yaw_tolerance:g} rad/s."
    )
    print(
        " X-Soll Y-Soll Yaw-Soll Run   X-Ist   Y-Ist   MAE-X   MAE-Y  RMSE-XY  Yaw-Ist MAE-Yaw XY-Gen. Messzeit Abdeck. Ende"
    )
    results = []
    for command_index, (vx, vy, yaw) in enumerate(
        itertools.product(xs, ys, yaws), start=1
    ):
        for trial in range(args.repeats):
            trial_seed = seed + trial
            torch.manual_seed(trial_seed)
            video_path = None
            trial_name = (
                f"{args.model_path.stem}_command_{command_index:04d}"
                f"_vx_{vx:g}_vy_{vy:g}_yaw_{yaw:g}_trial_{trial + 1:03d}"
            )
            if video_dir is not None:
                video_path = video_dir / f"{trial_name}.mp4"
            trajectory_path = (
                args.trajectory_dir.expanduser().resolve() / f"{trial_name}.csv"
                if args.trajectory_dir is not None else None
            )
            result = evaluate_speed_tracking(
                policy,
                env,
                vx,
                vy,
                total_steps,
                warmup_steps,
                args.tolerance,
                target_yaw_rate=yaw,
                yaw_tolerance=args.yaw_tolerance,
                video_path=video_path,
                trajectory_path=trajectory_path,
            )
            result.update(trial=trial + 1, seed=trial_seed)
            if video_path is not None:
                result["video_path"] = str(video_path)
            results.append(result)
            print_result(result)
    report = {
        "model_path": str(args.model_path),
        "config_source": config_source,
        "use_terrain": use_terrain,
        "terrain_option": args.terrain,
        "velocity_frame": "robot_local",
        "dt_s": env.dt,
        "yaw_rate_frame": "robot_local",
        "yaw_commands_rad_s": yaws,
        "x_commands_m_s": xs,
        "y_commands_m_s": ys,
        "repeats": args.repeats,
        "results": results,
    }
    if video_dir is not None:
        report["video_dir"] = str(video_dir)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.json_out.with_suffix('.tmp')
        temporary.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        temporary.replace(args.json_out)
        print(f"JSON gespeichert: {args.json_out}")
    print(
        f"Testzeit vollständig ohne Fehlerabbruch: {sum(r['completed'] for r in results)}/{len(results)} Starts."
    )
    return report


if __name__ == "__main__":
    main()
