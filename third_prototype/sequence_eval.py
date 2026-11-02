"""Evaluate command changes in one continuous episode, without resets between phases.

Run eval_d17_sequence.sh for the configured B37 test, or:
    python third_prototype/sequence_eval.py --model-path PATH --terrain flat

The first two seconds settle the reset pose. All transition samples afterwards
count toward phase metrics. A failure ends the entire trial, including its video.
The original 39-second test is followed by a high-speed block with 0.7/0.8 m/s,
both turning directions and a final stop. Total default duration: 73 seconds.
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import math
from pathlib import Path

import torch

if __package__:
    from . import speed_eval
else:
    import speed_eval


# name, duration in seconds, local forward speed in m/s, local yaw rate in rad/s
DEFAULT_SEQUENCE = (
    ("anfahren", 4.0, 0.2, 0.0),
    ("beschleunigen_04", 4.0, 0.4, 0.0),
    ("beschleunigen_06", 4.0, 0.6, 0.0),
    ("abbremsen_02", 4.0, 0.2, 0.0),
    ("stopp_1", 3.0, 0.0, 0.0),
    ("drehen_links", 4.0, 0.0, 0.4),
    ("stopp_2", 3.0, 0.0, 0.0),
    ("weiterfahren", 4.0, 0.4, 0.0),
    ("kurve_rechts", 4.0, 0.4, -0.4),
    ("stopp_3", 3.0, 0.0, 0.0),
    # Keep the original phases first so their transitions remain comparable.
    # Then test the upper command range, including the previously failing turn.
    ("beschleunigen_07", 4.0, 0.7, 0.0),
    ("beschleunigen_08", 6.0, 0.8, 0.0),
    ("kurve_rechts_08", 8.0, 0.8, -0.4),
    ("geradeaus_08", 4.0, 0.8, 0.0),
    ("kurve_links_08", 8.0, 0.8, 0.4),
    ("stopp_08", 4.0, 0.0, 0.0),
)


def steps_for(seconds, dt):
    return math.ceil(seconds / dt - 1e-10)


def phase_metrics(samples, phase, dt, *, tolerance, yaw_tolerance, stop_speed, hold_s):
    """Include the transition; report the time to complete a continuous good window.

    time_to_hold_s is the END of the first qualifying window, not a guarantee
    that tracking remains in tolerance afterwards. Stop path includes that window.
    """
    name, duration, vx, yaw = phase
    requested = steps_for(duration, dt)
    is_stop = vx == 0 and yaw == 0
    xy_limit = stop_speed if is_stop else tolerance
    good_streak = 0
    hold_steps = steps_for(hold_s, dt)
    time_to_hold = None
    path_until_hold = None
    within = 0
    for i, row in enumerate(samples):
        good = (math.hypot(row["vx_m_s"] - vx, row["vy_m_s"]) <= xy_limit
                and abs(row["yaw_rad_s"] - yaw) <= yaw_tolerance)
        within += good
        good_streak = good_streak + 1 if good else 0
        if time_to_hold is None and good_streak >= hold_steps:
            time_to_hold = (i + 1) * dt
            path_until_hold = row["phase_path_m"] if is_stop else None
    return {
        "name": name,
        "command_vx_m_s": vx,
        "command_vy_m_s": 0.0,
        "command_yaw_rate_rad_s": yaw,
        "requested_duration_s": requested * dt,
        "duration_s": len(samples) * dt,
        "started": bool(samples),
        "completed": len(samples) == requested and not any(r["end_reason"] for r in samples),
        "coverage_fraction": len(samples) / requested,
        "x": speed_eval.axis_metrics([r["vx_m_s"] for r in samples], vx, tolerance),
        "y": speed_eval.axis_metrics([r["vy_m_s"] for r in samples], 0.0, tolerance),
        "yaw": speed_eval.axis_metrics(
            [r["yaw_rad_s"] for r in samples], yaw, yaw_tolerance, unit="rad_s"),
        "hold_xy_tolerance_m_s": xy_limit,
        "hold_yaw_tolerance_rad_s": yaw_tolerance,
        "hold_duration_s": hold_steps * dt,
        "time_to_hold_s": time_to_hold,
        "within_tolerance_requested_fraction": within / requested,
        "stop_path_until_hold_m": path_until_hold,
        "phase_path_m": samples[-1]["phase_path_m"] if samples else None,
        "end_reason": (samples[-1]["end_reason"] or "phase_horizon") if samples else "not_reached",
    }


@torch.inference_mode()
def evaluate_sequence(policy, env, *, sequence=DEFAULT_SEQUENCE, warmup_s=2.0,
                      tolerance=0.1, yaw_tolerance=0.1, stop_speed=0.05,
                      hold_s=0.5, video_path=None):
    """Reset once per trial and preserve robot, action and policy state at switches."""
    if env.num_envs != 1:
        raise ValueError("Die Auswertung benötigt genau eine Umgebung.")
    if not math.isfinite(env.dt) or env.dt <= 0:
        raise ValueError("dt muss endlich und positiv sein.")
    settings = (warmup_s, tolerance, yaw_tolerance, stop_speed, hold_s)
    if (not all(math.isfinite(v) for v in settings) or warmup_s < 0
            or min(tolerance, yaw_tolerance, stop_speed, hold_s) <= 0):
        raise ValueError("Anlaufzeit muss nicht negativ, Toleranzen und Haltezeit müssen positiv sein.")
    if not sequence or any(
        not all(math.isfinite(v) for v in phase[1:]) or phase[1] <= 0
        for phase in sequence
    ):
        raise ValueError("Sequenz benötigt endliche Kommandos und positive Phasendauern.")
    warmup_steps = steps_for(warmup_s, env.dt)
    total_steps = warmup_steps + sum(steps_for(p[1], env.dt) for p in sequence)
    env.max_episode_length = total_steps + 1
    env.command_range_allowed = False
    policy.eval()
    policy.reset()
    env.set_commands(lin_vel_x=0.0, lin_vel_y=0.0, ang_vel_yaw=0.0)
    env.reset()
    obs = env.get_observations()
    previous_xy = env.base_pos[0, :2].tolist()
    samples = []
    phase_samples = [[] for _ in sequence]
    failure = None

    def take_step(phase_index, phase_name, vx, yaw, phase_path):
        nonlocal obs, previous_xy, failure
        obs, _, dones, extras = env.step(policy(obs, stochastic_output=False))
        state = extras["diagnostics"]  # Terminal snapshot survives automatic reset.
        velocity = state["base_lin_vel_local"][0].tolist()
        yaw_actual = state["base_ang_vel_local"][0, 2].item()
        pos = state["base_pos"][0].tolist()
        phase_path += math.hypot(pos[0] - previous_xy[0], pos[1] - previous_xy[1])
        previous_xy = pos[:2]
        if bool(dones[0].item()):
            if bool(extras["time_outs"][0].item()):
                failure = "timeout"
            else:
                reasons = [key for key in ("torso", "roll", "pitch", "fall")
                           if extras["foot_diag"].get(f"{key}_termination_fraction", 0) > 0]
                failure = "+".join(reasons) or "termination"
        row = {
            "time_s": (len(samples) + 1) * env.dt,
            "phase_index": phase_index, "phase": phase_name,
            "command_vx_m_s": vx, "command_yaw_rad_s": yaw,
            "vx_m_s": velocity[0], "vy_m_s": velocity[1], "yaw_rad_s": yaw_actual,
            "world_x_m": pos[0], "world_y_m": pos[1],
            "base_height_m": state["base_height_above_terrain"][0].item(),
            "phase_path_m": phase_path, "end_reason": failure,
        }
        # Do not silently export non-finite samples as successful measurements.
        if not all(math.isfinite(value) for value in row.values() if isinstance(value, float)):
            raise ValueError(f"Nicht endlicher Zustand bei t={row['time_s']:.3f}s.")
        samples.append(row)
        return row

    with speed_eval.record_video(env, video_path):
        for _ in range(warmup_steps):
            take_step(-1, "anlauf", 0.0, 0.0, 0.0)
            if failure:
                break
        for index, (name, duration, vx, yaw) in enumerate(sequence):
            if failure:
                break
            env.set_commands(lin_vel_x=vx, lin_vel_y=0.0, ang_vel_yaw=yaw)
            # set_commands() changes commands, but not the cached observation.
            # Refresh it so the FIRST action already sees the new command.
            env._compute_observations()
            obs = env.get_observations()
            path = 0.0
            for _ in range(steps_for(duration, env.dt)):
                row = take_step(index, name, vx, yaw, path)
                path = row["phase_path_m"]
                phase_samples[index].append(row)
                if failure:
                    break
    phases = [phase_metrics(rows, phase, env.dt, tolerance=tolerance,
                           yaw_tolerance=yaw_tolerance, stop_speed=stop_speed, hold_s=hold_s)
              for rows, phase in zip(phase_samples, sequence)]
    return {
        "completed": failure is None and len(samples) == total_steps,
        "end_reason": failure or "sequence_horizon",
        "requested_duration_s": total_steps * env.dt,
        "duration_s": len(samples) * env.dt,
        "warmup_s": warmup_steps * env.dt,
        "phases": phases,
        "samples": samples,
    }


def save_plot(samples, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    times = [r["time_s"] for r in samples]
    for ax, actual, command, label in (
        (axes[0], "vx_m_s", "command_vx_m_s", "vx [m/s]"),
        (axes[1], "yaw_rad_s", "command_yaw_rad_s", "Yaw [rad/s]"),
    ):
        ax.step(times, [r[command] for r in samples], where="pre", label="Soll", color="black")
        ax.plot(times, [r[actual] for r in samples], label="Ist", alpha=.8)
        ax.set_ylabel(label)
        ax.set_xlabel("Zeit [s]")
        ax.grid(alpha=.25)
        ax.legend()
        for previous, current in zip(samples, samples[1:]):
            if previous["phase_index"] != current["phase_index"]:
                ax.axvline(previous["time_s"], color="gray", alpha=.25)
    axes[2].plot([r["world_x_m"] for r in samples], [r["world_y_m"] for r in samples])
    axes[2].set_xlabel("Welt-X [m]")
    axes[2].set_ylabel("Welt-Y [m]")
    axes[2].set_aspect("equal", adjustable="datalim")
    axes[2].grid(alpha=.25)
    fig.suptitle("Kommandowechsel ohne Reset zwischen den Phasen")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--terrain", choices=("saved", "flat"), default="saved")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--yaw-tolerance", type=float, default=0.1)
    parser.add_argument("--stop-speed", type=float, default=0.05)
    parser.add_argument("--hold", type=float, default=0.5,
                        help="Zeit ununterbrochen innerhalb der Toleranzen, Standard 0.5 s.")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "video/sequence_eval")
    args = parser.parse_args(argv)
    model = args.model_path.expanduser().resolve()
    if not model.is_file():
        parser.error(f"Checkpoint existiert nicht: {model}")
    if args.repeats <= 0:
        parser.error("--repeats muss positiv sein.")
    if (not math.isfinite(args.warmup) or args.warmup < 0
            or not all(math.isfinite(v) and v > 0 for v in
                       (args.tolerance, args.yaw_tolerance, args.stop_speed, args.hold))):
        parser.error("Anlaufzeit muss nicht negativ, Toleranzen und Haltezeit müssen positiv sein.")
    (train_cfg, reward_cfg, _, saved_env_cfg), source = speed_eval.load_config(model)
    env_cfg = deepcopy(saved_env_cfg)
    if args.terrain == "flat":
        env_cfg["use_terrain"] = False
    seed = args.seed if args.seed is not None else env_cfg.get("seed", 1)
    if not 0 <= seed < 2**32 - args.repeats:
        parser.error("Seed liegt außerhalb des erlaubten Bereichs.")
    print(f"Checkpoint: {model}\nKonfiguration: {source}", flush=True)
    print(f"Initialisiere Genesis (Seed {seed}) ...", flush=True)
    device = speed_eval.initialize_simulation(seed)
    duration = args.warmup + sum(p[1] for p in DEFAULT_SEQUENCE)
    print(f"Baue Simulationsumgebung auf {device} ...", flush=True)
    env = speed_eval.build_env(device, env_cfg, reward_cfg, duration + .02)
    print("Lade Policy ...", flush=True)
    policy = speed_eval.load_policy(model, env, train_cfg, device)
    output = speed_eval.reserve_video_directory(args.output_dir.expanduser().resolve())
    print(f"Untergrund: {'fraktales Terrain' if env_cfg.get('use_terrain', True) else 'ebene Fläche'}")
    print(f"Ergebnisse: {output}")
    print("Ein Reset je Start; kein Reset bei Kommandowechseln.")
    print("Hold: Zeit bis zum Ende des ersten durchgehend passenden Zeitfensters; Stoppgrenze "
          f"{args.stop_speed:g} m/s, Yaw-Toleranz {args.yaw_tolerance:g} rad/s.")
    report = {
        "model_path": str(model), "config_source": source,
        "use_terrain": env_cfg.get("use_terrain", True), "terrain_option": args.terrain,
        "velocity_frame": "robot_local", "dt_s": env.dt,
        "sequence": [dict(name=p[0], duration_s=p[1], vx_m_s=p[2], yaw_rad_s=p[3])
                     for p in DEFAULT_SEQUENCE],
        "trials": [],
    }
    # Preserve the actual terrain generator settings, which are not in old W&B configs.
    if env_cfg.get("use_terrain", True):
        report["terrain_settings"] = env.terrain.morph.model_dump(
            mode="json", include={"n_subterrains", "subterrain_size", "horizontal_scale",
                                  "vertical_scale", "subterrain_types", "randomize",
                                  "subterrain_parameters"})
    report["base_init_pos_world_m"] = env.base_init_pos.tolist()
    for trial in range(args.repeats):
        torch.manual_seed(seed + trial)
        video = None if args.no_video else output / f"trial_{trial + 1:03d}.mp4"
        result = evaluate_sequence(
            policy, env, warmup_s=args.warmup, tolerance=args.tolerance,
            yaw_tolerance=args.yaw_tolerance, stop_speed=args.stop_speed,
            hold_s=args.hold, video_path=video)
        result.update(trial=trial + 1, seed=seed + trial)
        if video is not None:
            result["video_path"] = str(video)
        samples = result.pop("samples")
        csv_path = output / f"trial_{trial + 1:03d}.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
        plot_path = output / f"trial_{trial + 1:03d}.png"
        save_plot(samples, plot_path)
        result.update(csv_path=str(csv_path), plot_path=str(plot_path))
        report["trials"].append(result)
        (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"\nStart {trial + 1}: {result['end_reason']}")
        print("Phase                 vx-Soll Yaw-Soll MAE-X MAE-Yaw Hold[s] Stoppweg[m] InTol. Abdeckung Ende")
        def fmt(value):
            return f"{value:.3f}" if value is not None else "n/a"
        for phase in result["phases"]:
            print(f"{phase['name']:21} {phase['command_vx_m_s']:7.2f} "
                  f"{phase['command_yaw_rate_rad_s']:8.2f} "
                  f"{fmt(phase['x']['mae_m_s']):>5} {fmt(phase['yaw']['mae_rad_s']):>7} "
                  f"{fmt(phase['time_to_hold_s']):>7} {fmt(phase['stop_path_until_hold_m']):>11} "
                  f"{phase['within_tolerance_requested_fraction']:6.1%} "
                  f"{phase['coverage_fraction']:9.1%} {phase['end_reason']}", flush=True)
    print(f"\nVollständige Sequenzen: {sum(t['completed'] for t in report['trials'])}/{args.repeats}")
    return report


if __name__ == "__main__":
    main()
