"""CSV pose traces captured before automatic reset; quaternions use WXYZ."""
from __future__ import annotations

import csv
import math
from pathlib import Path


def pose_sample(state, time_s, measured, link_names):
    if "base_quat" not in state:
        raise RuntimeError("Zeitreihen benötigen diagnostics['base_quat'] vor dem Reset.")
    position = state["base_pos"][0].tolist()
    quat = state["base_quat"][0].tolist()
    norm = math.sqrt(sum(v * v for v in quat))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Ungültiges Orientierungsquaternion im Diagnostikzustand.")
    w, x, y, z = (v / norm for v in quat)
    row = {
        "time_s": time_s, "measurement": int(measured),
        **dict(zip(("world_x_m", "world_y_m", "world_z_m"), position)),
        **dict(zip(("quat_w", "quat_x", "quat_y", "quat_z"), quat)),
        "roll_rad": math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)),
        "pitch_rad": math.asin(max(-1., min(1., 2 * (w*y - z*x)))),
        "yaw_rad": math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)),
    }
    if "base_lin_vel_local" in state:
        row.update(zip(("vx_m_s", "vy_m_s", "vz_m_s"), state["base_lin_vel_local"][0].tolist()))
        row["yaw_rate_local_rad_s"] = state["base_ang_vel_local"][0, 2].item()
        row["yaw_rate_world_rad_s"] = state["base_ang_vel"][0, 2].item()
        row["height_above_terrain_m"] = state["base_height_above_terrain"][0].item()
        row.update({f"contact_{name}": int(value) for name, value in
                    zip(link_names, state["link_contacts"][0].tolist())})
    return row


def save_trajectory(path: Path, rows: list[dict], warmup_s: float,
                    commands: tuple[float, float, float], end_reason: str) -> dict:
    """Split warmup from measurement using actual boundary poses, never yaw rate integration."""
    initial = rows[0]
    previous_yaw = initial["yaw_rad"]
    unwrapped = previous_yaw
    boundary = initial
    for row in rows:
        yaw = row["yaw_rad"]
        unwrapped += math.atan2(math.sin(yaw - previous_yaw), math.cos(yaw - previous_yaw))
        row["yaw_unwrapped_rad"] = unwrapped
        previous_yaw = yaw
        row["relative_x_m"] = row["world_x_m"] - initial["world_x_m"]
        row["relative_y_m"] = row["world_y_m"] - initial["world_y_m"]
        row.update(zip(("command_vx_m_s", "command_vy_m_s", "command_yaw_rate_rad_s"), commands))
        row["end_reason"] = ""
        if row["time_s"] <= warmup_s + 1e-9:
            boundary = row
    rows[-1]["end_reason"] = end_reason
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    final = rows[-1]
    covered = final["time_s"] >= warmup_s - 1e-9
    measured = any(row["measurement"] for row in rows)
    return {
        "initial_world_yaw_rad": initial["yaw_unwrapped_rad"],
        "world_yaw_at_measurement_start_rad": boundary["yaw_unwrapped_rad"] if covered else None,
        "warmup_world_yaw_change_rad": boundary["yaw_unwrapped_rad"] - initial["yaw_unwrapped_rad"],
        "measurement_world_yaw_change_rad": final["yaw_unwrapped_rad"] - boundary["yaw_unwrapped_rad"] if measured else None,
        "warmup_net_world_x_m": boundary["world_x_m"] - initial["world_x_m"],
        "warmup_net_world_y_m": boundary["world_y_m"] - initial["world_y_m"],
        "measurement_net_world_x_m": final["world_x_m"] - boundary["world_x_m"] if measured else None,
        "measurement_net_world_y_m": final["world_y_m"] - boundary["world_y_m"] if measured else None,
        "max_abs_roll_rad": max(abs(row["roll_rad"]) for row in rows),
        "max_abs_pitch_rad": max(abs(row["pitch_rad"]) for row in rows),
    }
