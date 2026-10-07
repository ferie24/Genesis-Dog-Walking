"""Pure tensor logic for Go2 safety terminations."""

import torch


def sample_terrain_height(
    heights_m: torch.Tensor, xy_world: torch.Tensor, horizontal_scale: float
) -> torch.Tensor:
    """Bilinearly sample a terrain height map created at world XY origin."""
    if heights_m.ndim != 2 or xy_world.ndim != 2 or xy_world.shape[1] != 2:
        raise ValueError("Expected heights_m [rows, cols] and xy_world [envs, 2]")
    if horizontal_scale <= 0:
        raise ValueError("horizontal_scale must be positive")

    x = (xy_world[:, 0] / horizontal_scale).clamp(0, heights_m.shape[0] - 1)
    y = (xy_world[:, 1] / horizontal_scale).clamp(0, heights_m.shape[1] - 1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1 = (x0 + 1).clamp(max=heights_m.shape[0] - 1)
    y1 = (y0 + 1).clamp(max=heights_m.shape[1] - 1)
    tx, ty = x - x0, y - y0
    h0 = heights_m[x0, y0] * (1 - tx) + heights_m[x1, y0] * tx
    h1 = heights_m[x0, y1] * (1 - tx) + heights_m[x1, y1] * tx
    return h0 * (1 - ty) + h1 * ty


def body_contact_mask(link_contacts: torch.Tensor, body_link_indices: list[int]) -> torch.Tensor:
    """Combine torso and head contacts without treating leg contacts as falls."""
    return link_contacts[:, body_link_indices].any(dim=1)


def termination_masks(
    projected_gravity: torch.Tensor,
    base_height: torch.Tensor,
    min_base_height: float,
    episode_length: torch.Tensor,
    torso_contact: torch.Tensor,
    terminate_on_torso_contact: bool,
    grace_steps: int = 75,
) -> dict[str, torch.Tensor]:
    """Return per-environment reasons with a shared grace period after each reset.

    ``base_height`` is absolute world Z, matching the existing training rule.
    Terrain-relative height is measured separately for diagnosis and should
    replace this rule only after a simulation check.
    """
    active = episode_length >= grace_steps
    masks = {
        "roll": (projected_gravity[:, 1].abs() > 0.342) & active,
        "pitch": (projected_gravity[:, 0].abs() > 0.522) & active,
        "fall": (base_height < min_base_height) & active,
        "torso": (torso_contact.bool() & active) if terminate_on_torso_contact else torch.zeros_like(active),
    }
    masks["done"] = masks["roll"] | masks["pitch"] | masks["fall"] | masks["torso"]
    return masks
