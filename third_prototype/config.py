from copy import deepcopy


def build_configs(config_name: str) -> dict:
    """
    Seed sweep using the previous config B as the fixed baseline.

    A: seed 7 (reference).
    B: seed 11.
    C: seed 23.
    D: seed 42.
    E: seed 101.

    All rewards, PPO settings, curriculum, and environment parameters are
    identical except the seed. Commands retain the current B ranges:
    forward velocity 0.1-0.8 m/s and yaw velocity -0.5 to 0.5 rad/s.
    Train each variant from scratch and compare at the same training iteration:
    net forward distance, speed error, episode length, gait video, and falls.
    """

    # ============================================================
    # CONFIG A — PREVIOUS CONFIG B: FIXED BASELINE, SEED 7
    # Configs B-E change only the environment seed.
    # ============================================================

    config_A = {
        "Training_Config": {
            "run_name": None,
            "logger": "tensorboard",
            "num_steps_per_env": 96,
            "save_interval": 100,
            "obs_groups": {
                "actor": ["policy"],
                "critic": ["policy"],
            },
            "num_learning_iterations": 5000,
            "algorithm": {
                "class_name": "PPO",
                "clip_param": 0.2,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "gamma": 0.99,
                "lam": 0.95,
                "value_loss_coef": 1.0,
                "entropy_coef": 0.002,
                "learning_rate": 3e-4,
                "schedule": "fixed",
                "desired_kl": 0.02,
                "max_grad_norm": 1.0,
                "use_clipped_value_loss": True,
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
                    "learn_std": True,
                    "std_range": [0.05, 0.75],
                },
            },
            "critic": {
                "class_name": "MLPModel",
                "hidden_dims": [512, 256, 128],
                "activation": "elu",
                "obs_normalization": True,
            },
        },
        "Reward_Config": {
            "tracking_lin_vel_x": 3.0,
            "tracking_ang_vel": 0.25,
            "lin_vel_z": -1.0,
            "lin_vel_y": -5.0,
            "action_rate": -0.001,
            "similar_to_default": 0.0,
            "sideway_movement": 0.0,
            "tracking_sigma": 0.1,
            "x_progress": 0.5,
            # Preserve all reward weights from the selected config B.
            "orientation": -0.0,
            "rear_legs_air": -0.25,
            "heading_error": -0.5,
            "undesired_body_contact": -0.1,
        },
        "Curriculum_Config": {
            "enabled": False,
            "start_lin_vel_x": 0.5,
            "max_lin_vel_x": 0.5,
            "delta_lin_vel_x": 0.05,
            "curriculum_threshold": 0.85,
            "increase_anyway_threshold": 5000,
            "threshold_size": 30,
        },
        "Environment_Config": {
            "seed": 7,
            "use_terrain": True,
            "episode_length_s": 30.0,
            "num_envs": 4096,
            "command_range": {
                "lin_vel_x": [0.1, 0.8],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
            },
            "command_range_allowed": True,
            "terminate_on_torso_contact": True,
        },
    }

    if config_name == "config_A":
        return deepcopy(config_A)

    # Configs B-E differ from config A only by seed.
    elif config_name == "config_B":
        cfg = deepcopy(config_A)
        cfg["Environment_Config"]["seed"] = 11
        return cfg

    elif config_name == "config_C":
        cfg = deepcopy(config_A)
        cfg["Environment_Config"]["seed"] = 23
        cfg["Reward_Config"]["heading_error"] = -1.5
        return cfg

    elif config_name == "config_D":
        cfg = deepcopy(config_A)
        cfg["Environment_Config"]["seed"] = 42
        cfg["Training_Config"]["num_learning_iterations"] = 7000  # Ensure same number of iterations for config D
        return cfg

    elif config_name == "config_E":
        cfg = deepcopy(config_A)
        cfg["Environment_Config"]["seed"] = 101
        return cfg
    else:
        valid = [
            "config_A",
            "config_B",
            "config_C",
            "config_D",
            "config_E",
        ]
        raise ValueError(
            f"Unknown config_name: {config_name}. Valid configs are: {', '.join(valid)}"
        )


def get_hypothesis(config_name: str) -> str:
    hypotheses = {
        "config_A": "Selected previous config B unchanged, seed 7 (reference).",
        "config_B": "Same baseline as A, seed 11; check training reproducibility.",
        "config_C": "Same baseline as A, seed 23; check training reproducibility.",
        "config_D": "Same baseline as A, seed 42; check training reproducibility.",
        "config_E": "Same baseline as A, seed 101; check training reproducibility.",
    }

    if config_name not in hypotheses:
        raise ValueError(f"Unknown config_name: {config_name}")

    return hypotheses[config_name]


def list_configs() -> list[str]:
    return [
        "config_A",
        "config_B",
        "config_C",
        "config_D",
        "config_E",
    ]
