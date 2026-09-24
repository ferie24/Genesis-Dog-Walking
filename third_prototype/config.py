from copy import deepcopy


def build_configs(config_name: str) -> dict:
    """
    Exploration / policy-std sweep.

    Goal:
        Test whether the repeatedly observed rise of Policy/mean_std toward 1.0
        is responsible for unstable training and poor deterministic evaluation.

    Baseline:
        - orientation = -0.25
        - heading_error = -1.0
        - rear_legs_air = -1.0
        - undesired_body_contact = -0.1
        - forward command range = 0.2–0.8 m/s
        - terrain enabled
        - fixed PPO learning rate = 3e-4
        - entropy_coef = 0.002, learnable std in [0.05, 0.75]

    Existing variant overrides are retained; C and E match the baseline,
    and B and D are equivalent.

    Suggested order:
        config_A -> config_B -> config_C -> config_D -> config_E -> config_F

    Important diagnostics:
        Policy/mean_std
        Train/mean_reward
        Train/mean_episode_length
        Loss/value

        Gait/rear_both_air
        Gait/diagonal_support
        Gait/undesired_contact_fraction

        heading_error_abs_mean
        roll_termination
        pitch_termination
        fall_termination

    Config F is a control experiment:
        fixed Gaussian std = 0.30
        learn_std = False
        entropy_coef = 0.0
    """

    # ============================================================
    # CONFIG A — CURRENT EXPLORATION BASELINE
    #
    # Reference:
    #   entropy_coef = 0.002
    #   learn_std = True
    #   std_range = [0.05, 0.75]
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
            "num_learning_iterations": 4000,
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
            "tracking_lin_vel_x": 2.0,
            "tracking_ang_vel": 1.0,
            "lin_vel_z": -1.0,
            "lin_vel_y": -5.0,
            "action_rate": -0.001,
            "similar_to_default": 0.0,
            "sideway_movement": 0.0,
            "tracking_sigma": 0.1,
            "x_progress": 0.5,
            # Keep these fixed during the exploration sweep.
            "orientation": -0.25,
            "rear_legs_air": -1.0,
            "heading_error": -1.0,
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
                "lin_vel_x": [0.2, 0.8],
                "lin_vel_y": [0.0, 0.0],
                "ang_vel_yaw": [0.0, 0.0],
            },
            "command_range_allowed": True,
            "terminate_on_torso_contact": False,
        },
    }

    if config_name == "config_A":
        return deepcopy(config_A)

    # Configs B-F differ from config A only by seed.
    elif config_name == "config_B":
        cfg = deepcopy(config_A)
        cfg["Environment_Config"]["seed"] = 2
        return cfg

    elif config_name == "config_C":
        cfg = deepcopy(config_A)
        cfg["Environment_Config"]["seed"] = 3
        return cfg

    elif config_name == "config_D":
        cfg = deepcopy(config_A)
        cfg["Environment_Config"]["seed"] = 4
        return cfg

    elif config_name == "config_E":
        cfg = deepcopy(config_A)
        cfg["Environment_Config"]["seed"] = 5
        return cfg

    elif config_name == "config_F":
        cfg = deepcopy(config_A)
        cfg["Environment_Config"]["seed"] = 6
        return cfg
    elif config_name == "config_G":
        cfg = deepcopy(config_A)
        cfg["Environment_Config"]["seed"] = 1
        cfg["command_range"] = {
            "lin_vel_x": [0.0, 1.0],
            "lin_vel_y": [0.0, 0.0],
            "ang_vel_yaw": [0.0, 0.0],
        }
        return cfg
    else:
        valid = [
            "config_A",
            "config_B",
            "config_C",
            "config_D",
            "config_E",
            "config_F",
        ]
        raise ValueError(
            f"Unknown config_name: {config_name}. Valid configs are: {', '.join(valid)}"
        )


def get_hypothesis(config_name: str) -> str:
    hypotheses = {
        "config_A": (
            "Reference exploration setup with entropy 0.002 and learnable std "
            "up to 0.75, forward commands from 0.2 to 0.8 m/s, and heading, "
            "rear-leg air, and undesired-body-contact penalties."
        ),
        "config_B": "Same configuration as A with seed 2.",
        "config_C": "Same configuration as A with seed 3.",
        "config_D": "Same configuration as A with seed 4.",
        "config_E": "Same configuration as A with seed 5.",
        "config_F": "Same configuration as A with seed 6.",
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
        "config_F",
    ]
