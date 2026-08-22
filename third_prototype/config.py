from copy import deepcopy


def build_configs(config_name: str) -> dict:
    """
    Exploration / policy-std sweep.

    Goal:
        Test whether the repeatedly observed rise of Policy/mean_std toward 1.0
        is responsible for unstable training and poor deterministic evaluation.

    Baseline:
        - orientation = -0.25
        - heading_error = 0.0
        - rear_legs_air = 0.0
        - fixed command = 0.5 m/s
        - terrain enabled
        - fixed PPO learning rate = 3e-4

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
    #   entropy_coef = 0.005
    #   learn_std = True
    #   std_range = [0.05, 1.0]
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
                "entropy_coef": 0.005,
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
                    "std_range": [0.05, 1.0],
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
            "rear_legs_air": 0.0,
            "heading_error": 0.0,
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
            "seed": 1,
            "use_terrain": True,
            "episode_length_s": 30.0,
            "num_envs": 4096,
        },
    }

    if config_name == "config_A":
        return deepcopy(config_A)

    # ============================================================
    # CONFIG B — LOWER ENTROPY PRESSURE
    #
    # Only change:
    #   entropy_coef: 0.005 -> 0.003
    # ============================================================
    elif config_name == "config_B":
        cfg = deepcopy(config_A)
        cfg["Training_Config"]["algorithm"]["entropy_coef"] = 0.003
        return cfg

    # ============================================================
    # CONFIG C — EVEN LOWER ENTROPY PRESSURE
    #
    # Only change:
    #   entropy_coef: 0.005 -> 0.002
    # ============================================================
    elif config_name == "config_C":
        cfg = deepcopy(config_A)
        cfg["Training_Config"]["algorithm"]["entropy_coef"] = 0.002
        return cfg

    # ============================================================
    # CONFIG D — MODERATE ENTROPY + MODERATE STD CAP
    #
    # Changes:
    #   entropy_coef: 0.005 -> 0.003
    #   std_max:      1.0   -> 0.75
    # ============================================================
    elif config_name == "config_D":
        cfg = deepcopy(config_A)
        cfg["Training_Config"]["algorithm"]["entropy_coef"] = 0.003
        cfg["Training_Config"]["actor"]["distribution_cfg"]["std_range"] = [0.05, 0.75]
        return cfg

    # ============================================================
    # CONFIG E — LOWER ENTROPY + MODERATE STD CAP
    #
    # Changes:
    #   entropy_coef: 0.005 -> 0.002
    #   std_max:      1.0   -> 0.75
    # ============================================================
    elif config_name == "config_E":
        cfg = deepcopy(config_A)
        cfg["Training_Config"]["algorithm"]["entropy_coef"] = 0.002
        cfg["Training_Config"]["actor"]["distribution_cfg"]["std_range"] = [0.05, 0.75]
        return cfg

    # ============================================================
    # CONFIG F — FIXED EXPLORATION CONTROL
    #
    # Purpose:
    #   Test whether a stable fixed exploration level produces a better
    #   deterministic mean policy than the learnable-std variants.
    #
    # Changes:
    #   entropy_coef = 0.0
    #   init_std = 0.30
    #   learn_std = False
    #
    # std_range remains present but has no practical role while std is fixed.
    # ============================================================
    elif config_name == "config_F":
        cfg = deepcopy(config_A)

        cfg["Training_Config"]["algorithm"]["entropy_coef"] = 0.0

        distribution_cfg = cfg["Training_Config"]["actor"]["distribution_cfg"]
        distribution_cfg["init_std"] = 0.30
        distribution_cfg["learn_std"] = False
        distribution_cfg["std_range"] = [0.05, 1.0]

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
            f"Unknown config_name: {config_name}. "
            f"Valid configs are: {', '.join(valid)}"
        )


def get_hypothesis(config_name: str) -> str:
    hypotheses = {
        "config_A": (
            "Reference exploration setup with entropy 0.005 and learnable std "
            "up to 1.0."
        ),
        "config_B": (
            "Reducing entropy to 0.003 prevents std from rising excessively "
            "while preserving enough exploration for gait discovery."
        ),
        "config_C": (
            "Reducing entropy to 0.002 further improves deterministic-policy "
            "quality without collapsing exploration."
        ),
        "config_D": (
            "Entropy 0.003 plus std_max 0.75 provides a stable compromise "
            "between exploration and deterministic gait quality."
        ),
        "config_E": (
            "Entropy 0.002 plus std_max 0.75 provides a more conservative "
            "learnable exploration regime."
        ),
        "config_F": (
            "A fixed std of 0.30 tests whether learned exploration itself is "
            "causing the gap between training behavior and deterministic eval."
        ),
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