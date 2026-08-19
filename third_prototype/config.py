from copy import deepcopy


def build_configs(config_name: str) -> dict:
    """
    Rear-leg hopping experiment series.

    Goal:
        Reduce the undesired gait pattern where both rear legs leave
        the ground together, while preserving the otherwise solid
        walking behavior seen in the previous D run.

    IMPORTANT:
        These configs assume that Reward_Config supports the key

            "rear_pair_airborne"

        and that the corresponding reward function returns a positive
        penalty quantity, which is multiplied by the negative scale
        defined here.

    Recommended order:
        config_A -> config_B -> config_C -> config_D

    Logged gait metrics should include:
        Gait/front_both_air_fraction
        Gait/rear_both_air_fraction
        Gait/flight_fraction
        Gait/all_four_contact_fraction
        Gait/diagonal_support_fraction

    Keep all other code and hyperparameters unchanged between runs.
    """

    # ============================================================
    # CONFIG A — WALKING BASELINE
    #
    # Reproduces the previous D-type baseline:
    #   - solid forward walking
    #   - but noticeable simultaneous rear-leg hopping
    #
    # No rear-pair penalty yet.
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

            # New gait term.
            # Baseline: disabled.
            "rear_legs_air": 0.0,
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
    # CONFIG B — LIGHT REAR-PAIR PENALTY
    #
    # Hypothesis:
    #   A small penalty is enough to discourage simultaneous rear
    #   flight without preventing useful swing phases.
    #
    # Only change:
    #   rear_pair_airborne: 0.0 -> -0.05
    # ============================================================

    elif config_name == "config_B":
        cfg = deepcopy(config_A)
        cfg["Reward_Config"]["rear_legs_air"] = -0.05
        return cfg

    # ============================================================
    # CONFIG C — MEDIUM REAR-PAIR PENALTY
    #
    # Hypothesis:
    #   A stronger penalty further reduces the rear-leg hopping
    #   while still preserving locomotion.
    #
    # Only change:
    #   rear_legs_air: 0.0 -> -0.10
    # ============================================================

    elif config_name == "config_C":
        cfg = deepcopy(config_A)
        cfg["Reward_Config"]["rear_legs_air"] = -0.10
        return cfg

    # ============================================================
    # CONFIG D — STRONG REAR-PAIR PENALTY
    #
    # Hypothesis:
    #   Tests whether a clearly stronger penalty is required.
    #
    # Risk:
    #   The robot may avoid rear-leg flight by dragging the rear
    #   feet instead of learning a clean alternating gait.
    #
    # Only change:
    #   rear_pair_airborne: 0.0 -> -0.20
    # ============================================================

    elif config_name == "config_D":
        cfg = deepcopy(config_A)
        cfg["Reward_Config"]["rear_legs_air"] = -0.20
        return cfg

    else:
        valid = [
            "config_A",
            "config_B",
            "config_C",
            "config_D",
        ]
        raise ValueError(
            f"Unknown config_name: {config_name}. "
            f"Valid configs are: {', '.join(valid)}"
        )


def get_hypothesis(config_name: str) -> str:
    hypotheses = {
        "config_A": (
            "Walking baseline with no rear-pair airborne penalty."
        ),
        "config_B": (
            "A light rear-pair airborne penalty reduces rear-leg hopping "
            "without harming the otherwise useful gait."
        ),
        "config_C": (
            "A medium rear-pair airborne penalty produces a cleaner "
            "alternating rear-leg gait."
        ),
        "config_D": (
            "A strong rear-pair airborne penalty is required, but may "
            "cause rear-foot dragging as a new reward-hacking strategy."
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
    ]