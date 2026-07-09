"""Default SEA-Nav PPO config for the custom runner."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


_DEFAULT_TRAIN_CFG: dict[str, dict[str, Any]] = {
    "policy": {
        "init_noise_std": 0.6,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
    },
    "algorithm": {
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.003,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 5.0e-4,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    },
    "runner": {
        "policy_class_name": "DifferentiableSafeActorCritic",
        "algorithm_class_name": "PPO",
        "num_steps_per_env": 48,
        "max_iterations": 2000,
        "save_interval": 100,
        "experiment_name": "Go2_pos_rough",
        "run_name": "",
    },
}


def get_sea_nav_train_cfg(**runner_overrides: Any) -> dict[str, dict[str, Any]]:
    """Return a mutable SEA-Nav PPO config dictionary.

    Keyword arguments override entries in the ``runner`` section, matching the
    shape expected by ``rsl_rl.runners.OnPolicyRunner`` in this repository.
    """

    cfg = deepcopy(_DEFAULT_TRAIN_CFG)
    for key, value in runner_overrides.items():
        if value is not None:
            cfg["runner"][key] = value
    return cfg
