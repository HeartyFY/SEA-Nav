# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded smoke test for the migrated SEA-Nav Go2 Isaac Lab task."""

import argparse
import subprocess

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run a finite smoke test for the migrated SEA-Nav Go2 task.")
parser.add_argument("--task", type=str, default="Isaac-Go2-Nav-Direct-v0", help="Registered Isaac Lab task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=128, help="Number of environment steps to run.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--skip_gpu_preflight", action="store_true", default=False, help="Skip the NVIDIA driver check.")
parser.add_argument("--num_static_obstacles", type=int, default=None, help="Override static obstacle count per env.")
parser.add_argument("--num_dynamic_obstacles", type=int, default=None, help="Override dynamic obstacle count per env.")
parser.add_argument("--obstacle_spawn_radius", type=float, default=None, help="Override obstacle spawn radius in meters.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.skip_gpu_preflight:
    result = subprocess.run(["nvidia-smi"], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Isaac Sim requires a working NVIDIA driver/GPU. nvidia-smi failed: {message}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.num_static_obstacles is not None:
        env_cfg.num_static_obstacles = args_cli.num_static_obstacles
    if args_cli.num_dynamic_obstacles is not None:
        env_cfg.num_dynamic_obstacles = args_cli.num_dynamic_obstacles
    if args_cli.obstacle_spawn_radius is not None:
        env_cfg.obstacle_spawn_radius = args_cli.obstacle_spawn_radius
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    print(f"[INFO] observation shape: {tuple(obs['policy'].shape)}")
    print(f"[INFO] action space: {env.action_space}")

    reward_sum = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    for _ in range(args_cli.steps):
        with torch.inference_mode():
            actions = 2.0 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1.0
            obs, reward, terminated, truncated, _ = env.step(actions)
            reward_sum += reward
            if torch.any(terminated | truncated):
                print("[INFO] reset observed during smoke test")

    print(f"[INFO] completed {args_cli.steps} steps")
    print(f"[INFO] mean reward sum: {reward_sum.mean().item():.4f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
