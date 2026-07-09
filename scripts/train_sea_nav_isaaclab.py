"""Train SEA-Nav PPO on an Isaac Lab task using the modified local rsl_rl."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


SEA_NAV_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RSL_RL_ROOT = SEA_NAV_ROOT / "training" / "rsl_rl"
sys.path.insert(0, str(LOCAL_RSL_RL_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-Go2-Nav-Direct-v0", help="Registered Isaac Lab task name.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel Isaac Lab environments.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Number of SEA-Nav PPO iterations.")
parser.add_argument("--experiment_name", type=str, default=None, help="Experiment folder name under logs/sea_nav.")
parser.add_argument("--run_name", type=str, default="", help="Optional suffix for the run folder.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume from a SEA-Nav PPO checkpoint.")
parser.add_argument("--load_run", type=str, default=None, help="Run folder name to load when resuming.")
parser.add_argument("--checkpoint", type=int, default=-1, help="Checkpoint iteration to load, or -1 for latest.")
parser.add_argument("--wandb", action="store_true", default=False, help="Enable the runner's wandb logging path.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_static_obstacles", type=int, default=None, help="Override static obstacle count per env.")
parser.add_argument("--num_dynamic_obstacles", type=int, default=None, help="Override dynamic obstacle count per env.")
parser.add_argument("--obstacle_spawn_radius", type=float, default=None, help="Override obstacle spawn radius in meters.")
parser.add_argument("--collision_penalty", type=float, default=None, help="Override obstacle collision reward penalty.")
parser.add_argument("--action_scale", type=float, default=None, help="Override low-level locomotion action scale.")
parser.add_argument("--command_filter_alpha", type=float, default=None, help="Override high-level command filter alpha.")
parser.add_argument("--command_limit_vx", type=str, default=None, help='Override vx command range as "min,max".')
parser.add_argument("--command_limit_vy", type=str, default=None, help='Override vy command range as "min,max".')
parser.add_argument("--command_limit_yaw", type=str, default=None, help='Override yaw command range as "min,max".')
parser.add_argument(
    "--obstacle_clearance_reward_scale",
    type=float,
    default=None,
    help="Override obstacle clearance reward scale.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.env import IsaacLabToSeaNavAdapter  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from rsl_rl.utils import get_sea_nav_train_cfg  # noqa: E402


def _latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[-1]))
    if not checkpoints:
        raise FileNotFoundError(f"No SEA-Nav checkpoints found in {run_dir}")
    return checkpoints[-1]


def _checkpoint_path(log_root: Path, load_run: str | None, checkpoint: int) -> Path:
    if load_run is None:
        runs = sorted(path for path in log_root.iterdir() if path.is_dir())
        if not runs:
            raise FileNotFoundError(f"No SEA-Nav runs found in {log_root}")
        run_dir = runs[-1]
    else:
        run_dir = log_root / load_run
    if checkpoint == -1:
        return _latest_checkpoint(run_dir)
    return run_dir / f"model_{checkpoint}.pt"


def _parse_range(value: str, name: str) -> tuple[float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{name} must be formatted as min,max")
    if parts[0] >= parts[1]:
        raise ValueError(f"{name} lower bound must be smaller than upper bound")
    return parts[0], parts[1]


def main() -> None:
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    if args_cli.num_static_obstacles is not None:
        env_cfg.num_static_obstacles = args_cli.num_static_obstacles
    if args_cli.num_dynamic_obstacles is not None:
        env_cfg.num_dynamic_obstacles = args_cli.num_dynamic_obstacles
    if args_cli.obstacle_spawn_radius is not None:
        env_cfg.obstacle_spawn_radius = args_cli.obstacle_spawn_radius
    if args_cli.collision_penalty is not None:
        env_cfg.collision_penalty = args_cli.collision_penalty
    if args_cli.obstacle_clearance_reward_scale is not None:
        env_cfg.obstacle_clearance_reward_scale = args_cli.obstacle_clearance_reward_scale
    if args_cli.action_scale is not None:
        env_cfg.action_scale = args_cli.action_scale
    if args_cli.command_filter_alpha is not None:
        env_cfg.command_filter_alpha = args_cli.command_filter_alpha
    command_limits = list(env_cfg.command_limits)
    if args_cli.command_limit_vx is not None:
        command_limits[0] = _parse_range(args_cli.command_limit_vx, "--command_limit_vx")
    if args_cli.command_limit_vy is not None:
        command_limits[1] = _parse_range(args_cli.command_limit_vy, "--command_limit_vy")
    if args_cli.command_limit_yaw is not None:
        command_limits[2] = _parse_range(args_cli.command_limit_yaw, "--command_limit_yaw")
    env_cfg.command_limits = tuple(command_limits)

    train_cfg = get_sea_nav_train_cfg(
        max_iterations=args_cli.max_iterations,
        experiment_name=args_cli.experiment_name,
        run_name=args_cli.run_name,
    )
    experiment_name = train_cfg["runner"]["experiment_name"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{args_cli.run_name}" if args_cli.run_name else ""
    log_root = SEA_NAV_ROOT / "logs" / "sea_nav" / experiment_name
    log_dir = log_root / f"{timestamp}{suffix}"
    (log_dir / "params").mkdir(parents=True, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    env = IsaacLabToSeaNavAdapter(env)

    runner_args = SimpleNamespace(wandb=args_cli.wandb, rl_device=device)
    runner = OnPolicyRunner(env, train_cfg, log_dir=str(log_dir), args=runner_args, device=device)

    if args_cli.resume:
        resume_path = _checkpoint_path(log_root, args_cli.load_run, args_cli.checkpoint)
        print(f"[INFO] Loading SEA-Nav checkpoint: {resume_path}")
        runner.load(str(resume_path))

    with (log_dir / "params" / "agent.json").open("w") as file:
        json.dump(train_cfg, file, indent=2)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    runner.learn(
        num_learning_iterations=train_cfg["runner"]["max_iterations"],
        init_at_random_ep_len=True,
        config=train_cfg,
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
