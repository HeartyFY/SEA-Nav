"""Play or record a trained SEA-Nav policy on the Isaac Lab Go2 navigation task."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher


SEA_NAV_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RSL_RL_ROOT = SEA_NAV_ROOT / "training" / "rsl_rl"
os.sys.path.insert(0, str(LOCAL_RSL_RL_ROOT))


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


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-Go2-Nav-Direct-v0", help="Registered Isaac Lab task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to play.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--experiment_name", type=str, default="Go2_pos_rough", help="Experiment folder under logs/sea_nav.")
parser.add_argument("--load_run", type=str, default=None, help="Run folder to load. Defaults to latest run.")
parser.add_argument("--checkpoint", type=int, default=-1, help="Checkpoint iteration to load, or -1 for latest.")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Explicit checkpoint path. Overrides run lookup.")
parser.add_argument(
    "--fixed_command",
    type=str,
    default=None,
    help="Bypass the learned policy and send a fixed high-level command as vx,vy,yaw.",
)
parser.add_argument(
    "--debug_interval",
    type=int,
    default=0,
    help="Print root height, gravity z, and command every N steps. Use 0 to disable.",
)
parser.add_argument(
    "--hold_default_pose",
    action="store_true",
    default=False,
    help="Bypass the low-level JIT policy and hold the robot default joint pose.",
)
parser.add_argument(
    "--loco_obs_order",
    choices=("reindexed", "native"),
    default="reindexed",
    help="Joint order used for low-level JIT joint position/velocity observations.",
)
parser.add_argument(
    "--loco_action_order",
    choices=("reindexed", "native"),
    default="reindexed",
    help="Joint order used when applying low-level JIT actions to Isaac Lab joints.",
)
parser.add_argument(
    "--loco_action_scale",
    type=float,
    default=None,
    help="Override env_cfg.action_scale for low-level JIT joint target offsets.",
)
parser.add_argument("--steps", type=int, default=1000, help="Playback steps. Use 0 to run until the app is closed.")
parser.add_argument("--real_time", action="store_true", default=False, help="Throttle playback to simulation dt.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--video", action="store_true", default=False, help="Record an MP4 video and exit.")
parser.add_argument("--video_length", type=int, default=600, help="Recorded video length in environment steps.")
parser.add_argument("--video_dir", type=str, default=None, help="Video output directory. Defaults beside the checkpoint.")
parser.add_argument("--video_width", type=int, default=960, help="Recorded video width.")
parser.add_argument("--video_height", type=int, default=540, help="Recorded video height.")
parser.add_argument("--num_static_obstacles", type=int, default=None, help="Override static obstacle count per env.")
parser.add_argument("--num_dynamic_obstacles", type=int, default=None, help="Override dynamic obstacle count per env.")
parser.add_argument("--obstacle_spawn_radius", type=float, default=None, help="Override obstacle spawn radius in meters.")
parser.add_argument("--obstacle_debug_vis", action="store_true", default=False, help="Show obstacle debug markers.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True
    if not getattr(args_cli, "rendering_mode_explicit", False):
        args_cli.rendering_mode = "performance"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from rsl_rl.env import IsaacLabToSeaNavAdapter  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from rsl_rl.utils import get_sea_nav_train_cfg  # noqa: E402


def main() -> None:
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    log_root = SEA_NAV_ROOT / "logs" / "sea_nav" / args_cli.experiment_name
    fixed_command = None
    if args_cli.fixed_command is not None:
        values = [float(value.strip()) for value in args_cli.fixed_command.split(",")]
        if len(values) != 3:
            raise ValueError("--fixed_command must contain exactly three comma-separated values: vx,vy,yaw")
        fixed_command = torch.tensor(values, device=device, dtype=torch.float32).unsqueeze(0)
        checkpoint_path = None
    else:
        checkpoint_path = (
            Path(args_cli.checkpoint_path).expanduser()
            if args_cli.checkpoint_path is not None
            else _checkpoint_path(log_root, args_cli.load_run, args_cli.checkpoint)
        )
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

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
    if args_cli.obstacle_debug_vis:
        env_cfg.obstacle_debug_vis = True
    if args_cli.hold_default_pose:
        env_cfg.debug_hold_default_pose = True
    env_cfg.debug_reindex_loco_observations = args_cli.loco_obs_order == "reindexed"
    env_cfg.debug_reindex_loco_actions = args_cli.loco_action_order == "reindexed"
    if args_cli.loco_action_scale is not None:
        env_cfg.action_scale = args_cli.loco_action_scale
    if args_cli.video:
        env_cfg.viewer.resolution = (args_cli.video_width, args_cli.video_height)
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (3.0, 3.0, 2.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.3)

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    if args_cli.video:
        video_dir = Path(args_cli.video_dir) if args_cli.video_dir else checkpoint_path.parent / "videos" / "play"
        video_kwargs = {
            "video_folder": str(video_dir),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording policy playback video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env)
    env = IsaacLabToSeaNavAdapter(env)

    if fixed_command is None:
        train_cfg = get_sea_nav_train_cfg(experiment_name=args_cli.experiment_name)
        runner = OnPolicyRunner(env, train_cfg, log_dir=None, args=argparse.Namespace(wandb=False), device=device)
        print(f"[INFO] Loading SEA-Nav checkpoint: {checkpoint_path}")
        runner.load(str(checkpoint_path), load_optimizer=False)
        policy = runner.get_inference_policy(device=device)
    else:
        fixed_command = fixed_command.repeat(env.num_envs, 1)
        policy = None
        print(f"[INFO] Bypassing learned policy with fixed command: {args_cli.fixed_command}")
        print(f"[INFO] Isaac Lab joint order: {env.unwrapped._robot.joint_names}")

    obs, _ = env.reset()
    dt = env.unwrapped.step_dt
    max_steps = args_cli.video_length if args_cli.video else args_cli.steps
    step_count = 0

    while simulation_app.is_running() and (max_steps <= 0 or step_count < max_steps):
        start_time = time.time()
        with torch.inference_mode():
            actions = fixed_command if fixed_command is not None else policy(obs.to(device))
            obs, _, _, _, _ = env.step(actions)
        step_count += 1
        if args_cli.debug_interval > 0 and step_count % args_cli.debug_interval == 0:
            root_height = env.unwrapped._robot.data.root_pos_w[:, 2].mean().item()
            gravity_z = env.unwrapped._robot.data.projected_gravity_b[:, 2].mean().item()
            low_level_abs_max = env.unwrapped._low_level_actions.abs().max().item()
            target_delta_abs_max = (
                env.unwrapped._processed_actions - env.unwrapped._robot.data.default_joint_pos
            ).abs().max().item()
            print(
                f"[DEBUG] step={step_count} root_z={root_height:.3f} "
                f"gravity_z={gravity_z:.3f} low_level_abs_max={low_level_abs_max:.3f} "
                f"target_delta_abs_max={target_delta_abs_max:.3f} "
                f"command={actions[0].detach().cpu().tolist()}"
            )

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()
    if args_cli.video:
        print(f"[INFO] Video written under: {video_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
