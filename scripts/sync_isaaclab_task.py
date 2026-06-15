"""Sync migrated SEA-Nav Go2 task files into the active Isaac Lab checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SEA_NAV_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTIVE_ISAACLAB_ROOT = Path("/home/hfy/IsaacLab")
VENDORED_ISAACLAB_ROOT = SEA_NAV_ROOT / "IsaacLab"

TASK_FILES = (
    "source/isaaclab_tasks/isaaclab_tasks/direct/go2_nav/go2_nav_env.py",
    "source/isaaclab_tasks/isaaclab_tasks/direct/go2_nav/go2_nav_env_cfg.py",
    "source/isaaclab_tasks/isaaclab_tasks/direct/go2_nav/agents/rsl_rl_ppo_cfg.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/go2_nav/go2_nav_env_cfg.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/go2_nav/agents/rsl_rl_ppo_cfg.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaaclab-root",
        type=Path,
        default=DEFAULT_ACTIVE_ISAACLAB_ROOT,
        help="Active Isaac Lab checkout used by the isaaclab conda environment.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print files that would be copied without writing.")
    args = parser.parse_args()

    if not VENDORED_ISAACLAB_ROOT.is_dir():
        raise FileNotFoundError(f"Missing vendored Isaac Lab checkout: {VENDORED_ISAACLAB_ROOT}")
    if not args.isaaclab_root.is_dir():
        raise FileNotFoundError(f"Missing active Isaac Lab checkout: {args.isaaclab_root}")

    for relative_path in TASK_FILES:
        source = VENDORED_ISAACLAB_ROOT / relative_path
        destination = args.isaaclab_root / relative_path
        if not source.is_file():
            raise FileNotFoundError(source)
        print(f"{source} -> {destination}")
        if not args.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    print("[INFO] sync complete" if not args.dry_run else "[INFO] dry run complete")


if __name__ == "__main__":
    main()
