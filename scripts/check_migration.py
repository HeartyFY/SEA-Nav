"""Offline checks for the SEA-Nav Isaac Lab 2.2 migration."""

from __future__ import annotations

import filecmp
import py_compile
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import torch


SEA_NAV_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ISAACLAB_ROOT = Path("/home/hfy/IsaacLab")
VENDORED_ISAACLAB_ROOT = SEA_NAV_ROOT / "IsaacLab"

TASK_FILES = (
    "source/isaaclab_tasks/isaaclab_tasks/direct/go2_nav/go2_nav_env.py",
    "source/isaaclab_tasks/isaaclab_tasks/direct/go2_nav/go2_nav_env_cfg.py",
    "source/isaaclab_tasks/isaaclab_tasks/direct/go2_nav/agents/rsl_rl_ppo_cfg.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/go2_nav/go2_nav_env_cfg.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/go2_nav/agents/rsl_rl_ppo_cfg.py",
)


def check_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def check_task_files() -> None:
    for relative_path in TASK_FILES:
        vendored_path = VENDORED_ISAACLAB_ROOT / relative_path
        active_path = ACTIVE_ISAACLAB_ROOT / relative_path
        check_file(vendored_path)
        py_compile.compile(str(vendored_path), doraise=True)
        if active_path.exists() and not filecmp.cmp(vendored_path, active_path, shallow=False):
            raise RuntimeError(f"Active IsaacLab copy differs from SEA-Nav copy: {relative_path}")


def check_low_level_policy() -> None:
    policy_dir = SEA_NAV_ROOT / "training/legged_gym/legged_gym/ctrl_model"
    encoder_vel = torch.jit.load(str(policy_dir / "encoder_vel.jit"), map_location="cpu").eval()
    encoder_latent = torch.jit.load(str(policy_dir / "encoder_latent.jit"), map_location="cpu").eval()
    body = torch.jit.load(str(policy_dir / "body_latest.jit"), map_location="cpu").eval()

    history = torch.zeros(2, 450)
    prop = torch.zeros(2, 45)
    yaw = torch.zeros(2, 1)
    velocity = encoder_vel(history)
    latent = encoder_latent(history)
    actions = body(torch.cat((velocity, prop, yaw, latent), dim=-1))

    if velocity.shape != (2, 2):
        raise RuntimeError(f"Unexpected encoder_vel output: {velocity.shape}")
    if latent.shape != (2, 16):
        raise RuntimeError(f"Unexpected encoder_latent output: {latent.shape}")
    if actions.shape != (2, 12):
        raise RuntimeError(f"Unexpected body policy output: {actions.shape}")
    if not torch.isfinite(actions).all():
        raise RuntimeError("Low-level policy produced non-finite actions")


def check_go2_asset_and_spaces() -> None:
    urdf_path = SEA_NAV_ROOT / "training/legged_gym/resources/go2_description/urdf/go2_description_v8.urdf"
    check_file(urdf_path)
    root = ET.parse(urdf_path).getroot()
    actuated_joints = [joint.attrib["name"] for joint in root.findall("joint") if joint.attrib.get("type") != "fixed"]
    expected_joints = [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ]
    if actuated_joints != expected_joints:
        raise RuntimeError(f"Unexpected Go2 joint order: {actuated_joints}")

    cfg_path = VENDORED_ISAACLAB_ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/go2_nav/go2_nav_env_cfg.py"
    env_path = VENDORED_ISAACLAB_ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/go2_nav/go2_nav_env.py"
    cfg_text = cfg_path.read_text()
    env_text = env_path.read_text()
    required_cfg_patterns = {
        "action_space = 3": r"action_space\s*=\s*3",
        "observation_space = 550": r"observation_space\s*=\s*550",
        "history_length = 10": r"history_length\s*=\s*10",
        "num_rays = 41": r"num_rays\s*=\s*41",
    }
    for label, pattern in required_cfg_patterns.items():
        if re.search(pattern, cfg_text) is None:
            raise RuntimeError(f"Missing expected config invariant: {label}")
    if "self._reindex(self._low_level_actions)" not in env_text:
        raise RuntimeError("Low-level policy actions must be reindexed before joint targets")
    if "self._update_history" not in env_text:
        raise RuntimeError("History buffers must use reset-aware update logic")


def check_task_registration() -> None:
    direct_init = (
        VENDORED_ISAACLAB_ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/go2_nav/__init__.py"
    ).read_text()
    manager_init = (
        VENDORED_ISAACLAB_ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/go2_nav/__init__.py"
    ).read_text()
    required_snippets = {
        "direct id": "Isaac-Go2-Nav-Direct-v0",
        "direct env": "go2_nav_env:Go2NavEnv",
        "direct cfg": "go2_nav_env_cfg:Go2NavEnvCfg",
        "manager id": "Isaac-Go2-Nav-v0",
        "manager env": "isaaclab.envs:ManagerBasedRLEnv",
        "manager cfg": "go2_nav_env_cfg:Go2NavEnvCfg",
    }
    combined = direct_init + "\n" + manager_init
    for label, snippet in required_snippets.items():
        if snippet not in combined:
            raise RuntimeError(f"Missing task registration snippet ({label}): {snippet}")


def check_gpu() -> None:
    result = subprocess.run(["nvidia-smi"], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        print("[WARN] nvidia-smi failed; Isaac Sim cannot run until the NVIDIA driver/GPU is available.")
        print(result.stderr.strip() or result.stdout.strip())
    else:
        print("[INFO] nvidia-smi ok")


def main() -> None:
    check_task_files()
    check_low_level_policy()
    check_go2_asset_and_spaces()
    check_task_registration()
    check_gpu()
    print("[INFO] offline migration checks passed")


if __name__ == "__main__":
    main()
