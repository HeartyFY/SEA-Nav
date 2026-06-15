# SEA-Nav: Efficient Policy Learning for Safe and Agile Quadruped Navigation in Cluttered Environments


**Project Website**: [https://11chens.github.io/sea-nav](https://11chens.github.io/sea-nav/)

<p align="center">
  <img src="imgs/terser.jpg" width="80%">
</p>

---

## Installation

This branch targets Isaac Lab 2.2.0 on Isaac Sim 5.0.0. The legacy Isaac Gym
implementation remains under `training/legged_gym`, but the runnable migrated
environment is registered as an Isaac Lab task.

### 1. Isaac Lab environment
Use the Isaac Lab 2.2.0 environment:
```bash
cd /home/hfy/IsaacLab
./isaaclab.sh --conda isaaclab
./isaaclab.sh --install rsl_rl
```

### 2. SEA-Nav assets
Keep this repository at `/home/hfy/SEA-Nav`, or set `SEA_NAV_ROOT`/edit the
task config if you move it. The Isaac Lab task loads:
- the local Go2 URDF from `training/legged_gym/resources/go2_description`
- the migrated SEA-Nav low-level TorchScript controllers from `training/legged_gym/legged_gym/ctrl_model`

If your active Isaac Lab checkout is `/home/hfy/IsaacLab`, sync task changes with:
```bash
python3 /home/hfy/SEA-Nav/scripts/sync_isaaclab_task.py
```

---

## Usage

### Smoke test
Run offline migration checks first:
```bash
python3 /home/hfy/SEA-Nav/scripts/check_migration.py
```

Run one Go2 navigation simulation with random high-level commands:
```bash
cd /home/hfy/IsaacLab
SEA_NAV_ROOT=/home/hfy/SEA-Nav ./isaaclab.sh -p /home/hfy/SEA-Nav/scripts/smoke_go2_nav.py --num_envs 1 --steps 128 --headless
```

### Training
Train the migrated direct Isaac Lab environment with RSL-RL:
```bash
cd /home/hfy/IsaacLab
SEA_NAV_ROOT=/home/hfy/SEA-Nav ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Go2-Nav-Direct-v0 --headless
```

For a fast training smoke test:
```bash
cd /home/hfy/IsaacLab
SEA_NAV_ROOT=/home/hfy/SEA-Nav ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Go2-Nav-Direct-v0 --num_envs 2 --max_iterations 1 --headless --run_name smoke
```

---

## Deployment (Coming soon)
For instructions on deploying to real-world robots, please refer to the [deployment README](deployment/README.md).
