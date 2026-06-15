# SEA-Nav rsl_rl for Isaac Lab

This package keeps the SEA-Nav PPO runner and adds an Isaac Lab adapter:

- `rsl_rl.env.IsaacLabToSeaNavAdapter`
- `rsl_rl.utils.get_sea_nav_train_cfg`
- `/home/hfy/SEA-Nav/scripts/train_sea_nav_isaaclab.py`

Install it in the Isaac Lab Python environment:

```bash
/home/hfy/SEA-Nav/IsaacLab/isaaclab.sh -p -m pip install -e /home/hfy/SEA-Nav/training/rsl_rl
```

Run training through the SEA-Nav launch script, not Isaac Lab's stock RSL-RL trainer:

```bash
/home/hfy/SEA-Nav/IsaacLab/isaaclab.sh -p /home/hfy/SEA-Nav/scripts/train_sea_nav_isaaclab.py \
  --task Isaac-Go2-Nav-Direct-v0 \
  --num_envs 1024 \
  --max_iterations 2000 \
  --headless
```

The adapter expects the Isaac Lab task to expose ray observations as `rays` or `_rays`.
For the migrated Go2 task in this repository it infers:

- `his_len` from `cfg.history_length`
- `num_props` from the flattened observation shape
- critic observations from `extras["observations"]["critic"]`, falling back to policy observations
