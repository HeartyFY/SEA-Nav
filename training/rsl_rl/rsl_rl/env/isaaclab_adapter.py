"""Compatibility adapter from Isaac Lab's RSL-RL wrapper to SEA-Nav PPO."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch


class _CfgProxy:
    """Expose ``cfg.env.his_len`` while preserving access to the Isaac Lab cfg."""

    def __init__(self, cfg: Any, his_len: int):
        self._cfg = cfg
        env_cfg = getattr(cfg, "env", None)
        if env_cfg is None:
            env_cfg = SimpleNamespace()
        self.env = SimpleNamespace(**getattr(env_cfg, "__dict__", {}))
        self.env.his_len = his_len

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cfg, name)


class IsaacLabToSeaNavAdapter:
    """Adapt Isaac Lab 2.2 ``RslRlVecEnvWrapper`` output to SEA-Nav's old runner API.

    The SEA-Nav runner expects the Legged-Gym-style environment API where
    ``get_observations()`` returns only a tensor and ``step()`` returns
    ``obs, privileged_obs, rewards, dones, infos``. Isaac Lab's wrapper returns
    ``(obs, extras)`` from ``get_observations()`` and keeps critic observations
    inside ``extras["observations"]``. This adapter translates only that surface.
    """

    def __init__(self, env: Any, num_props: int | None = None, his_len: int | None = None):
        self.env = env
        self.unwrapped = env.unwrapped
        self.device = self.unwrapped.device
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.num_nav_actions = self.num_actions
        self.max_episode_length = env.max_episode_length

        self._last_obs: torch.Tensor | None = None
        self._last_extras: dict = {}
        obs, extras = self._refresh_observations()
        self.num_obs = obs.shape[1]
        self.num_privileged_obs = self._critic_obs(obs, extras).shape[1]

        self.rays = self._resolve_rays()
        self.num_props = num_props if num_props is not None else self._resolve_num_props(obs, his_len)
        resolved_his_len = his_len if his_len is not None else self._resolve_his_len(obs)
        self.cfg = _CfgProxy(self.unwrapped.cfg, resolved_his_len)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_length_buf = value

    def get_observations(self) -> torch.Tensor:
        return self._last_obs

    def get_privileged_observations(self) -> torch.Tensor:
        return self._critic_obs(self._last_obs, self._last_extras)

    def get_extras(self) -> dict:
        return self._last_extras

    def get_live_metrics(self) -> dict:
        metric_fn = getattr(self.unwrapped, "get_live_metrics", None)
        if metric_fn is None:
            return {}
        return metric_fn()

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs, extras = self._split_reset(self.env.reset())
        self._set_last(obs, extras)
        return obs, extras

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        result = self.env.step(actions)
        if len(result) == 4:
            obs, rewards, dones, infos = result
            infos = dict(infos)
        elif len(result) == 5:
            obs_dict, rewards, terminated, truncated, infos = result
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            dones = (terminated | truncated).to(dtype=torch.long)
            infos = dict(infos)
            if isinstance(obs_dict, dict):
                infos["observations"] = obs_dict
        else:
            raise RuntimeError(f"Unexpected Isaac Lab step return length: {len(result)}")

        if torch.any(dones > 0):
            env_log = getattr(self.unwrapped, "extras", {}).get("log")
            if env_log is not None:
                infos["log"] = env_log
                infos["episode"] = env_log
            elif "log" in infos:
                infos["episode"] = infos["log"]
        privileged_obs = self._critic_obs(obs, infos)
        self._set_last(obs, infos)
        return obs, privileged_obs, rewards, dones, infos

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: int = -1) -> int:
        return self.env.seed(seed)

    def _split_observations(self, result: Any) -> tuple[torch.Tensor, dict]:
        if isinstance(result, tuple):
            obs, extras = result
        else:
            obs, extras = result, {}
        if isinstance(obs, dict):
            extras = dict(extras)
            extras.setdefault("observations", obs)
            obs = obs["policy"]
        return obs, extras

    def _refresh_observations(self) -> tuple[torch.Tensor, dict]:
        obs, extras = self._split_observations(self.env.get_observations())
        self._set_last(obs, extras)
        return obs, extras

    def _set_last(self, obs: torch.Tensor, extras: dict) -> None:
        self._last_obs = obs
        self._last_extras = extras

    def _split_reset(self, result: Any) -> tuple[torch.Tensor, dict]:
        obs, extras = result
        if isinstance(obs, dict):
            extras = dict(extras)
            extras.setdefault("observations", obs)
            obs = obs["policy"]
        return obs, extras

    def _critic_obs(self, obs: torch.Tensor, extras: dict) -> torch.Tensor:
        observations = extras.get("observations", {})
        if isinstance(observations, dict):
            return observations.get("critic", obs)
        return obs

    def _resolve_rays(self) -> torch.Tensor:
        for name in ("rays", "_rays"):
            rays = getattr(self.unwrapped, name, None)
            if rays is not None:
                return rays
        raise AttributeError(
            "SEA-Nav PPO requires the Isaac Lab task to expose ray observations as "
            "`rays` or `_rays` on the unwrapped environment."
        )

    def _resolve_his_len(self, obs: torch.Tensor) -> int:
        cfg = self.unwrapped.cfg
        if hasattr(cfg, "env") and hasattr(cfg.env, "his_len"):
            return int(cfg.env.his_len)
        if hasattr(cfg, "history_length"):
            return int(cfg.history_length)
        num_rays = int(self.rays.shape[1])
        one_step_obs = num_rays + 12 + 2
        if obs.shape[1] % one_step_obs == 0:
            return int(obs.shape[1] // one_step_obs)
        raise AttributeError("Could not infer SEA-Nav observation history length from the Isaac Lab task.")

    def _resolve_num_props(self, obs: torch.Tensor, his_len: int | None) -> int:
        cfg = self.unwrapped.cfg
        if hasattr(self.unwrapped, "num_props"):
            return int(self.unwrapped.num_props)
        if hasattr(cfg, "num_props"):
            return int(cfg.num_props)
        if hasattr(cfg, "env") and hasattr(cfg.env, "num_props"):
            return int(cfg.env.num_props)

        resolved_his_len = his_len if his_len is not None else self._resolve_his_len(obs)
        if obs.shape[1] % resolved_his_len != 0:
            raise AttributeError("Could not infer SEA-Nav proprioceptive observation width.")
        num_goal_obs = int(getattr(cfg, "num_goal_obs", 2))
        return int(obs.shape[1] // resolved_his_len - self.rays.shape[1] - num_goal_obs)
