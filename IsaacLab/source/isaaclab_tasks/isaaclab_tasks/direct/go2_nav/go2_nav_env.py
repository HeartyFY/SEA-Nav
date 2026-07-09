# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, sample_uniform, yaw_quat

from .go2_nav_env_cfg import Go2NavEnvCfg


class Go2NavEnv(DirectRLEnv):
    cfg: Go2NavEnvCfg

    def __init__(self, cfg: Go2NavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._nav_actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._filtered_commands = torch.zeros_like(self._nav_actions)
        self._low_level_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._previous_low_level_actions = torch.zeros_like(self._low_level_actions)

        self._loco_obs_history = torch.zeros(self.num_envs, self.cfg.history_length, 45, device=self.device)
        self._obs_history = torch.zeros(self.num_envs, self.cfg.history_length, 55, device=self.device)
        self._rays = torch.full((self.num_envs, self.cfg.num_rays), self.cfg.max_ray_distance, device=self.device)
        self._goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._previous_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self._collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._min_obstacle_clearance = torch.full((self.num_envs,), self.cfg.max_ray_distance, device=self.device)
        self._last_reward_terms = {}
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "alive",
                "progress",
                "heading",
                "goal",
                "obstacle_clearance",
                "action_rate",
                "joint_velocity",
                "termination",
                "collision",
            ]
        }

        self._static_obstacle_pos_w = torch.zeros(
            self.num_envs, self.cfg.num_static_obstacles, 2, device=self.device
        )
        self._dynamic_obstacle_pos_w = torch.zeros(
            self.num_envs, self.cfg.num_dynamic_obstacles, 2, device=self.device
        )
        self._dynamic_obstacle_vel_w = torch.zeros_like(self._dynamic_obstacle_pos_w)
        self._static_obstacle_radius = torch.zeros(self.num_envs, self.cfg.num_static_obstacles, device=self.device)
        self._dynamic_obstacle_radius = torch.zeros(self.num_envs, self.cfg.num_dynamic_obstacles, device=self.device)

        ray_angles = torch.linspace(
            -torch.deg2rad(torch.tensor(self.cfg.ray_fov_deg, device=self.device)) / 2.0,
            torch.deg2rad(torch.tensor(self.cfg.ray_fov_deg, device=self.device)) / 2.0,
            self.cfg.num_rays,
            device=self.device,
        )
        self._ray_dirs_b = torch.stack((torch.cos(ray_angles), torch.sin(ray_angles)), dim=-1)

        command_limits = torch.tensor(self.cfg.command_limits, device=self.device)
        self._command_lower = command_limits[:, 0]
        self._command_upper = command_limits[:, 1]
        self._load_low_level_policy()
        self.set_debug_vis(self.cfg.obstacle_debug_vis)

    def _load_low_level_policy(self) -> None:
        policy_dir = Path(self.cfg.low_level_policy_dir)
        required_files = {
            "body": policy_dir / "body_latest.jit",
            "encoder_vel": policy_dir / "encoder_vel.jit",
            "encoder_latent": policy_dir / "encoder_latent.jit",
        }
        missing_files = [str(path) for path in required_files.values() if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(f"Missing SEA-Nav low-level policy file(s): {missing_files}")

        self._loco_policy = torch.jit.load(str(required_files["body"]), map_location=self.device).eval()
        self._encoder_vel = torch.jit.load(str(required_files["encoder_vel"]), map_location=self.device).eval()
        self._encoder_latent = torch.jit.load(str(required_files["encoder_latent"]), map_location=self.device).eval()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._update_dynamic_obstacles()
        self._nav_actions = actions.clone().clamp(-3.0, 3.0)
        alpha = self.cfg.command_filter_alpha
        self._filtered_commands = alpha * self._nav_actions + (1.0 - alpha) * self._filtered_commands
        self._filtered_commands = torch.max(torch.min(self._filtered_commands, self._command_upper), self._command_lower)
        if self.cfg.debug_hold_default_pose:
            self._low_level_actions.zero_()
            self._processed_actions = self._robot.data.default_joint_pos
            return
        self._low_level_actions = self._compute_low_level_actions(self._filtered_commands).clamp(-100.0, 100.0)
        self._processed_actions = (
            self.cfg.action_scale * self._maybe_reindex_actions(self._low_level_actions)
            + self._robot.data.default_joint_pos
        )

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._processed_actions)

    def _compute_low_level_actions(self, commands: torch.Tensor) -> torch.Tensor:
        scale = torch.tensor(
            [self.cfg.lin_vel_obs_scale, self.cfg.lin_vel_obs_scale, self.cfg.ang_vel_obs_scale],
            device=self.device,
        )
        loco_obs = torch.cat(
            (
                self._robot.data.root_ang_vel_b * self.cfg.ang_vel_obs_scale,
                self._robot.data.projected_gravity_b,
                commands * scale,
                self._maybe_reindex_observations(self._robot.data.joint_pos - self._robot.data.default_joint_pos)
                * self.cfg.dof_pos_obs_scale,
                self._maybe_reindex_observations(self._robot.data.joint_vel) * self.cfg.dof_vel_obs_scale,
                self._previous_low_level_actions,
            ),
            dim=-1,
        )
        self._loco_obs_history = self._update_history(self._loco_obs_history, loco_obs)

        history = self._loco_obs_history.reshape(self.num_envs, -1)
        base_lin_vel_pred = self._encoder_vel(history)
        latent = self._encoder_latent(history)
        actor_obs = torch.cat(
            (base_lin_vel_pred, loco_obs, self._robot.data.root_ang_vel_b[:, 2:3] * self.cfg.ang_vel_obs_scale, latent),
            dim=-1,
        )
        low_level_actions = self._loco_policy(actor_obs)
        return low_level_actions

    def _get_observations(self) -> dict:
        self._update_obstacle_rays()
        goal_local_pos = self._compute_goal_local_pos()
        prop = torch.cat(
            (
                self._robot.data.projected_gravity_b,
                self._filtered_commands,
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
            ),
            dim=-1,
        )
        obs = torch.cat(
            (
                prop,
                torch.log2(self._rays.clamp(min=self.cfg.min_ray_distance, max=self.cfg.max_ray_distance)),
                goal_local_pos,
            ),
            dim=-1,
        )
        self._obs_history = self._update_history(self._obs_history, obs)
        return {"policy": self._obs_history.reshape(self.num_envs, -1)}

    def _get_rewards(self) -> torch.Tensor:
        goal_distance = self._goal_distance()
        progress = self._previous_goal_distance - goal_distance
        self._previous_goal_distance = goal_distance.detach()
        heading_alignment = torch.nn.functional.cosine_similarity(
            self._robot.data.root_lin_vel_b[:, :2],
            self._compute_goal_local_pos(),
            dim=-1,
            eps=1.0e-6,
        )
        action_rate = torch.sum(torch.square(self._low_level_actions - self._previous_low_level_actions), dim=1)
        clearance_violation = torch.clamp(self.cfg.obstacle_clearance - self._min_obstacle_clearance, min=0.0)
        reward_terms = {
            "alive": torch.full_like(goal_distance, self.cfg.alive_reward_scale),
            "progress": self.cfg.progress_reward_scale * progress,
            "heading": self.cfg.heading_reward_scale * heading_alignment,
            "goal": self.cfg.goal_reward_scale
            * self.reset_terminated.float()
            * (goal_distance < self.cfg.goal_reached_threshold),
            "obstacle_clearance": self.cfg.obstacle_clearance_reward_scale * torch.square(clearance_violation),
            "action_rate": self.cfg.action_rate_reward_scale * action_rate,
            "joint_velocity": self.cfg.joint_velocity_reward_scale * torch.sum(torch.square(self._robot.data.joint_vel), dim=1),
            "termination": self.cfg.termination_penalty
            * self.reset_terminated.float()
            * (goal_distance >= self.cfg.goal_reached_threshold),
            "collision": self.cfg.collision_penalty * self._collision.float(),
        }
        rewards = torch.sum(torch.stack(list(reward_terms.values())), dim=0)
        for key, value in reward_terms.items():
            self._episode_sums[key] += value
        self._last_reward_terms = {key: value.detach() for key, value in reward_terms.items()}
        self._previous_low_level_actions = self._low_level_actions.clone()
        return rewards

    def get_live_metrics(self) -> dict:
        goal_distance = self._goal_distance()
        base_too_low = self._robot.data.root_pos_w[:, 2] < self.cfg.termination_height
        tipped_over = self._robot.data.projected_gravity_b[:, 2] > self.cfg.termination_projected_gravity_z
        metrics = {
            "Metrics/current_success_rate": torch.mean((goal_distance < self.cfg.goal_reached_threshold).float()),
            "Metrics/current_collision_rate": torch.mean(self._collision.float()),
            "Metrics/current_fall_tip_rate": torch.mean((base_too_low | tipped_over).float()),
            "Metrics/current_mean_min_clearance": torch.mean(self._min_obstacle_clearance),
            "Metrics/current_goal_distance": torch.mean(goal_distance),
        }
        for key, value in self._last_reward_terms.items():
            metrics[f"Reward_Terms/{key}"] = torch.mean(value)
        return metrics

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        goal_reached = self._goal_distance() < self.cfg.goal_reached_threshold
        base_too_low = self._robot.data.root_pos_w[:, 2] < self.cfg.termination_height
        tipped_over = self._robot.data.projected_gravity_b[:, 2] > self.cfg.termination_projected_gravity_z
        self._collision, self._min_obstacle_clearance = self._obstacle_collision_and_clearance()
        return goal_reached | base_too_low | tipped_over | self._collision, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._log_episode_metrics(env_ids)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._nav_actions[env_ids] = 0.0
        self._filtered_commands[env_ids] = 0.0
        self._low_level_actions[env_ids] = 0.0
        self._previous_low_level_actions[env_ids] = 0.0
        self._loco_obs_history[env_ids] = 0.0
        self._obs_history[env_ids] = 0.0
        self._rays[env_ids] = self.cfg.max_ray_distance
        self._collision[env_ids] = False
        self._min_obstacle_clearance[env_ids] = self.cfg.max_ray_distance

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._terrain.env_origins[env_ids]

        yaw = sample_uniform(-3.14159, 3.14159, (len(env_ids),), self.device)
        zeros = torch.zeros_like(yaw)
        root_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
        root_state[:, 7:] = 0.0

        self._sample_goals(env_ids, root_state[:, :3])
        self._sample_obstacles(env_ids, root_state[:, :3])
        self._previous_goal_distance[env_ids] = torch.linalg.norm(
            self._goal_pos_w[env_ids, :2] - root_state[:, :2],
            dim=1,
        )

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._update_obstacle_rays()

    def _log_episode_metrics(self, env_ids: torch.Tensor) -> None:
        extras = {}
        for key in self._episode_sums.keys():
            extras[f"Episode_Reward/{key}"] = torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        goal_distance = self._goal_distance(env_ids)
        success = self.reset_terminated[env_ids] & (goal_distance < self.cfg.goal_reached_threshold)
        collision = self._collision[env_ids]
        base_too_low = self._robot.data.root_pos_w[env_ids, 2] < self.cfg.termination_height
        tipped_over = self._robot.data.projected_gravity_b[env_ids, 2] > self.cfg.termination_projected_gravity_z
        fall_or_tip = base_too_low | tipped_over

        num_resets = max(len(env_ids), 1)
        extras["Metrics/success_rate"] = torch.count_nonzero(success).float() / num_resets
        extras["Metrics/collision_rate"] = torch.count_nonzero(collision).float() / num_resets
        extras["Metrics/fall_tip_rate"] = torch.count_nonzero(fall_or_tip).float() / num_resets
        extras["Metrics/time_out_rate"] = torch.count_nonzero(self.reset_time_outs[env_ids]).float() / num_resets
        extras["Metrics/mean_min_clearance"] = torch.mean(self._min_obstacle_clearance[env_ids])

        self.extras["log"] = extras

    def _sample_goals(self, env_ids: torch.Tensor, root_pos_w: torch.Tensor) -> None:
        radius = sample_uniform(
            self.cfg.goal_distance_range[0],
            self.cfg.goal_distance_range[1],
            (len(env_ids),),
            self.device,
        )
        angle = sample_uniform(-3.14159, 3.14159, (len(env_ids),), self.device)
        self._goal_pos_w[env_ids, 0] = root_pos_w[:, 0] + radius * torch.cos(angle)
        self._goal_pos_w[env_ids, 1] = root_pos_w[:, 1] + radius * torch.sin(angle)
        self._goal_pos_w[env_ids, 2] = root_pos_w[:, 2]

    def _compute_goal_local_pos(self) -> torch.Tensor:
        goal_delta = self._goal_pos_w - self._robot.data.root_pos_w
        return quat_apply_inverse(yaw_quat(self._robot.data.root_quat_w), goal_delta)[:, :2]

    def _goal_distance(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            return torch.linalg.norm(self._goal_pos_w[:, :2] - self._robot.data.root_pos_w[:, :2], dim=1)
        return torch.linalg.norm(
            self._goal_pos_w[env_ids, :2] - self._robot.data.root_pos_w[env_ids, :2],
            dim=1,
        )

    def _sample_obstacles(self, env_ids: torch.Tensor, root_pos_w: torch.Tensor) -> None:
        if self.cfg.num_static_obstacles > 0:
            static_radius = self._sample_radius(
                len(env_ids), self.cfg.num_static_obstacles, self.cfg.static_obstacle_radius_range
            )
            self._static_obstacle_radius[env_ids] = static_radius
            self._static_obstacle_pos_w[env_ids] = self._sample_obstacle_positions(
                root_pos_w, self._goal_pos_w[env_ids, :2], static_radius
            )

        if self.cfg.num_dynamic_obstacles > 0:
            dynamic_radius = self._sample_radius(
                len(env_ids), self.cfg.num_dynamic_obstacles, self.cfg.dynamic_obstacle_radius_range
            )
            self._dynamic_obstacle_radius[env_ids] = dynamic_radius
            self._dynamic_obstacle_pos_w[env_ids] = self._sample_obstacle_positions(
                root_pos_w, self._goal_pos_w[env_ids, :2], dynamic_radius
            )
            speed = self._rand_range(
                (len(env_ids), self.cfg.num_dynamic_obstacles),
                self.cfg.dynamic_obstacle_speed_range,
            )
            direction = torch.empty(len(env_ids), self.cfg.num_dynamic_obstacles, device=self.device).uniform_(
                -3.14159, 3.14159
            )
            self._dynamic_obstacle_vel_w[env_ids, :, 0] = speed * torch.cos(direction)
            self._dynamic_obstacle_vel_w[env_ids, :, 1] = speed * torch.sin(direction)

    def _sample_radius(self, num_envs: int, num_obstacles: int, radius_range: tuple[float, float]) -> torch.Tensor:
        return self._rand_range((num_envs, num_obstacles), radius_range)

    def _sample_obstacle_positions(
        self, root_pos_w: torch.Tensor, goal_pos_w: torch.Tensor, radius: torch.Tensor
    ) -> torch.Tensor:
        num_envs, num_obstacles = radius.shape
        root_xy = root_pos_w[:, :2]
        positions = root_xy.unsqueeze(1).repeat(1, num_obstacles, 1)

        for _ in range(12):
            spawn_radius = self._rand_range((num_envs, num_obstacles), (1.0, self.cfg.obstacle_spawn_radius))
            angle = torch.empty(num_envs, num_obstacles, device=self.device).uniform_(-3.14159, 3.14159)
            candidates = root_xy.unsqueeze(1) + torch.stack(
                (spawn_radius * torch.cos(angle), spawn_radius * torch.sin(angle)),
                dim=-1,
            )

            root_clearance = torch.linalg.norm(candidates - root_xy.unsqueeze(1), dim=-1) - radius
            goal_clearance = torch.linalg.norm(candidates - goal_pos_w.unsqueeze(1), dim=-1) - radius
            valid = (
                (root_clearance > self.cfg.obstacle_robot_clearance)
                & (goal_clearance > self.cfg.obstacle_goal_clearance)
            )
            positions = torch.where(valid.unsqueeze(-1), candidates, positions)

        return positions

    def _rand_range(self, shape: tuple[int, ...], value_range: tuple[float, float]) -> torch.Tensor:
        return torch.empty(*shape, device=self.device).uniform_(value_range[0], value_range[1])

    def _update_dynamic_obstacles(self) -> None:
        if self.cfg.num_dynamic_obstacles == 0:
            return
        self._dynamic_obstacle_pos_w += self._dynamic_obstacle_vel_w * self.step_dt

        env_origins = self._terrain.env_origins[:, None, :2]
        rel_pos = self._dynamic_obstacle_pos_w - env_origins
        rel_norm = torch.linalg.norm(rel_pos, dim=-1)
        out_of_bounds = rel_norm > self.cfg.obstacle_spawn_radius
        if torch.any(out_of_bounds):
            clamped_rel = rel_pos / rel_norm.clamp(min=1.0e-6).unsqueeze(-1) * self.cfg.obstacle_spawn_radius
            self._dynamic_obstacle_pos_w = torch.where(
                out_of_bounds.unsqueeze(-1), env_origins + clamped_rel, self._dynamic_obstacle_pos_w
            )
            self._dynamic_obstacle_vel_w = torch.where(
                out_of_bounds.unsqueeze(-1), -self._dynamic_obstacle_vel_w, self._dynamic_obstacle_vel_w
            )

    def _all_obstacles(self) -> tuple[torch.Tensor, torch.Tensor]:
        positions = []
        radii = []
        if self.cfg.num_static_obstacles > 0:
            positions.append(self._static_obstacle_pos_w)
            radii.append(self._static_obstacle_radius)
        if self.cfg.num_dynamic_obstacles > 0:
            positions.append(self._dynamic_obstacle_pos_w)
            radii.append(self._dynamic_obstacle_radius)
        if not positions:
            return (
                torch.empty(self.num_envs, 0, 2, device=self.device),
                torch.empty(self.num_envs, 0, device=self.device),
            )
        return torch.cat(positions, dim=1), torch.cat(radii, dim=1)

    def _obstacle_collision_and_clearance(self) -> tuple[torch.Tensor, torch.Tensor]:
        obstacle_pos_w, obstacle_radius = self._all_obstacles()
        if obstacle_pos_w.shape[1] == 0:
            no_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            max_clearance = torch.full((self.num_envs,), self.cfg.max_ray_distance, device=self.device)
            return no_collision, max_clearance

        root_xy = self._robot.data.root_pos_w[:, None, :2]
        inflated_radius = obstacle_radius + self.cfg.robot_collision_radius
        signed_clearance = torch.linalg.norm(obstacle_pos_w - root_xy, dim=-1) - inflated_radius
        min_clearance = torch.min(signed_clearance, dim=1).values
        return min_clearance <= 0.0, min_clearance

    def _update_obstacle_rays(self) -> None:
        obstacle_pos_w, obstacle_radius = self._all_obstacles()
        if obstacle_pos_w.shape[1] == 0:
            self._rays.fill_(self.cfg.max_ray_distance)
            return

        delta_w = obstacle_pos_w - self._robot.data.root_pos_w[:, None, :2]
        delta_w_3d = torch.cat((delta_w, torch.zeros_like(delta_w[..., :1])), dim=-1)
        num_obstacles = obstacle_pos_w.shape[1]
        yaw_quat_w = yaw_quat(self._robot.data.root_quat_w).repeat_interleave(num_obstacles, dim=0)
        delta_b = quat_apply_inverse(yaw_quat_w, delta_w_3d.reshape(-1, 3)).reshape(
            self.num_envs, num_obstacles, 3
        )[..., :2]

        inflated_radius = obstacle_radius + self.cfg.robot_collision_radius
        projection = torch.einsum("eod,rd->eor", delta_b, self._ray_dirs_b)
        center_dist_sq = torch.sum(delta_b * delta_b, dim=-1, keepdim=True)
        perpendicular_dist_sq = center_dist_sq - projection * projection
        radius_sq = inflated_radius.unsqueeze(-1) * inflated_radius.unsqueeze(-1)
        discriminant = radius_sq - perpendicular_dist_sq
        valid_hit = (projection > 0.0) & (discriminant >= 0.0)
        hit_distance = projection - torch.sqrt(torch.clamp(discriminant, min=0.0))
        hit_distance = torch.where(valid_hit & (hit_distance >= 0.0), hit_distance, torch.inf)
        rays = torch.min(hit_distance, dim=1).values
        rays = torch.where(torch.isinf(rays), torch.full_like(rays, self.cfg.max_ray_distance), rays)
        self._rays = rays.clamp(min=self.cfg.min_ray_distance, max=self.cfg.max_ray_distance)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "obstacle_visualizer"):
                marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Go2Nav/obstacles",
                    markers={
                        "static": sim_utils.SphereCfg(
                            radius=1.0,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 1.0)),
                        ),
                        "dynamic": sim_utils.SphereCfg(
                            radius=1.0,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.1)),
                        ),
                    },
                )
                self.obstacle_visualizer = VisualizationMarkers(marker_cfg)
            self.obstacle_visualizer.set_visibility(True)
        elif hasattr(self, "obstacle_visualizer"):
            self.obstacle_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "obstacle_visualizer"):
            return
        positions, radii = self._all_obstacles()
        if positions.shape[1] == 0:
            return
        translations = torch.cat(
            (
                positions.reshape(-1, 2),
                torch.full((positions.numel() // 2, 1), 0.35, device=self.device),
            ),
            dim=-1,
        )
        scales = torch.cat(
            (
                radii.reshape(-1, 1).repeat(1, 2),
                torch.full((radii.numel(), 1), 0.35, device=self.device),
            ),
            dim=-1,
        )
        marker_indices = torch.cat(
            (
                torch.zeros(self.num_envs * self.cfg.num_static_obstacles, dtype=torch.long, device=self.device),
                torch.ones(self.num_envs * self.cfg.num_dynamic_obstacles, dtype=torch.long, device=self.device),
            )
        )
        self.obstacle_visualizer.visualize(
            translations=translations,
            scales=scales,
            marker_indices=marker_indices,
        )

    @staticmethod
    def _reindex(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]

    def _maybe_reindex_observations(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.cfg.debug_reindex_loco_observations:
            return self._reindex(tensor)
        return tensor

    def _maybe_reindex_actions(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.cfg.debug_reindex_loco_actions:
            return self._reindex(tensor)
        return tensor

    def _update_history(self, history: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        reset_mask = (self.episode_length_buf <= 1).view(self.num_envs, 1, 1)
        filled_history = current.unsqueeze(1).repeat(1, history.shape[1], 1)
        shifted_history = torch.cat((history[:, 1:], current.unsqueeze(1)), dim=1)
        return torch.where(reset_mask, filled_history, shifted_history)
