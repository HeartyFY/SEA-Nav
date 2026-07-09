# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


def _sea_nav_root() -> Path:
    env_root = os.environ.get("SEA_NAV_ROOT")
    if env_root and (Path(env_root) / "training/legged_gym").is_dir():
        return Path(env_root)
    for parent in Path(__file__).resolve().parents:
        if (parent / "training/legged_gym").is_dir():
            return parent
    fallback = Path.home() / "SEA-Nav"
    if (fallback / "training/legged_gym").is_dir():
        return fallback
    return Path(__file__).resolve().parents[6]


SEA_NAV_GO2_URDF = _sea_nav_root() / "training/legged_gym/resources/go2_description/urdf/go2_description_v8.urdf"


SEA_NAV_GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(SEA_NAV_GO2_URDF),
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="none",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness=30.0,
            damping=0.75,
            friction=0.0,
        ),
    },
)


@configclass
class Go2NavEnvCfg(DirectRLEnvCfg):
    episode_length_s = 60.0
    decimation = 4
    action_space = 3
    observation_space = 550
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=True)

    robot_cfg: ArticulationCfg = SEA_NAV_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    low_level_policy_dir = str(_sea_nav_root() / "training/legged_gym/legged_gym/ctrl_model")
    history_length = 10
    num_rays = 41
    min_ray_distance = 0.1
    max_ray_distance = 5.0
    ray_fov_deg = 180.0
    action_scale = 0.10
    command_filter_alpha = 0.5
    command_limits = ((-0.5, 2.0), (-1.0, 1.0), (-1.0, 1.0))
    goal_distance_range = (2.0, 5.0)
    goal_reached_threshold = 0.5
    termination_height = 0.18
    termination_projected_gravity_z = -0.2

    num_static_obstacles = 6
    num_dynamic_obstacles = 3
    obstacle_spawn_radius = 5.5
    obstacle_robot_clearance = 0.9
    obstacle_goal_clearance = 0.9
    static_obstacle_radius_range = (0.25, 0.65)
    dynamic_obstacle_radius_range = (0.20, 0.45)
    dynamic_obstacle_speed_range = (0.15, 0.75)
    robot_collision_radius = 0.35
    obstacle_clearance = 0.35
    obstacle_clearance_reward_scale = -0.35
    collision_penalty = -10.0
    obstacle_debug_vis = False
    debug_hold_default_pose = False
    debug_reindex_loco_observations = True
    debug_reindex_loco_actions = True

    lin_vel_obs_scale = 2.0
    ang_vel_obs_scale = 0.25
    dof_pos_obs_scale = 1.0
    dof_vel_obs_scale = 0.05

    progress_reward_scale = 4.0
    goal_reward_scale = 10.0
    heading_reward_scale = 1.0
    alive_reward_scale = 0.05
    action_rate_reward_scale = -0.02
    joint_velocity_reward_scale = -0.0005
    termination_penalty = -5.0
