# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.direct.go2_nav.go2_nav_env_cfg import SEA_NAV_GO2_CFG


@configclass
class Go2NavSceneCfg(InteractiveSceneCfg):
    """Scene for the Unitree Go2 navigation task."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = SEA_NAV_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2000.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the Go2 joint-position controller."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Policy observations for Go2 locomotion/navigation."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.2, n_max=0.2))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """Minimal rewards that keep the robot upright and penalize unnecessary motion."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-5.0)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0005)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    """Termination terms for the Go2 task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_too_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.18})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.4})


@configclass
class Go2NavEnvCfg(ManagerBasedRLEnvCfg):
    scene: Go2NavSceneCfg = Go2NavSceneCfg(num_envs=1024, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 60.0
        self.viewer.eye = (6.0, 6.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.4)
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
