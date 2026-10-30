# -*- coding: utf-8 -*-
"""Constrained-opening laptop packing, v2.

Coupled drawer/object geometry is owned by geometry.py. All step actions use
robosuite actuators and MuJoCo contact dynamics; only reset places the object.
Success requires full containment, release, closure and 0.5 s stability.
"""
from collections import OrderedDict
import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import Arena
from robosuite.models.objects import BoxObject
from react_manip.visual_task import ManipulationTask
from react_manip.core.visual_randomization import (
    apply_visual_scenario, ensure_workspace_camera, resolve_scenario,
)
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler

_ARENA_XML = os.path.join(os.path.dirname(__file__), "assets", "laptop_drawer_arena_v4.xml")

# Legacy constants retained for imports by old diagnostic scripts only.
# V2 geometry and scoring MUST use self.geometry / self.laptop_size.
H = 0.35
CAB_FRONT_X = -0.05   # 柜体前脸 x（整体向 -x 移近 150mm，靠近机械臂）
CAB_BACK_X = 0.37     # 柜体后墙 x
CAB_HALF_W = 0.18     # 柜体 y 半宽（外侧）
CAB_INNER_HALF_W = 0.172  # 侧壁内表面 y 半宽（0.18 - 0.008）
DRAWER_H = 0.09      # 抽屉内腔高（加高盒体，约 8.6cm 可用高度）
DRAWER_FLOOR_Z = 0.658  # drawer floor upper surface
DESK_TOP_Z = 0.80    # 桌面顶面 z
CAB_TOP_Z = DESK_TOP_Z  # laptop rests directly on the desktop
CAB_EDGE_NX_X = -0.05  # 柜顶 -x 边缘（mac 从此悬出 5cm）
DESK_EDGE_X = -0.45  # 桌面 -x 边缘（遗留注释，无引用）
LAPTOP_SIZE = (0.31, 0.215, 0.0155)  # L×w×t
LAPTOP_MASS = 1.6    # kg
# Legacy nominal initial pose (not used by the v2 reset sampler).
LAPTOP_INIT_POS = np.array([0.145, 0.0, CAB_TOP_Z + LAPTOP_SIZE[2] / 2])

_DEBUG_MARKER_CAPACITY = 32


class LaptopDrawerArena(Arena):
    """柜体+抽屉+置物台场景。"""

    def __init__(self, scenario=None):
        super().__init__(_ARENA_XML)
        self.scenario = resolve_scenario(scenario)
        # Keep the canonical camera available even for legacy callers that do
        # not provide a Scenario/Profile (e.g. direct benchmark factories).
        ensure_workspace_camera(self)
        from .geometry import geometry_from_scenario, build_drawer
        self.geometry = geometry_from_scenario(self.scenario)
        build_drawer(self, self.geometry)
        if scenario is not None:
            apply_visual_scenario(self, self.scenario)
        self.drawer_joint = "drawer_slide"
        self.handle_site = "drawer_handle_site"
        # 移除 robosuite 自动添加的 mocap 辅助 body（left/right_eef_target 的
        # 半透明球/方块）：IK_POSE 控制器每步把 mocap body 移到 eef 目标，操作时
        # 渲染在柜子上方（视频里像"球"）。geom_rgba/vopt 隐藏均无效（渲染缓存），
        # 从元素树删除最彻底。mink 运动学驱动不用 mocap，控制器仅构造时引用。
        for b in self.worldbody.findall("body"):
            if b.get("name") in ("left_eef_target", "right_eef_target"):
                self.worldbody.remove(b)


class LaptopIntoDrawer(ManipulationEnv):
    """Grounded desk with an under-desk drawer and a laptop."""

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="Robotiq85Gripper",
        base_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=False,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=2400,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        deterministic_initial_laptop=False,
        seed=None,
        scenario=None,
        episode_instance=None,
    ):
        self.episode_instance = episode_instance
        self.use_object_obs = use_object_obs
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.placement_initializer = placement_initializer
        self.deterministic_initial_laptop = deterministic_initial_laptop
        self.scenario = resolve_scenario(scenario)
        self.mobile_base_command = np.zeros(3, dtype=float)
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    # Legacy robosuite object sensors are exact simulator state. They are not
    # part of the normal policy interface; GeometryFeatureAdapter is the
    # canonical privileged Ground Truth interface for this task.
    privileged_observation_keys = (
        "laptop_pos",
        "laptop_ori",
        "drawer_qpos",
        "handle_pos",
    )

    def set_mobile_base_command(self, vx: float, wz: float) -> None:
        """Set normalized body-forward/yaw speed for differential wheel drive.

        This is consumed during the next physical ``step``; it never edits
        qpos/qvel. The current robosuite controller exposes arm actions only.
        """
        if not np.all(np.isfinite([vx, wz])):
            raise ValueError("mobile command must be finite")
        self.mobile_base_command[:] = (float(np.clip(vx, -1, 1)), 0.0, float(np.clip(wz, -1, 1)))

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step=policy_step)
        from react_manip.robots.mobile_base import set_wheel_command
        vx, _, wz = self.mobile_base_command
        set_wheel_command(self, vx, wz)

    def _load_model(self):
        super()._load_model()
        # All robot bases are placed on the room floor. Platform geometry is
        # part of the robot asset (when applicable), never task scenery.
        if self.robots[0].name in ("MobileDualPiperX", "AgileXMobileAloha"):
            self.robots[0].robot_model.set_base_xpos([-0.95, 0.0, 0.0])
        else:
            self.robots[0].robot_model.set_base_xpos([-0.61, 0.0, 0.0])

        mujoco_arena = LaptopDrawerArena(self.scenario)
        mujoco_arena.set_origin([0, 0, 0])
        if self.robots[0].name in ("MobileDualPiperX", "AgileXMobileAloha"):
            import xml.etree.ElementTree as ET
            ET.SubElement(mujoco_arena.worldbody, "camera", name="mobileview",
                          pos="-2.4 -3.0 2.4", xyaxes="0.83844 -0.54499 0 0.23944 0.36838 0.89831")
        from .geometry import bevel_laptop
        self.geometry = mujoco_arena.geometry
        # Preallocate a generic pool because MuJoCo geometry cannot be added
        # after the model is compiled. Callers configure every visible
        # property and optional attachment through set_debug_markers().
        import xml.etree.ElementTree as ET
        for index in range(_DEBUG_MARKER_CAPACITY):
            prefix = f'debug_marker_{index}'
            ET.SubElement(mujoco_arena.asset, 'material',
                          name=prefix + '_material', rgba='1 1 1 0',
                          emission='0')
            body = ET.SubElement(mujoco_arena.worldbody, 'body', name=prefix,
                                 mocap='true', pos='0 0 -10')
            ET.SubElement(body, 'geom', name=prefix + '_geom', type='sphere',
                          size='0.02', material=prefix + '_material',
                          contype='0', conaffinity='0', group='1')
        self.laptop_size = np.array(self.geometry.laptop_size)
        self.drawer_floor_z = self.geometry.floor
        self.laptop = BoxObject(name="laptop", size=self.laptop_size / 2,
            rgba=(.15,.17,.22,1), density=self.geometry.mass / np.prod(self.laptop_size),
            friction=(.6,.01,.001))
        bevel_laptop(self.laptop, self.geometry)
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.laptop)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="LaptopSampler", mujoco_objects=self.laptop,
                x_range=(.24,.29), y_range=(-.13,.13), rotation=(-np.pi,np.pi),
                rotation_axis="z", ensure_object_boundary_in_range=False,
                ensure_valid_placement=True, reference_pos=np.zeros(3), rng=self.rng)

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.laptop,
        )
        from .contacts import install_laptop_contacts
        install_laptop_contacts(self.model.contact, self.laptop)


    def _setup_references(self):
        super()._setup_references()
        self.laptop_body_id = self.sim.model.body_name2id(self.laptop.root_body)
        self.drawer_qpos_addr = self.sim.model.get_joint_qpos_addr("drawer_slide")
        self.handle_site_id = self.sim.model.site_name2id("drawer_handle_site")
        self._gripper_geom_ids = [i for i in range(self.sim.model.ngeom)
                                  if (self.sim.model.geom_id2name(i) or '').startswith('gripper')]
        self._debug_marker_slots = []
        for index in range(_DEBUG_MARKER_CAPACITY):
            prefix = f'debug_marker_{index}'
            body_id = self.sim.model.body_name2id(prefix)
            geom_id = self.sim.model.geom_name2id(prefix + '_geom')
            self._debug_marker_slots.append({
                'mocap_id': int(self.sim.model.body_mocapid[body_id]),
                'geom_id': geom_id,
                'material_id': int(self.sim.model.geom_matid[geom_id]),
            })
        self._debug_markers = OrderedDict()

    def set_debug_markers(self, markers):
        """Replace the rendered debug markers with generic marker specs.

        ``markers`` maps an arbitrary name to a specification containing:
        ``position`` (required), ``color`` (RGB or RGBA), ``emission``,
        ``size``, and optional ``attach_to``. Without ``attach_to``, position
        is fixed in world coordinates. An attachment is ``{"type":
        "body"|"site"|"geom", "name": ...}``, and position is then a local
        offset that follows that simulator object.
        """
        if markers is None:
            markers = {}
        if not isinstance(markers, dict):
            raise TypeError('markers must be a mapping from names to specifications')
        if len(markers) > len(self._debug_marker_slots):
            raise ValueError(f'at most {len(self._debug_marker_slots)} debug markers are supported')

        configured = OrderedDict()
        for slot, (name, spec) in zip(self._debug_marker_slots, markers.items()):
            if not isinstance(spec, dict) or 'position' not in spec:
                raise ValueError(f'debug marker {name!r} requires a position')
            position = np.asarray(spec['position'], dtype=float)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError(f'debug marker {name!r} position must be three finite values')
            color = np.asarray(spec.get('color', (1, 1, 1, 1)), dtype=float)
            if color.shape == (3,):
                color = np.r_[color, 1.]
            if color.shape != (4,) or not np.all(np.isfinite(color)):
                raise ValueError(f'debug marker {name!r} color must be RGB or RGBA')
            emission = float(spec.get('emission', 0.))
            size = float(spec.get('size', .02))
            if not np.isfinite(emission) or emission < 0:
                raise ValueError(f'debug marker {name!r} emission must be non-negative')
            if not np.isfinite(size) or size <= 0:
                raise ValueError(f'debug marker {name!r} size must be positive')

            attachment = self._resolve_debug_attachment(name, spec.get('attach_to'))
            self.sim.model.geom_rgba[slot['geom_id']] = color
            self.sim.model.mat_rgba[slot['material_id']] = color
            self.sim.model.mat_emission[slot['material_id']] = emission
            self.sim.model.geom_size[slot['geom_id'], 0] = size
            configured[str(name)] = {
                **slot, 'position': position.copy(), 'attachment': attachment,
            }

        for slot in self._debug_marker_slots[len(configured):]:
            self.sim.model.geom_rgba[slot['geom_id'], 3] = 0.
            self.sim.model.mat_rgba[slot['material_id'], 3] = 0.
            self.sim.data.mocap_pos[slot['mocap_id']] = (0., 0., -10.)
        self._debug_markers = configured
        self._update_debug_markers()
        self.sim.forward()

    def _resolve_debug_attachment(self, marker_name, attachment):
        if attachment is None:
            return None
        if not isinstance(attachment, dict):
            raise ValueError(f'debug marker {marker_name!r} attach_to must be a mapping')
        kind = attachment.get('type')
        target_name = attachment.get('name')
        lookups = {
            'body': self.sim.model.body_name2id,
            'site': self.sim.model.site_name2id,
            'geom': self.sim.model.geom_name2id,
        }
        if kind not in lookups or not isinstance(target_name, str):
            raise ValueError(
                f'debug marker {marker_name!r} attachment needs body/site/geom type and name')
        try:
            target_id = int(lookups[kind](target_name))
        except Exception as exc:
            raise ValueError(
                f'debug marker {marker_name!r} attachment {kind} {target_name!r} was not found') from exc
        return kind, target_id

    def _update_debug_markers(self):
        positions = {
            'body': (self.sim.data.body_xpos, self.sim.data.body_xmat),
            'site': (self.sim.data.site_xpos, self.sim.data.site_xmat),
            'geom': (self.sim.data.geom_xpos, self.sim.data.geom_xmat),
        }
        for marker in getattr(self, '_debug_markers', {}).values():
            point = marker['position']
            if marker['attachment'] is not None:
                kind, target_id = marker['attachment']
                origins, rotations = positions[kind]
                point = origins[target_id] + rotations[target_id].reshape(3, 3).dot(point)
            self.sim.data.mocap_pos[marker['mocap_id']] = point

    def _post_action(self, action):
        result = super()._post_action(action)
        self._update_debug_markers()
        self.sim.forward()
        return result

    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def laptop_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.laptop_body_id])

            @sensor(modality=modality)
            def laptop_ori(obs_cache):
                return np.array(self.sim.data.body_xquat[self.laptop_body_id])

            @sensor(modality=modality)
            def drawer_qpos(obs_cache):
                return np.array([self.sim.data.qpos[self.drawer_qpos_addr]])

            @sensor(modality=modality)
            def handle_pos(obs_cache):
                return np.array(self.sim.data.site_xpos[self.handle_site_id])

            for name, s in [
                ("laptop_pos", laptop_pos),
                ("laptop_ori", laptop_ori),
                ("drawer_qpos", drawer_qpos),
                ("handle_pos", handle_pos),
            ]:
                observables[name] = Observable(name=name, sensor=s, sampling_rate=self.control_freq)
        return observables

    def _reset_internal(self):
        self.mobile_base_command.fill(0)
        super()._reset_internal()
        # Keep the robot posture stable across repeated resets (keyboard
        # Backspace must reproduce the startup EEF pose). Object placement is
        # still randomized independently below.
        robot_qpos = np.array([self.sim.data.qpos[a] for a in self.robots[0]._ref_joint_pos_indexes])
        if getattr(self, "_fixed_robot_qpos", None) is None:
            self._fixed_robot_qpos = robot_qpos.copy()
        else:
            self.sim.data.qpos[self.robots[0]._ref_joint_pos_indexes] = self._fixed_robot_qpos
            self.sim.forward()
            # Robot.reset() initialized IK targets before the fixed posture
            # was restored. Rebind every controller target to the restored
            # EEF pose, otherwise the first step pulls the arm back toward the
            # previous reset posture.
        # Always synchronize controller state after reset. This includes the
        # first reset, where the robot posture is captured above, and avoids
        # stale IK targets pulling the arm toward a previous episode.
        if getattr(self, "_fixed_robot_qpos", None) is not None:
            composite = self.robots[0].composite_controller
            update_state = getattr(composite, "update_state", None)
            if callable(update_state):
                update_state()
            reset_controller = getattr(composite, "reset", None)
            if callable(reset_controller):
                reset_controller()
            for controller in getattr(composite, "part_controllers", {}).values():
                reset_goal = getattr(controller, "reset_goal", None)
                if callable(reset_goal):
                    reset_goal()
                # IK_POSE stores its own Cartesian target. Force it to the
                # post-reset EEF pose after all state caches are refreshed.
                if hasattr(controller, "reference_target_pos"):
                    site = self.robots[0].eef_site_id.get("right")
                    if site is not None:
                        controller.reference_target_pos = np.array(self.sim.data.site_xpos[site], copy=True)
                        from robosuite.utils.transform_utils import mat2quat
                        controller.reference_target_orn = mat2quat(
                            np.array(self.sim.data.site_xmat[site], copy=True).reshape(3, 3)
                        )
                        interp = getattr(controller, "interpolator_pos", None)
                        if interp is not None and hasattr(interp, "set_goal"):
                            interp.set_goal(np.zeros(3))
        self.sim.data.qpos[self.drawer_qpos_addr] = 0.
        self._success_since = None
        self._stable_success = False
        self._success_awarded = False
        if not self.deterministic_reset:
            if self.episode_instance is not None:
                from .instances import validate_episode_instance
                validate_episode_instance(self.scenario.parameters, self.episode_instance)
                x, y, yaw = self.episode_instance['xy_yaw']
                pos, quat = np.array([x,y,0.]), np.array([np.cos(yaw/2),0.,0.,np.sin(yaw/2)])
            elif self.deterministic_initial_laptop:
                pos, quat = np.array([.265,0.,0.]), np.array([1.,0.,0.,0.])
            else:
                pos, quat, _ = self.placement_initializer.sample()[self.laptop.name]
                pos, quat = np.asarray(pos, float), np.asarray(quat,float)
            pos[2] = DESK_TOP_Z + self.laptop_size[2]/2 + self.geometry.foot_height + .001
            self._set_laptop_pose(pos, quat)

    def _set_laptop_pose(self, pos, quat):
        """笔记本是 free joint：位姿写入 qpos（body_pos 对 free body 无效）。"""
        jnt_addr = self.sim.model.get_joint_qpos_addr(self.laptop.joints[0])[0]
        self.sim.data.qpos[jnt_addr:jnt_addr + 3] = pos
        self.sim.data.qpos[jnt_addr + 3:jnt_addr + 7] = quat
        self.sim.forward()

    def _drawer_inner_bounds(self, margin=0.003):
        """抽屉可放置区域 xy 边界（世界系，含安全边距）。"""
        q = self.sim.data.qpos[self.drawer_qpos_addr]
        g = self.geometry
        return (g.front+.008+q+margin, g.front+.008+q+g.depth-margin,
                -g.width/2+margin, g.width/2-margin)

    def _laptop_corners(self):
        pos = np.array(self.sim.data.body_xpos[self.laptop_body_id])
        rotation = np.array(self.sim.data.body_xmat[self.laptop_body_id]).reshape(3,3)
        half = self.laptop_size/2
        local = np.array([[x*half[0],y*half[1],z] for x in (-1,1) for y in (-1,1)
                          for z in (-half[2]-self.geometry.foot_height, half[2])])
        return pos + local @ rotation.T

    def _laptop_xy_corners(self):
        return self._laptop_corners()[:,:2]

    def _laptop_in_drawer(self):
        # Physical containment ends at the inner faces. Clearance above zero
        # is useful robustness margin, but must not make a fully inside laptop
        # reopen the drawer for another alignment cycle.
        x0,x1,y0,y1 = self._drawer_inner_bounds(margin=0.)
        c = self._laptop_corners()
        return bool(np.all((c[:,0]>=x0)&(c[:,0]<=x1)&(c[:,1]>=y0)&(c[:,1]<=y1)
                           &(c[:,2]>=self.drawer_floor_z-.002)&(c[:,2]<=self.drawer_floor_z+self.geometry.height)))

    def _laptop_flat(self):
        """笔记本平放：底面法向与竖直夹角 < 2°。"""
        quat = self.sim.data.body_xquat[self.laptop_body_id]
        z_axis = np.array([2 * (quat[1] * quat[3] + quat[0] * quat[2]),
                           2 * (quat[2] * quat[3] - quat[0] * quat[1]),
                           quat[0] ** 2 - quat[1] ** 2 - quat[2] ** 2 + quat[3] ** 2])
        tilt = np.arccos(np.clip(z_axis[2], -1, 1))
        return tilt < np.deg2rad(2.0)

    def _laptop_bottom_contact(self):
        """笔记本底面触底（z < 抽屉底板+半厚+容差）。"""
        pos = self.sim.data.body_xpos[self.laptop_body_id]
        return abs(pos[2] - self.laptop_size[2] / 2 - self.geometry.foot_height - self.drawer_floor_z) < 0.008

    def _check_success(self):
        return bool(getattr(self, '_stable_success', False))

    def _success_candidate(self):
        if abs(float(self.sim.data.qpos[self.drawer_qpos_addr])) > .003:
            return False
        if not self._laptop_in_drawer() or not self._laptop_bottom_contact():
            return False
        velocity = self.sim.data.cvel[self.laptop_body_id]
        if np.linalg.norm(velocity[:3]) > .1 or np.linalg.norm(velocity[3:]) > .015:
            return False
        x0,x1,y0,y1 = self._drawer_inner_bounds(margin=0)
        if self._gripper_geom_ids:
            points = self.sim.data.geom_xpos[self._gripper_geom_ids]
            if np.any((points[:,0]>=x0-.01)&(points[:,0]<=x1+.01)&
                      (points[:,1]>=y0-.01)&(points[:,1]<=y1+.01)&
                      (points[:,2]>=self.drawer_floor_z-.01)&
                      (points[:,2]<=self.drawer_floor_z+self.geometry.height)):
                return False
        for c in self.sim.data.contact:
            names = [self.sim.model.geom_id2name(int(g)) or '' for g in (c.geom1,c.geom2)]
            if any(n.startswith('gripper') for n in names) and any(n.startswith(('laptop','drawer','cabinet')) for n in names):
                return False
        return True

    def step(self, action):
        obs, _, done, info = super().step(action)
        now = float(self.sim.data.time)
        if self._success_candidate():
            if self._success_since is None: self._success_since = now
        else:
            self._success_since = None
        self._stable_success = self._success_since is not None and now-self._success_since >= .5-1e-8
        event = self._stable_success and not self._success_awarded
        self._success_awarded |= self._stable_success
        done = bool(done or self._stable_success)
        self.done = done
        info.update(success_state=self._stable_success, terminal_success=self._stable_success,
                    total_score=100 * int(self._success_awarded), score_delta=100 * int(event),
                    events=['task_success'] if event else [], episode_finished=done)
        info['TimeLimit.truncated'] = bool(done and not self._stable_success)
        return obs, float(event) * (self.reward_scale if self.reward_scale is not None else 1.), done, info

    def _task_instruction(self):
        return (
            "Open the drawer, grasp the laptop, and place it completely inside. "
            "Tilt and reposition the laptop if necessary to pass through the opening. "
            "Release the laptop, withdraw the gripper, and close the drawer. "
            "The laptop must remain stable and the drawer must stay closed "
            "for at least 0.5 seconds. The drawer must be within 0.003 m of closed, "
            "the laptop fully contained with its bottom within 0.008 m of the drawer floor, "
            "linear speed <=0.015 m/s and angular speed <=0.1 rad/s. "
            "Keep the gripper outside the drawer interior and clear of the laptop, drawer and cabinet. "
            f"Success earns 100 and ends the episode; timeout at {self.horizon / self.control_freq:g} seconds earns zero. "
            "You may adjust and retry within the episode."
        )

    def get_episode_language_prompt(self):
        from react_manip.core.language import task_language_prompt
        return task_language_prompt(
            self, task_id='laptop_into_drawer',
            task_instruction=self._task_instruction(),
            success_condition='The laptop is fully contained and stable, the gripper is released and clear, and the drawer stays closed for 0.5 seconds.',
            constraints=(),
        )

    def reward(self, action=None):
        reward = 0.0
        if self._check_success():
            reward = 1.0
        elif self.reward_shaping:
            # 接近奖励：笔记本到抽屉开口中心的距离
            q = self.sim.data.qpos[self.drawer_qpos_addr]
            target = np.array([self.geometry.front+.008+q+self.geometry.depth/2, 0.0,
                               self.drawer_floor_z+self.geometry.height/2])
            dist = np.linalg.norm(self.sim.data.body_xpos[self.laptop_body_id] - target)
            reward += 0.5 * (1 - np.tanh(2.0 * dist))
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
