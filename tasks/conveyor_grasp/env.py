"""Physical finite-queue category sorting with automatic belt re-entry."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import Arena
from react_manip.core.visual_randomization import apply_visual_scenario, resolve_scenario
from react_manip.core.scenarios import Scenario, sample_scenario
from react_manip.visual_task import ManipulationTask
from react_manip.category_objects import _add_object, _variant_spec, _vec, OBJECT_CATEGORY_NAMES
BELT_Z = 0.9
X_VIS_START, X_VIS_END = (0.95, 4.25)
X_REEMERGE, X_RETURN_END = (0.65, 4.65)
FEED_LANES_Y = (0.0,)
TABLE_Z = 0.75
BIN_HALF = np.array([0.24, 0.14, 0.12])
DUAL_PIPER_BASE_Y = 0.70
DUAL_PIPER_READY_QPOS = np.tile([0.0, 0.2, -1.1, 1.2, 0.0, 0.0], 2)

def _solid(parent, name, pos, size, material='metal_mat'):
    attrs = dict(type='box', pos=_vec(pos), size=_vec(size), material=material, friction='.8 .01 .001', solref='.005 1', condim='6')
    ET.SubElement(parent, 'geom', name=name, group='0', **attrs)
    ET.SubElement(parent, 'geom', name=name + '_visual', group='1', contype='0', conaffinity='0', density='0', **attrs)

class ConveyorArena(Arena):

    def __init__(self, env):
        super().__init__(str(Path(__file__).parent / 'assets/conveyor_arena.xml'))
        for node in list(self.worldbody):
            if node.get('name', '').startswith(('bin', 'cover_', 'chute_')):
                self.worldbody.remove(node)
        belt = self.worldbody.find(".//geom[@name='belt_top_col']")
        belt.attrib.update(friction=f'{env._mu_belt} .01 .001', solref='.005 1', condim='6', priority='1')
        belt.set('size', '2.3 .20 .02')
        belt.set('pos', '2.6 0 0')
        self.worldbody.find(".//geom[@name='belt_top_vis']").attrib.update(size='2.3 .20 .02', pos='2.6 0 .001')
        self.worldbody.find(".//joint[@name='belt_slide']").set('range', '-.2 .2')
        for name, x, half in (('inlet', 0.525, 0.425), ('outlet', 4.675, 0.425)):
            _solid(self.worldbody, name + '_roof', (x, 0, 1.25), (half, 0.34, 0.025))
            for side in (-1, 1):
                _solid(self.worldbody, name + f'_side{side}', (x, side * 0.32, 1.05), (half, 0.02, 0.2))
                _solid(self.worldbody, name + f'_leg{side}', (x, side * 0.25, 0.4), (0.04, 0.04, 0.4))
            _solid(self.worldbody, name + '_deck', (x, 0, 0.82), (half, 0.3, 0.02))
        ET.SubElement(self.asset, 'material', name='table_mat', rgba='.50 .32 .17 1')
        if env.aloha_layout:
            # Separate side tables leave the fixed dual-arm stand and shoulders clear.
            table_half_x = .37 if env.dual_piper_layout else .30
            for side, x in enumerate((1.25, 2.75)):
                _solid(self.worldbody, f'table_side_{side}', (x, .65, .725), (table_half_x, .37, .025), 'table_mat')
                for dx in (-.25, .25):
                    for y in (.32, .98):
                        _solid(self.worldbody, f'table_side_{side}_leg_{dx}_{y}',
                               (x + dx, y, .35), (.03, .03, .35))
        elif env.panda_layout:
            # A single notched table clears Panda's rear controller box.
            for name, x in (('table', 1.45), ('table_right', 2.55)):
                _solid(self.worldbody, name, (x, 1.05, .725), (.29, .36, .025), 'table_mat')
            _solid(self.worldbody, 'table_bridge', (2, 1.29, .725), (.26, .12, .025), 'table_mat')
        else:
            _solid(self.worldbody, 'table', (2, 1.28 + env.collection_shift, 0.725), (0.65, 0.36, 0.025), 'table_mat')
        for x in (() if env.aloha_layout else (1.2, 2.8) if env.panda_layout else (1.4, 2.6)):
            for y in (0.73, 1.37):
                _solid(self.worldbody, f'table_leg_{x}_{y}', (x, y + env.collection_shift, 0.35), (0.03, 0.03, 0.35))
        for i, center in enumerate(env.bin_centers):
            material = f'basket_mat_{i}'
            colors = ('.50 .65 .73 1', '.76 .64 .45 1', '.65 .72 .56 1', '.72 .64 .72 1')
            ET.SubElement(self.asset, 'material', name=material, rgba=colors[i], specular='.15')
            body = ET.SubElement(self.worldbody, 'body', name=f'basket_{i}', pos=_vec(center))
            _solid(body, f'basket_{i}_floor', (0, 0, 0), (0.24, 0.14, 0.01), material)
            for side in (-1, 1):
                _solid(body, f'basket_{i}_x{side}', (side * 0.25, 0, 0.12), (0.01, 0.16, 0.12), material)
                _solid(body, f'basket_{i}_y{side}', (0, side * 0.15, 0.12), (0.24, 0.01, 0.12), material)
        equality = ET.SubElement(self.root, 'equality')
        for i, r in enumerate(env.records):
            r['extent'] = _add_object(self.asset, self.worldbody, name=f'target_{i}', category=r['category'], variant=r['variant'], pos=(0, 0, -5 - i))
            scale = env.target_size_scales[i]
            r['extent'] = r['extent'] * scale
            for g in self.worldbody.find(f"body[@name='target_{i}']").findall('geom'):
                for key in ('size', 'pos', 'fromto'):
                    if key in g.attrib:
                        g.set(key, _vec(np.fromstring(g.get(key), sep=' ') * scale))
                if g.get('mesh'):
                    mesh = self.asset.find(f"mesh[@name='{g.get('mesh')}']")
                    mesh.set('scale', _vec(np.fromstring(mesh.get('scale', '1 1 1'), sep=' ') * scale))
                if g.get('density') != '0':
                    g.set('density', str(320 * env.mass_scale))
            ET.SubElement(equality, 'weld', name=f'pending_{i}', body1=f'target_{i}', active='true')
        for i, r in enumerate(env.examples):
            extent = _variant_spec(r['category'], r['variant'])['extent']
            pos = env.bin_centers[r['basket']] + [-0.11 + 0.22 * r['slot'], -0.07, 0.014 + extent[2]]
            r['extent'] = _add_object(self.asset, self.worldbody, name=f'example_{i}', category=r['category'], variant=r['variant'], pos=pos)
        camera_shift = env.collection_shift if env.mobile_layout else 0.0
        for name, pos, target, fovy in (('agentview', (2, 3.8 + camera_shift, 2.8), (2, 0.8 + camera_shift / 2, 0.8), 68), ('frontview', (4.8, 3.5 + camera_shift, 2.8), (2, 0.6 + camera_shift / 2, 0.8), 62), ('sideview', (6.2, 0.8, 2.8), (2, 0.6 + camera_shift / 2, 0.8), 60), ('basket_view', (2, 3.1 + camera_shift, 2), (2, 1.5 + camera_shift, 0.8), 65), ('workspace_topview', (2, 0.7 + camera_shift / 2, 2.4), (2, 0.7 + camera_shift / 2, 0), 100)):
            old = self.worldbody.find(f"camera[@name='{name}']")
            if old is not None:
                self.worldbody.remove(old)
            z = np.asarray(pos) - target
            z = z / np.linalg.norm(z)
            x = np.cross([0, 0, 1], z)
            if np.linalg.norm(x) < 1e-06:
                x = np.array([1.0, 0, 0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)
            ET.SubElement(self.worldbody, 'camera', name=name, pos=_vec(pos), xyaxes=_vec(np.r_[x, y]), fovy=str(fovy))
        apply_visual_scenario(self, env.scenario or Scenario('1.0', 'conveyor_grasp', 'default', env.seed, {}))

class ConveyorGrasp(ManipulationEnv):
    task_id = 'conveyor_grasp'
    step_handles_robot_action = True
    privileged_observation_keys = ('object-state',)

    def __init__(self, robots='Sawyer', *, seed=0, scenario=None, robot_id=None, n_objects=None, category_count=None, controller_configs=None, gripper_types='default', initialization_noise=None, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, use_object_obs=False, camera_names=('agentview',), control_freq=20, horizon=6000, **kwargs):
        self.scenario = resolve_scenario(scenario)
        if self.scenario is None:
            profile = Path(__file__).resolve().parents[2] / 'configs/conveyor_grasp/profile_default.json'
            self.scenario = sample_scenario(json.loads(profile.read_text()), seed=0 if seed is None else int(seed))
            # Keep explicit constructor size overrides in the resolved record.
            for key, value in (('n_objects', n_objects), ('category_count', category_count)):
                if value is not None:
                    self.scenario.parameters.setdefault('task', {})[key] = int(value)
        if self.scenario.task_id != self.task_id:
            raise ValueError('Scenario task_id must be conveyor_grasp')
        self.robot_id = robot_id
        self.seed = int(self.scenario.seed) if self.scenario is not None else 0 if seed is None else int(seed)
        rng = np.random.default_rng(self.seed)
        p = self.scenario.parameters if self.scenario is not None else {}
        self.n_objects = int(p.get('task', {}).get('n_objects', n_objects if n_objects is not None else rng.integers(10, 31)))
        self.category_count = int(p.get('task', {}).get('category_count', category_count if category_count is not None else rng.integers(2, 5)))
        if not 10 <= self.n_objects <= 30 or not 2 <= self.category_count <= 4:
            raise ValueError('requires 10–30 objects and 2–4 categories')
        self._v_belt = float(p.get('belt', {}).get('speed', .15))
        self._mu_belt = float(p.get('belt', {}).get('friction', .6))
        self.mass_scale = float(p.get('object', {}).get('mass_scale', 1))
        variation = self.scenario.metadata.get('episode_variation', {}) if self.scenario else {}
        self.target_size_scales = variation.get('target_size_scales', [1.] * self.n_objects)
        if (len(self.target_size_scales) != self.n_objects or
                not all(np.isfinite(s) and .8 <= s <= 1.15 for s in self.target_size_scales)):
            raise ValueError('invalid per-target Episode sizes')
        if not 0.02 <= self._v_belt <= 0.4 or not 0.1 <= self._mu_belt <= 1.5 or not 0.2 <= self.mass_scale <= 3:
            raise ValueError('unsupported speed/friction/mass scale')
        from .instances import sample_layout, validate_layout
        if 'inventory' in p:
            layout = validate_layout(p['inventory'], self.n_objects, self.category_count)
        else:
            # Preserve bare/legacy Scenario sampling defaults.
            rng.choice(8, self.category_count, replace=False)
            speed, friction = rng.uniform(.08, .2), rng.uniform(.4, .8)
            self._v_belt = float(p.get('belt', {}).get('speed', speed))
            self._mu_belt = float(p.get('belt', {}).get('friction', friction))
            counts = p.get('basket', {}).get('example_counts')
            if counts is not None and (len(counts) != self.category_count or any(type(c) is not int or not 0 <= c <= 2 for c in counts)):
                raise ValueError('basket.example_counts must contain 0–2 per basket')
            layout = sample_layout(self.seed, self.n_objects, self.category_count, counts)
        self.categories = np.asarray(layout['categories'], dtype=int)
        self.records = layout['targets']
        self.examples = layout['examples']
        self.fixed_categories = np.asarray(layout['fixed_categories'], dtype=int)
        self._yaws = np.asarray(layout['target_yaws'], dtype=float)
        self.bin_centers = np.array([(1.7, 0.85, 0.76), (2.3, 0.85, 0.76), (1.7, 1.17, 0.76), (2.3, 1.17, 0.76)][:self.category_count])
        self.panda_layout = 'panda' in str(robots).lower()
        self.mobile_layout = 'mobile' in str(robots).lower()
        robot_token = str(robots).lower().replace('_', '')
        self.dual_piper_layout = 'dualpiperx' in robot_token and not self.mobile_layout
        self.aloha_layout = any(x in robot_token for x in ('dualpiperx', 'aloha')) and not self.mobile_layout
        if self.aloha_layout:
            centers = ([(1.25, .48, .76), (2.75, .48, .76),
                        (1.37, .82, .76), (2.63, .82, .76)]
                       if self.dual_piper_layout else
                       [(1.25, .50, .76), (2.75, .50, .76),
                        (1.25, .90, .76), (2.75, .90, .76)])
            self.bin_centers = np.array(centers[:self.category_count])
        # Reserve space for PiPER's folded upper arm. Panda instead needs
        # a central notch for the controller box supplied by its own mount.
        # Mobile manipulation needs floor-level manoeuvring space, not just
        # collision-free arms: table front = 1.90, belt support edge = 0.30.
        self.collection_shift = 0.98 if self.mobile_layout else 0.08 if str(robots).lower() in ('piperx', 'agilexpiper', 'legacy_piperx') else 0.0
        self.bin_centers[:, 1] += self.collection_shift
        if self.panda_layout:
            self.bin_centers[:, 0] += np.where(self.bin_centers[:, 0] < 2, -.25, .25)
        if str(robots).lower() == 'sawyer':
            from react_manip.robots.resolution import resolve_robosuite_robot
            robots = resolve_robosuite_robot('sawyer').robot_name
        if controller_configs is None:
            from react_manip.robots.resolution import default_controller_config
            controller_configs = default_controller_config(robots)
        super().__init__(robots=robots, controller_configs=controller_configs, gripper_types=gripper_types, initialization_noise=initialization_noise, has_renderer=has_renderer, has_offscreen_renderer=has_offscreen_renderer, use_camera_obs=use_camera_obs, camera_names=camera_names, control_freq=control_freq, horizon=horizon, lite_physics=False, seed=self.seed, **kwargs)

    def _load_model(self):
        """Compile the workcell and exclude hidden inventory from camera bounds."""
        super()._load_model()
        robot = self.robots[0].robot_model
        name = type(robot).__name__.lower()
        # Sawyer's integrated pedestal has a 0.31 m radius: reserve real
        # clearance from both belt rails and the collection table.
        y = (0.60 if 'sawyer' in name else 0.8 if 'mobile' in name else
             DUAL_PIPER_BASE_Y if self.dual_piper_layout else
             0.92 if 'aloha' in name.replace('_', '') else
             0.65 if 'panda' in name else 0.82)
        robot.set_base_xpos([2, y, 0])
        robot.set_base_ori([0, 0, -np.pi / 2])
        arena = ConveyorArena(self)
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots], mujoco_objects=[])
        for weld in arena.root.findall('equality/weld'):
            self.model.equality.append(ET.fromstring(ET.tostring(weld)))
        option = self.model.root.find('option')
        option.attrib.update(integrator='implicitfast', timestep='.002', impratio='10')
        flag = option.find('flag')
        if flag is None:
            flag = ET.SubElement(option, 'flag')
        flag.attrib.update(multiccd='enable', nativeccd='disable')
        statistic = self.model.root.find('statistic')
        if statistic is None:
            statistic = ET.SubElement(self.model.root, 'statistic')
        statistic.attrib.update(center='2 .7 .8', extent='4')

    def _setup_references(self):
        super()._setup_references()
        m = self.sim.model
        self.obj_body_ids = np.array([m.body_name2id(f'target_{i}') for i in range(self.n_objects)])
        self.example_body_ids = np.array([m.body_name2id(f'example_{i}') for i in range(len(self.examples))], dtype=int)
        self.obj_jnt_addrs = np.array([m.jnt_qposadr[m.joint_name2id(f'target_{i}_free')] for i in range(self.n_objects)])
        self.obj_qvel_addrs = np.array([m.jnt_dofadr[m.joint_name2id(f'target_{i}_free')] for i in range(self.n_objects)])
        self.pending_ids = [mujoco.mj_name2id(m._model, mujoco.mjtObj.mjOBJ_EQUALITY, f'pending_{i}') for i in range(self.n_objects)]
        self.obj_geom_ids = [np.flatnonzero(m.geom_bodyid == b) for b in self.obj_body_ids]
        self._collision = (m.geom_contype.copy(), m.geom_conaffinity.copy(), m.geom_rgba.copy(), m.geom_group.copy())
        j = m.joint_name2id('belt_slide')
        self.belt_qpos_addr, self.belt_qvel_addr = (int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]))
        self.belt_actuator_id = m.actuator_name2id('belt_vel')

    def _reset_internal(self):
        super()._reset_internal()
        self.queue = list(range(self.n_objects))
        self._return_states = {}
        self.phases = np.full(self.n_objects, 'queued', dtype='U12')
        self.bin_categories = self.fixed_categories.copy()
        self._stable_ticks = np.zeros(self.n_objects, int)
        self._last_bins = np.full(self.n_objects, -1, int)
        self._counted = np.zeros(self.n_objects, bool)
        self._sorted_count = 0
        self._success_ticks = 0
        self._finished = False
        self._recycles = 0
        self._events = []
        self._bin_ids = np.full(self.n_objects, -1)
        self._mixed = np.zeros(self.category_count, bool)
        self._examples_retained = True
        for ids in self.obj_geom_ids:
            self.sim.model.geom_contype[ids] = 0
            self.sim.model.geom_conaffinity[ids] = 0
            self.sim.model.geom_rgba[ids, 3] = 0
            self.sim.model.geom_group[ids] = 5
        self.sim.data.qvel[self.belt_qvel_addr] = self._v_belt
        self.sim.data.ctrl[self.belt_actuator_id] = self._v_belt
        for x in np.linspace(1.15, 3.75, 6):
            for y in FEED_LANES_Y:
                if self.queue:
                    self._release(self.queue.pop(0), x, y)
        if self.dual_piper_layout:
            robot = self.robots[0]
            self.sim.data.qpos[robot._ref_joint_pos_indexes] = DUAL_PIPER_READY_QPOS
            self.sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
        self.sim.forward()

    def _release(self, i, x, y):
        """Spawn once or reintroduce an untouched object under the inlet hood."""
        m, d = (self.sim.model, self.sim.data)
        q, v = (int(self.obj_jnt_addrs[i]), int(self.obj_qvel_addrs[i]))
        d.eq_active[self.pending_ids[i]] = False
        state = self._return_states.pop(i, None)
        if state is None:
            yaw = self._yaws[i]
            d.qpos[q:q + 7] = [x, y, BELT_Z + self.records[i]['extent'][2] + 0.003, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            d.qvel[v:v + 6] = [self._v_belt, 0, 0, 0, 0, 0]
        else:
            # Periodic translation only: retain overshoot, lateral offset,
            # height, quaternion, linear velocity and angular velocity.
            pose, velocity, _ = state
            d.qpos[q:q + 7] = pose
            d.qvel[v:v + 6] = velocity
        ids = self.obj_geom_ids[i]
        m.geom_contype[ids] = self._collision[0][ids]
        m.geom_conaffinity[ids] = self._collision[1][ids]
        m.geom_rgba[ids] = self._collision[2][ids]
        m.geom_group[ids] = self._collision[3][ids]
        self.phases[i] = 'belt'
        self.sim.forward()

    def _pre_action(self, action, policy_step=False):
        """Run normal robot actuators and continuously drive the moving surface."""
        super()._pre_action(action, policy_step=policy_step)
        self.sim.data.ctrl[self.belt_actuator_id] = self._v_belt
        if self.sim.data.qpos[self.belt_qpos_addr] > 0.1:
            self.sim.data.qpos[self.belt_qpos_addr] -= 0.2
            self.sim.forward()

    def _touching_robot(self, i):
        ids = set(self.obj_geom_ids[i])
        for c in self.sim.data.contact:
            other = c.geom2 if c.geom1 in ids else c.geom1 if c.geom2 in ids else None
            if other is not None and (self.sim.model.geom_id2name(other) or '').startswith(('robot', 'gripper', 'fixed_mount')):
                return True
        return False

    def _transport(self):
        """Advance belt objects and recycle targets that miss the baskets.

        A gripper can momentarily touch a target while it is being carried to
        a basket.  Such a target must not be recycled while it is still in the
        gripper, but it also must not become permanently ``removed`` when the
        gripper knocks it off the belt.  Only targets whose centre is actually
        in a basket are terminally removed; every other target that leaves the
        belt is queued for re-emergence at the inlet.
        """
        m, d = (self.sim.model, self.sim.data)
        for i in range(self.n_objects):
            if self.phases[i] != 'belt':
                continue
            pos = self.obj_pos(i)

            # A target in a basket is intentionally taken off the conveyor.
            # Do this even while the gripper is still touching it so that the
            # next step cannot accidentally send it back through the inlet.
            in_bin = self._in_bin(int(self.obj_body_ids[i]), self.records[i]['extent']) >= 0
            touching = self._touching_robot(i)
            if in_bin:
                self.phases[i] = 'removed'
                continue

            # While the gripper is holding a target, leave it in the belt
            # phase.  Once released, the normal off-belt checks below either
            # recycle it or (if it reaches the outlet) preserve its periodic
            # state.  This avoids a transient contact making the target vanish.
            if touching:
                continue

            off_belt = (abs(pos[1]) > 0.25 or pos[2] < 0.8 or
                        (pos[2] > BELT_Z + self.records[i]['extent'][2] + 0.12))
            if off_belt:
                # A released target can still be above a basket for a few
                # physics steps before it falls into it.  Keep that landing
                # window; otherwise a normal high release would be mistaken
                # for a dropped target and recycled immediately.
                near_bin_xy = any(np.all(np.abs(pos[:2] - center[:2]) <= BIN_HALF[:2] + .05)
                                   for center in self.bin_centers)
                if near_bin_xy and pos[2] > BELT_Z - .1:
                    continue
                # The saved pose used for normal outlet wrapping is not valid
                # after a fall or lateral shove.  Re-spawn from the canonical
                # inlet pose instead, preserving the object's category/yaw.
                self.phases[i] = 'queued'
                self.queue.append(i)
                self._recycles += 1
                self._events.append('object_recycled')
            elif pos[0] > X_RETURN_END:
                ids = self.obj_geom_ids[i]
                m.geom_contype[ids] = 0
                m.geom_conaffinity[ids] = 0
                m.geom_rgba[ids, 3] = 0
                m.geom_group[ids] = 5
                q, v = (int(self.obj_jnt_addrs[i]), int(self.obj_qvel_addrs[i]))
                pose = d.qpos[q:q + 7].copy()
                pose[0] -= X_RETURN_END - X_REEMERGE
                half = np.abs(d.body_xmat[self.obj_body_ids[i]].reshape(3, 3)) @ self.records[i]['extent']
                self._return_states[i] = (pose, d.qvel[v:v + 6].copy(), half)
                d.qpos[q:q + 7] = [0, 0, -5 - i, 1, 0, 0, 0]
                d.qvel[v:v + 6] = 0
                d.eq_active[self.pending_ids[i]] = True
                self.phases[i] = 'queued'
                self.queue.append(i)
                self._recycles += 1
                self._events.append('object_recycled')
                self.sim.forward()
        for y in FEED_LANES_Y:
            if not self.queue:
                break
            next_id = self.queue[0]
            state = self._return_states.get(next_id)
            if state is None:
                center = np.array([X_REEMERGE, y, BELT_Z + self.records[next_id]['extent'][2] + .003])
                yaw = self._yaws[next_id]
                rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
                half = np.abs(rotation) @ self.records[next_id]['extent']
            else:
                center, half = state[0][:3], state[2]
            clear = True
            for i in range(self.n_objects):
                if self.phases[i] == 'queued':
                    continue
                other_half = np.abs(d.body_xmat[self.obj_body_ids[i]].reshape(3, 3)) @ self.records[i]['extent']
                if np.all(np.abs(self.obj_pos(i) - center) <= half + other_half + .025):
                    clear = False
                    break
            if clear and self.queue:
                self._release(self.queue.pop(0), X_REEMERGE, y)

    def _in_bin(self, bid, extent):
        """Admit by body center; extent is retained for caller compatibility."""
        position = self.sim.data.body_xpos[bid]
        for i, center in enumerate(self.bin_centers):
            local = position - center
            if np.all(np.abs(local[:2]) <= BIN_HALF[:2] + 0.003) and -0.01 <= local[2] <= 2*BIN_HALF[2]:
                return i
        return -1

    def _evaluate_sorting(self):
        """Update ownership in deterministic admission order, then purity and score."""
        bins = np.array([self._in_bin(int(b), r['extent']) if self.phases[i] != 'queued' else -1 for i, (b, r) in enumerate(zip(self.obj_body_ids, self.records))])
        stable = np.array([np.linalg.norm(self.obj_vel(i)[:3]) < 0.04 and np.linalg.norm(self.obj_vel(i)[3:]) < 0.3 and (not self._touching_robot(i)) for i in range(self.n_objects)])
        self._stable_ticks = np.where((bins >= 0) & stable & (bins == self._last_bins), self._stable_ticks + 1, 0)
        self._last_bins = bins
        ready = self._stable_ticks >= int(np.ceil(0.5 * self.control_freq))
        example_bins = [self._in_bin(int(b), r['extent']) for b, r in zip(self.example_body_ids, self.examples)]
        for k in range(self.category_count):
            if self.fixed_categories[k] < 0 and (not np.any(bins == k)) and (k not in example_bins):
                self.bin_categories[k] = -1
        for i in np.flatnonzero(ready):
            k = bins[i]
            c = self.records[i]['category']
            if self.bin_categories[k] < 0 and c not in self.bin_categories:
                self.bin_categories[k] = c
        mixed = np.zeros(self.category_count, bool)
        for k in range(self.category_count):
            cats = {self.records[i]['category'] for i in np.flatnonzero(bins == k)}
            cats.update((r['category'] for b, r in zip(example_bins, self.examples) if b == k))
            if self.fixed_categories[k] >= 0:
                cats.add(int(self.fixed_categories[k]))
            mixed[k] = len(cats) > 1
        counted = np.array([bool((ready[i] or self._counted[i]) and bins[i] >= 0 and (not mixed[bins[i]]) and (self.bin_categories[bins[i]] == r['category'])) for i, r in enumerate(self.records)])
        retained = all((b == r['basket'] for b, r in zip(example_bins, self.examples)))
        self._success_ticks = self._success_ticks + 1 if np.all(counted) and np.all(ready) and retained and (not mixed.any()) else 0
        self._counted = counted
        self._sorted_count = int(counted.sum())
        self._bin_ids = bins
        self._mixed = mixed
        self._examples_retained = retained
        return self._check_success()

    @property
    def total_score(self):
        return 100.0 * self._sorted_count / self.n_objects

    @property
    def in_box_count(self):
        return self._sorted_count

    def obj_pos(self, i):
        return self.sim.data.body_xpos[self.obj_body_ids[i]].copy()

    def obj_vel(self, i):
        v = int(self.obj_qvel_addrs[i])
        return self.sim.data.qvel[v:v + 6].copy()

    def eef_pos(self, arm=None):
        sites = self.robots[0].eef_site_id
        return self.sim.data.site_xpos[sites[arm or next(iter(sites))]].copy()

    def _check_success(self):
        return self._success_ticks >= int(np.ceil(self.control_freq))

    def reward(self, action=None):
        return 0.0

    def _task_instruction(self):
        return (
            f"Sort all {self.n_objects} target objects from the conveyor into "
            f"{self.category_count} baskets, one basket per category. "
            "Match the example objects already in a basket; the first target "
            "settled in an empty basket assigns it an unassigned category. "
            "Never mix categories and keep all examples in their original baskets. "
            "Belt speed and individual target sizes vary between Episodes and stay fixed during each Episode. "
            "Observe the motion and dimensions, adapt your grasp timing, and classify by category rather than size. "
            "A target counts after it is released and stable in the correct basket "
            f"for 0.5 seconds with linear speed <0.04 m/s and angular speed <0.3 rad/s. Each correctly sorted target earns 100/{self.n_objects} "
            "points; the maximum is 100 points. Removing a counted target or "
            "mixing its basket removes its points until corrected. "
            "Keep all targets correctly sorted, all baskets unmixed, and all "
            "examples retained for one second to succeed and finish. "
            f"The time limit is {self.horizon / self.control_freq:g} seconds; timeout retains the current score. "
            "Uncollected conveyor objects recirculate. A basket without examples can be reassigned once fully emptied."
        )

    def _observation(self, obs):
        obs = dict(obs)
        obs.pop('object-state', None)
        obs.update(score=np.array([self.total_score], np.float32), remaining_count=np.array([self.n_objects - self._sorted_count]), target_count=np.array([self.n_objects]))
        return obs

    def reset(self, **kwargs):
        return self._observation(super().reset())

    def step(self, action):
        if self._finished:
            raise ValueError('episode ended; reset required')
        a = np.asarray(action, float)
        if a.shape != (self.action_dim,) or not np.isfinite(a).all():
            raise ValueError('invalid robot action')
        obs, _, done, info = super().step(np.clip(a, -1, 1))
        before = self._sorted_count
        self._events = []
        self._transport()
        success = self._evaluate_sorting()
        self._finished = bool(success or done)
        if self._sorted_count > before:
            self._events.append('correct_placement')
        if self._sorted_count < before:
            self._events.append('correct_placement_removed')
        if success:
            self._events.append('all_objects_sorted')
        score_delta = 100.0 * (self._sorted_count - before) / self.n_objects
        info.update(total_score=self.total_score, score_delta=score_delta, completion_fraction=self._sorted_count / self.n_objects, remaining_count=self.n_objects - self._sorted_count, mixed_basket_count=int(self._mixed.sum()), examples_retained=self._examples_retained, recycle_count=self._recycles, events=self._events, success_state=success, episode_finished=self._finished)
        if self._finished:
            info['terminal_success'] = success
        return (self._observation(self._get_observations(force_update=True)), score_delta, self._finished, info)

    def get_episode_language_prompt(self):
        text = self._task_instruction()
        return dict(task_id=self.task_id, task_instruction=text, round_instruction='', round_index=None,
                    instruction=text, text=text, success_condition='All targets correctly sorted, no mixed baskets, examples retained, stable for one second.', public_context=dict(target_count=self.n_objects, basket_count=self.category_count, points_per_object=100.0 / self.n_objects, score_max=100.0))
