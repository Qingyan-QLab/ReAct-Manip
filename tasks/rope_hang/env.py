"""Robots manipulate a weighted articulated chain using physical contacts only."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import Arena
from react_manip.core.scenarios import sample_scenario
from react_manip.core.visual_randomization import apply_visual_scenario, resolve_scenario
from react_manip.robots.resolution import default_controller_config
from react_manip.visual_task import ManipulationTask
from .model import Rope, RodChain, BAR, TABLE_Z, add_fixture
from .state import Stability, hanging_geometry, support_coordinates
from .initialization import InitialLayout
from . import geometry  # noqa: F401

ROOT=Path(__file__).resolve().parent


class RopeHang(ManipulationEnv):
    task_id='rope_hang'
    robot_control_mode='eef_delta'
    step_handles_robot_action=True
    privileged_observation_keys=('object-state',)

    def __init__(self, robots='dual_piperx', *, seed=0, scenario=None, controller_configs=None,
                 gripper_types='default', initialization_noise=None, control_freq=50,
                 horizon=9000, has_renderer=False, has_offscreen_renderer=False,
                 use_camera_obs=False, use_object_obs=False, **kwargs):
        from react_manip.robots.resolution import robot_id_from_model, resolve_robosuite_robot
        resolved=resolve_robosuite_robot(robot_id_from_model(robots))
        self.robot_id=resolved.robot_id
        robots=resolved.robot_name
        if gripper_types=='default':
            gripper_types=resolved.gripper_types
        self.embodiment_config=kwargs.pop('embodiment_config', None)
        if self.embodiment_config is None:
            self.embodiment_config=json.loads((ROOT/'embodiments.json').read_text()).get(self.robot_id,{})
        self.embodiment_config=deepcopy(self.embodiment_config)
        self.active_arm=self.embodiment_config.get('active_arm','right')
        if self.active_arm not in resolved.arms:
            raise ValueError('active_arm is unavailable on this embodiment')
        self.scenario=resolve_scenario(scenario)
        if self.scenario is None:
            profile=json.loads((ROOT.parents[1]/'configs/rope_hang/profile_default.json').read_text())
            self.scenario=sample_scenario(profile,seed=seed or 0)
        if self.scenario.task_id!=self.task_id:
            raise ValueError('Scenario task_id must be rope_hang')
        # Retain internal geometry names and old Scenario replay compatibility.
        self.rope=(RodChain(**self.scenario.parameters['chain'])
                   if 'chain' in self.scenario.parameters else
                   Rope(**self.scenario.parameters.get('rope',{})))
        self.episode_instance=kwargs.pop('episode_instance', None)
        self.initial_layout=InitialLayout(**self.scenario.parameters.get('initialization',{}))
        self.initial_layout.points(self.rope)  # reject overlap/out-of-table configurations before compiling
        self.rebalance_config=self.scenario.parameters.get('rebalance',{})
        if self.initial_layout.layout=='draped' and self.robot_id=='panda':
            q=np.asarray(self.rebalance_config.get('standby_qpos'),float)
            if q.shape!=(7,) or not np.isfinite(q).all():
                raise ValueError('Draped reset requires seven finite rebalance.standby_qpos values')
        self.position_scale=.04
        self.rotation_scale=.15
        self._references_ready=False
        self._reset_tracking()
        cfg=deepcopy(controller_configs or default_controller_config(robots))
        parts=cfg['body_parts']
        for arm_name in resolved.arms:
            arm=parts.get('arms',parts)[arm_name]
            if arm['type']!='OSC_POSE' or arm.get('impedance_mode','fixed')!='fixed':
                raise ValueError('rope_hang requires fixed-impedance OSC_POSE')
            arm.update(input_ref_frame='world',input_type='delta',kp=200,damping_ratio=1,
                       output_max=[self.position_scale]*3+[self.rotation_scale]*3,
                       output_min=[-self.position_scale]*3+[-self.rotation_scale]*3)
        kwargs.setdefault('camera_names','agentview')
        kwargs.setdefault('render_camera','agentview')
        from react_manip.optional_integrations import preserve_native_action_units
        preserve_native_action_units(cfg, self.robot_id)
        super().__init__(robots=robots,controller_configs=cfg,gripper_types='default',
                         initialization_noise=initialization_noise,control_freq=control_freq,
                         horizon=horizon,lite_physics=False,has_renderer=has_renderer,
                         has_offscreen_renderer=has_offscreen_renderer,use_camera_obs=use_camera_obs,
                         seed=self.scenario.seed,**kwargs)

    def initialize_time(self, control_freq):
        super().initialize_time(control_freq)
        self.model_timestep=float(self.sim.model.opt.timestep)

    def _load_model(self):
        super()._load_model()
        self._references_ready=False
        if 'stand' in self.embodiment_config:
            from react_manip.robots.mounts import use_existing_mount, install_fixed_stand
            model=self.robots[0].robot_model
            if self.robot_id in ('mobile_dual_piperx', 'aloha_mobile'):
                raise ValueError('A free-root mobile robot cannot use a fixed stand')
            use_existing_mount(model)
            stand=dict(self.embodiment_config['stand'])
            deck=stand.pop('deck',None)
            install_fixed_stand(model,**stand)
            if deck is not None:
                import xml.etree.ElementTree as ET
                from .model import vec
                ET.SubElement(model.worldbody.find('body'),'geom',
                    name=model.naming_prefix+'fixed_stand_deck',type='box',
                    pos=vec(deck['pos']),size=vec(deck['size']),
                    rgba='.25 .28 .32 1',friction='.8 .01 .001',group='1')
        base=np.asarray(self.embodiment_config.get('base_pos',[0,0,0]),float)
        if base.shape!=(3,) or not np.isfinite(base).all():
            raise ValueError('embodiment base_pos must contain three finite coordinates')
        self.robots[0].robot_model.set_base_xpos(base)
        if 'init_qpos' in self.embodiment_config:
            q=np.asarray(self.embodiment_config['init_qpos'],float)
            expected=np.asarray(self.robots[0].robot_model.init_qpos).shape
            if q.shape!=expected or not np.isfinite(q).all():
                raise ValueError(f'embodiment init_qpos must have shape {expected}')
            self.robots[0].init_qpos=q
        elif self.initial_layout.layout=='draped' and self.robot_id=='panda':
            self.robots[0].init_qpos=np.asarray(self.rebalance_config['standby_qpos'],float)
        arena=Arena(str(ROOT/'assets/arena.xml'))
        add_fixture(arena.root)
        if self.initial_layout.layout=='draped':
            import xml.etree.ElementTree as ET
            from react_manip.cameras import look_at
            from .model import vec
            position=np.array([.10,-1.40,1.15])
            rotation=look_at(position,np.array([.55,-.25,.85]))
            ET.SubElement(arena.worldbody,'camera',name='rebalance_side',pos=vec(position),
                          xyaxes=vec(np.r_[rotation[:,0],rotation[:,1]]),fovy='50')
        apply_visual_scenario(arena,self.scenario)
        self.model=ManipulationTask(mujoco_arena=arena,
                                   mujoco_robots=[r.robot_model for r in self.robots],mujoco_objects=[])
        self.model.root.find('option').attrib.update(arena.root.find('option').attrib)
        # Append after robosuite's merge, which does not preserve plugin extensions.
        self.rope.add_to(self.model.root)
        self.camera_workspace=dict(center=(.55,0,.85),half_size=(.55,.55))

    def _setup_references(self):
        super()._setup_references()
        m=self.sim.model
        self.node_ids=[m.site_name2id(f'rope_node_{i}') for i in range(self.rope.segments+1)]
        self.rope_geom_links={m.geom_name2id(f'rope_geom_{i}'):i for i in range(self.rope.segments)}
        self.rope_geom_ids=set(self.rope_geom_links)
        self.rope_body_ids=[m.body_name2id(f'rope_{i}') for i in range(self.rope.segments)]
        self.bar_id=m.geom_name2id('bar')
        self.floor_id=m.geom_name2id('floor')
        self.table_ids={i for i in range(m.ngeom) if (m.geom_id2name(i) or '').startswith('table_')}
        gripper=self.robots[0].gripper[self.active_arm]
        self.finger_groups=[{m.geom_name2id(n) for n in gripper.important_geoms[k]}
                            for k in ('left_finger','right_finger')]
        self.gripper_ids={m.geom_name2id(n) for n in gripper.contact_geoms}
        if 'fingerpad_friction' in self.embodiment_config:
            for group in self.finger_groups:
                for gid in group:
                    if any(part in m.geom_id2name(gid) for part in ('pad','tip')):
                        m.geom_friction[gid,0]=self.embodiment_config['fingerpad_friction']
        self._references_ready=True

    def _reset_tracking(self):
        self.stability=Stability()
        self._grasped=self._lifted=self._success=self._finished=False
        self._grasp_duration=0.
        self._previous_points=None
        self._previous_time=None
        self._last_grip=-1.
        self._milestones=[]
        self.phase='ready'
        self._failure=None
        self._eligible=False
        self._bar_contact=False
        self._external_contact=False
        self.max_node_speed=0.

    def _reset_internal(self):
        super()._reset_internal()
        if self.episode_instance is not None:
            from dataclasses import replace
            from .instances import validate_episode_instance
            validate_episode_instance(self.scenario.parameters, self.episode_instance)
            self.initial_layout=InitialLayout(**self.scenario.parameters['initialization'])
            if self.initial_layout.layout != 'draped':
                self.initial_layout=replace(self.initial_layout, center_xy=tuple(self.episode_instance['center_xy']))
        self.initial_layout.apply(self.sim.model._model,self.sim.data._data,self.rope)
        if self.episode_instance is not None:
            model=self.sim.model
            root=model.body_name2id(self.robots[0].robot_model.root_body)
            def on_robot(body):
                while body:
                    if body==root:
                        return True
                    body=int(model.body_parentid[body])
                return False
            robot_geoms={i for i in range(model.ngeom) if on_robot(int(model.geom_bodyid[i]))}
            for c in self.sim.data.contact[:self.sim.data.ncon]:
                a,b=int(c.geom1),int(c.geom2)
                if c.dist < -1e-6 and ((a in self.rope_geom_ids and b in robot_geoms) or
                                      (b in self.rope_geom_ids and a in robot_geoms)):
                    raise ValueError('Episode chain initially penetrates robot')
        self._reset_tracking()
        if self.initial_layout.layout=='draped':
            # This mode starts suspended. No grasp or rescue is fabricated.
            self._lifted=True
            self.phase='watching_slip'

    def get_episode_language_prompt(self):
        from react_manip.core.language import task_language_prompt
        return task_language_prompt(
            self, task_id=self.task_id, task_instruction=self._task_instruction(),
            success_condition="The chain remains independently suspended without sustained sliding for 5 seconds after release.",
            constraints=("Do not tie or wrap the chain.",),
        )

    def rope_points(self):
        return self.sim.data.site_xpos[self.node_ids].copy()

    def eef_pos(self, arm=None):
        arm=arm or self.active_arm
        return self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]].copy()

    def eef_mat(self, arm=None):
        arm=arm or self.active_arm
        return self.sim.data.site_xmat[self.robots[0].eef_site_id[arm]].reshape(3,3).copy()

    def _update_observables(self, force=False):
        super()._update_observables(force=force)
        if force or not self._references_ready:
            return
        now=float(self.sim.data.time)
        p=self.rope_points()
        if self._previous_time is None:
            self._previous_time=now;self._previous_points=p
            return
        dt=now-self._previous_time
        if dt<0:
            self._failure='physics_instability'
            return
        if dt==0:
            return
        if any(self.sim.data._data.warning[i].number for i in (4,5,6)):
            self._failure='physics_instability'
        self.max_node_speed=float(np.max(np.linalg.norm(p-self._previous_points,axis=1))/dt)
        self._previous_points=p;self._previous_time=now
        bar=False;external=False;floor=False;table=False;finger=[False,False]
        contact_links=set()
        for c in self.sim.data.contact[:self.sim.data.ncon]:
            a,b=int(c.geom1),int(c.geom2)
            if a not in self.rope_geom_ids and b not in self.rope_geom_ids:
                continue
            if a in self.rope_geom_ids and b in self.rope_geom_ids:
                # Self-supported knots/folds are not the requested single drape.
                external=True
                continue
            other=b if a in self.rope_geom_ids else a
            if other==self.bar_id:
                bar=True
                contact_links.add(self.rope_geom_links[a if a in self.rope_geom_ids else b])
            else:
                external=True
                floor |= other==self.floor_id
                table |= other in self.table_ids
            for k,group in enumerate(self.finger_groups):
                finger[k] |= other in group
        self._grasp_duration=self._grasp_duration+dt if all(finger) else 0.
        if not self._grasped and self._grasp_duration>=.10:
            self._grasped=True;self.phase='grasped';self._milestones.append('rope_grasped')
        if self._grasped and not self._lifted and not table and not floor and p[:,2].max()>TABLE_Z+.08:
            self._lifted=True;self.phase='lifted';self._milestones.append('rope_lifted')
        self._bar_contact=bar;self._external_contact=external
        clear=float(np.min(np.linalg.norm(p-self.eef_pos(),axis=1)))>.12
        self._eligible=bool(self._lifted and self._last_grip<-.5 and bar and not external and clear
                            and hanging_geometry(p,self.rope.radius))
        support=(support_coordinates(p,self.rope.ds,self.rope.radius,contact_links)
                 if self._eligible else None)
        self._eligible=self._eligible and support is not None
        self._success=self.stability.update(support,self._eligible,dt)
        if self._eligible:
            self.phase='observing_support'
        elif self._lifted:
            self.phase='adjusting'
        if floor:
            self._failure='rope_dropped_on_floor'
        if not np.isfinite(self.sim.data.qpos).all() or not np.isfinite(self.sim.data.qvel).all():
            self._failure='non_finite_state'

    def reward(self, action=None):
        return float(self._success)

    def _task_instruction(self):
        start = ("Catch the slipping nonuniform articulated metal-rod chain already draped over the horizontal bar and redistribute its hanging lengths. "
                 if self.initial_layout.layout == 'draped' else
                 "Pick up the folded articulated metal-rod chain from the table and swing it over the horizontal bar. ")
        return (start + "Open the gripper and retreat at least 0.12 m from every chain node. "
                "Leave a single drape with its ends on opposite sides and both at least 0.06 m below the bar. "
                "Only the bar may support the chain; table, robot and self-contact invalidate the hold. Do not tie or wrap it. "
                "Hold for five continuous seconds with material and axial support slip ranges <=0.010 m and "
                "absolute slip trends <=0.002 m/s. Free-end swinging alone is allowed. "
                "Invalid support or excessive sliding restarts the hold; floor contact ends the attempt with zero points. "
                f"Success earns 100 and ends the episode; timeout at {self.horizon / self.control_freq:g} seconds earns zero.")

    def _check_success(self):
        return bool(self._success)

    def step(self, action):
        if self._finished:
            raise RuntimeError('Episode ended; reset required')
        a=np.asarray(action,float)
        if a.shape!=(self.action_dim,) or not np.isfinite(a).all():
            raise ValueError(f'Expected finite action shape {(self.action_dim,)}')
        start,end=self.robots[0].composite_controller._action_split_indexes[self.active_arm+'_gripper']
        self._last_grip=float(a[start])
        obs,_,done,_=super().step(np.clip(a,-1,1))
        if self._failure:
            self._success=False;self.phase='failed'
        elif self._success:
            self.phase='success'
        terminated=bool(self._success or self._failure)
        truncated=bool(done and not terminated)
        self._finished=terminated or truncated
        milestones=self._milestones;self._milestones=[]
        info=dict(success_state=self._success,terminal_success=self._success,
                  success_rule="supported_no_slip_v2",
                  events=['rope_hung'] if self._success else [],milestones=milestones,
                  total_score=100.*float(self._success),phase=self.phase,
                  stable_seconds=self.stability.elapsed,
                  support_slip_range_m=self.stability.range.tolist(),
                  support_slip_trend_m_s=self.stability.trend.tolist(),
                  max_node_speed_m_s=self.max_node_speed,elapsed_sec=float(self.sim.data.time),
                  failure={'recoverable':False,'reason':self._failure} if self._failure else None)
        return obs,float(self._success),terminated,truncated,info
