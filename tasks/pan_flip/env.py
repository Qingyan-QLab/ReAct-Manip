"""Actuator-driven single/dual-arm pan; free rigid pie; read-only substep evaluation."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import Arena
from react_manip.robots.resolution import default_controller_config
from react_manip.core.visual_randomization import apply_visual_scenario, resolve_scenario
from react_manip.core.scenarios import sample_scenario
from react_manip.visual_task import ManipulationTask
from .assets.gen_assets import add_pan, add_pie
from . import geometry  # register the task-local read-only adapter

ROOT = Path(__file__).resolve().parent


class PanFlip(ManipulationEnv):
    task_id = 'pan_flip'
    robot_control_mode = 'eef_delta'
    step_handles_robot_action = True
    privileged_observation_keys = ('object-state',)

    def __init__(self, robots='dual_piperx', *, seed=0, scenario=None,
                 controller_configs=None, gripper_types='default',
                 initialization_noise=None, control_freq=50, horizon=500,
                 use_camera_obs=False, use_object_obs=False,
                 has_renderer=False, has_offscreen_renderer=False, **kwargs):
        from react_manip.robots.resolution import robot_id_from_model, resolve_robosuite_robot
        resolved=resolve_robosuite_robot(robot_id_from_model(robots))
        self.robot_id=resolved.robot_id
        robots=resolved.robot_name
        if gripper_types == 'default': gripper_types=resolved.gripper_types
        self.embodiment=json.loads((ROOT/'embodiments.json').read_text())[self.robot_id]
        self.scenario = resolve_scenario(scenario)
        if self.scenario is None:
            self.scenario = sample_scenario(json.loads((ROOT.parents[1]/'configs/pan_flip/profile_default.json').read_text()), seed=seed or 0)
        if self.scenario.task_id != self.task_id:
            raise ValueError('Scenario task_id must be pan_flip')
        p = self.scenario.parameters
        self.pan_radius = float(p['pan']['radius_m'])
        self.pan_mass = float(p['pan']['mass_kg'])
        self.handle_length = float(p['pan'].get('handle_length_m', .17))
        self.grip_fraction = float(p['pan'].get('grip_fraction', 11/17))
        self.pie_radius = float(p['pie']['radius_m'])
        self.pie_half_thickness = float(p['pie']['thickness_m'])/2
        self.pie_mass = float(p['pie']['mass_kg'])
        # Old saved Scenarios describe a uniform disk and remain replayable.
        self.pie_mass_uniformity = float(p['pie'].get('mass_uniformity', 1.0))
        self.pie_density_direction_deg = float(p['pie'].get('density_direction_deg', 0.0))
        self.friction = float(p['contact']['friction'])
        self.start_xy = np.asarray(p['pie']['start_xy_m'],float)
        values = [self.pan_radius,self.pan_mass,self.pie_radius,self.pie_half_thickness,
                  self.pie_mass,self.friction,self.pie_mass_uniformity,self.pie_density_direction_deg,
                  self.handle_length,self.grip_fraction]
        if (not np.isfinite(values).all() or not .12 <= self.pan_radius <= .19
                or not .03 <= self.pie_radius <= .09 or not .004 <= self.pie_half_thickness <= .02
                or not .02 <= self.pie_mass <= .25 or not .15 <= self.pan_mass <= .7
                or not .14 <= self.handle_length <= .20 or not .5 <= self.grip_fraction <= .75
                or not 0 <= self.pie_mass_uniformity <= 1
                or not .05 <= self.friction <= 1.2 or self.start_xy.shape != (2,)
                or not np.isfinite(self.start_xy).all()
                or np.linalg.norm(self.start_xy)+self.pie_radius > self.pan_radius-.015):
            raise ValueError('Invalid pan/pie/contact parameters')
        self.position_scale, self.rotation_scale = .15, 1.0
        self._reset_tracking()
        cfg = deepcopy(controller_configs or default_controller_config(robots))
        parts = cfg['body_parts']
        for arm in parts.get('arms',parts).values():
            if arm['type'] != 'OSC_POSE' or arm.get('impedance_mode','fixed') != 'fixed':
                raise ValueError('pan_flip requires fixed-impedance OSC_POSE')
            arm.update(input_ref_frame='world', input_type='delta',kp=self.embodiment.get('kp',350),damping_ratio=1,
                       output_max=[self.position_scale]*3+[self.rotation_scale]*3,
                       output_min=[-self.position_scale]*3+[-self.rotation_scale]*3)
        kwargs.setdefault('render_camera','pan_side')
        kwargs.setdefault('camera_names','pan_side')
        from react_manip.optional_integrations import preserve_native_action_units
        preserve_native_action_units(cfg, self.robot_id)
        super().__init__(robots=robots,seed=seed,controller_configs=cfg,
                         gripper_types=gripper_types,initialization_noise=None,
                         control_freq=control_freq,horizon=horizon,lite_physics=False,
                         use_camera_obs=use_camera_obs,has_renderer=has_renderer,
                         has_offscreen_renderer=has_offscreen_renderer,**kwargs)

    def _load_model(self):
        super()._load_model()
        robot = self.robots[0]
        robot.robot_model.set_base_xpos(self.embodiment.get("base_pos",[0,0,0]))
        if self.robot_id in ('panda','panda_libero'):
            robot.init_qpos = np.array([0,np.pi/16,0,-5*np.pi/6,0,np.pi/16+5*np.pi/6,np.pi/4])
        # Task-specific wrist pitch / roll keeps the handle forward, the pan
        # level and the wrist camera above the handle on the official PiPER-X.
        if 'init_qpos' in self.embodiment:
            robot.init_qpos=np.asarray(self.embodiment['init_qpos'])
        eef = robot.robot_model.worldbody.find(".//body[@name='gripper0_right_eef']")
        if eef is None:
            raise ValueError('Expected right gripper EEF attachment body')
        pan=add_pan(eef,self.pan_radius,self.pan_mass,self.friction,
                self.handle_length,self.grip_fraction)
        if 'tool_quat' in self.embodiment:
            from scipy.spatial.transform import Rotation
            quat=np.asarray(self.embodiment['tool_quat'])
            rotation=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
            offset=rotation@np.array([self.pan_radius-.02+self.handle_length*self.grip_fraction,0,0])
            pan.set('pos',' '.join(map(str,offset)))
            pan.set('quat',' '.join(map(str,quat)))
        arena = Arena(str(ROOT/'assets/arena.xml'))
        add_pie(arena.worldbody,arena.asset,self.pie_radius,self.pie_half_thickness,
                self.pie_mass,self.friction,self.pie_mass_uniformity,self.pie_density_direction_deg)
        import xml.etree.ElementTree as ET
        from react_manip.cameras import look_at
        pos = np.array([1.05,-1.65,1.65]); target=np.array([.62,0,1.02])
        rot=look_at(pos,target)
        ET.SubElement(arena.worldbody,'camera',name='pan_side',pos=' '.join(map(str,pos)),
                      xyaxes=' '.join(map(str,np.r_[rot[:,0],rot[:,1]])),fovy='42')
        apply_visual_scenario(arena,self.scenario)
        self.model = ManipulationTask(mujoco_arena=arena,
                                     mujoco_robots=[r.robot_model for r in self.robots])
        self.model.root.find("option").set('cone','elliptic')
        self.model.root.find("option").set('iterations','100')
        flag=self.model.root.find("option").find('flag')
        if flag is None: flag=ET.SubElement(self.model.root.find("option"),'flag')
        flag.set('multiccd','enable')
        self.camera_workspace = dict(center=(.6,0,1.0),half_size=(.45,.4))

    def _setup_references(self):
        super()._setup_references()
        m=self.sim.model
        self.pan_body_id=m.body_name2id('pan'); self.pie_body_id=m.body_name2id('pie')
        self.pie_geom_id=m.geom_name2id('pie_collision')
        self.bottom_geom_id=m.geom_name2id('pan_bottom')
        self.pan_geom_ids={m.geom_name2id('pan_bottom')} | {m.geom_name2id(f'pan_wall_{i}') for i in range(32)}
        self.pie_qpos_addr=m.get_joint_qpos_addr('pie_joint')[0]
        self.pie_qvel_addr=m.get_joint_qvel_addr('pie_joint')[0]
        self.moving_robot_geoms={i for i in range(m.ngeom)
                                 if m.geom_bodyid[i] != self.pie_body_id
                                 and m.body_weldid[m.geom_bodyid[i]] != 0}
        self.gripper_geom_ids={m.geom_name2id(n) for n in self.robots[0].gripper['right'].contact_geoms}
        self.handle_geom_id=m.geom_name2id('pan_handle')
        self.floor_geom_id=m.geom_name2id('floor')
        self.wheel_geom_ids={i for i,n in enumerate(m.geom_names) if n and
                             n.startswith('robot0_mobile_') and n.endswith(('_tire','_roller'))}

    def _reset_tracking(self):
        self.phase='ready'; self._success=False; self._finished=False; self._failure=None
        self._armed=False; self._support_time=0.; self._air_time=0.; self._settle_time=0.
        self._airborne=False; self._landed=False; self.max_clearance=0.; self.flight_duration=0.
        # Airborne motion remains useful telemetry, but it is no longer a
        # prerequisite: a contact-guided pan flip is a valid completion.
        self._flipped=False
        self._flipped_in_air=False
        self._last_eval_time=0.; self._events=[]; self._milestones=[]
        self.release_position=None; self.release_velocity=None; self.release_angular_velocity=None
        self._candidate_release=None; self._tracking=False

    def _reset_internal(self):
        self._reset_tracking()
        super()._reset_internal()
        self.sim.forward()
        # Initialization only. The pan is part of the articulated robot and
        # the pie is a free body; neither pose/velocity is rewritten in step.
        if self.robot_id in ('piperx', 'dual_piperx', 'mobile_dual_piperx'):
            if self.pan_mat()[2, 2] < np.cos(np.deg2rad(1)):
                raise ValueError('Initial pan must be level and face upward')
        p=self.pan_pos()+self.pan_mat() @ np.r_[self.start_xy,self.pie_half_thickness+.0005]
        quat=self.sim.data.body_xquat[self.pan_body_id].copy()
        self.sim.data.set_joint_qpos('pie_joint',np.r_[p,quat])
        self.sim.data.qvel[self.pie_qvel_addr:self.pie_qvel_addr+6]=0
        self.sim.forward()
        self._last_eval_time=float(self.sim.data.time)
        self._tracking=True

    def pan_pos(self): return self.sim.data.body_xpos[self.pan_body_id].copy()
    def pan_mat(self): return self.sim.data.body_xmat[self.pan_body_id].reshape(3,3).copy()
    def pie_pos(self): return self.sim.data.body_xpos[self.pie_body_id].copy()
    def pie_mat(self): return self.sim.data.body_xmat[self.pie_body_id].reshape(3,3).copy()
    def pie_velocity(self): return self.sim.data.get_body_xvelp('pie').copy()
    def pie_angular_velocity(self): return self.sim.data.get_body_xvelr('pie').copy()
    def eef_pos(self,arm="right"): return self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]].copy()
    def eef_mat(self,arm="right"): return self.sim.data.site_xmat[self.robots[0].eef_site_id[arm]].reshape(3,3).copy()

    def relative_state(self):
        R=self.pan_mat(); offset=self.pie_pos()-self.pan_pos()
        local=R.T @ offset
        normal=R.T @ self.pie_mat()[:,2]
        support_extent=self.pie_radius*np.sqrt(max(0.,1-normal[2]**2))+self.pie_half_thickness*abs(normal[2])
        pan_v=self.sim.data.get_body_xvelp('pan')
        pan_w=self.sim.data.get_body_xvelr('pan')
        relative_v=self.pie_velocity()-pan_v-np.cross(pan_w,offset)
        return local,normal,float(local[2]-support_extent),relative_v,self.pie_angular_velocity()-pan_w

    def _contacts(self):
        others=set()
        for c in self.sim.data.contact[:self.sim.data.ncon]:
            if c.dist > .0005: continue
            if c.geom1 == self.pie_geom_id: others.add(c.geom2)
            if c.geom2 == self.pie_geom_id: others.add(c.geom1)
        return bool(others & self.pan_geom_ids),self.bottom_geom_id in others,others-self.pan_geom_ids

    def _robot_collision(self):
        """Allow the fixed handle's finger contacts, reject arm/tool obstacles."""
        for c in self.sim.data.contact[:self.sim.data.ncon]:
            if c.dist >= -.0001:
                continue
            pair={c.geom1,c.geom2}
            if self.floor_geom_id in pair and pair & self.wheel_geom_ids:
                continue
            if self.pie_geom_id in pair or not pair & self.moving_robot_geoms:
                continue
            if pair <= self.gripper_geom_ids | {self.handle_geom_id}:
                continue
            return tuple(self.sim.model.geom_id2name(i) for i in (c.geom1,c.geom2))
        return None

    def _update_observables(self, force=False):
        super()._update_observables(force=force)
        if force or not self._tracking: return
        now=float(self.sim.data.time); dt=now-self._last_eval_time
        if dt <= 0: return
        self._last_eval_time=now
        self._evaluate_physics(dt)

    def _evaluate_physics(self, dt):
        if self._success or self._failure: return
        contact,bottom,foreign=self._contacts()
        local,normal,clearance,rel_v,rel_w=self.relative_state()
        if foreign:
            self._failure='pie_contact_outside_pan'; return
        if self._robot_collision():
            self._failure='robot_or_tool_collision'; return
        self._support_time=self._support_time+dt if bottom else 0.
        if self._support_time >= .1: self._armed=True
        if self._armed and not contact and not self._landed:
            if self._candidate_release is None:
                self._candidate_release=(self.pie_pos(),self.pie_velocity(),self.pie_angular_velocity())
            self._air_time+=dt
            self.max_clearance=max(self.max_clearance,clearance)
            if not self._airborne and self._air_time >= .04 and clearance >= .015:
                self._airborne=True; self.phase='flight'
                self.release_position,self.release_velocity,self.release_angular_velocity=self._candidate_release
                self._milestones.append('pie_airborne')
        elif not self._airborne:
            self._air_time=0.; self._candidate_release=None
        if self._airborne and not self._landed:
            self.flight_duration=self._air_time
            if not contact and self.pie_mat()[2,2] < -.5:
                if not self._flipped_in_air:
                    self._milestones.append('pie_inverted_in_air')
                self._flipped_in_air=True
            if contact:
                self._landed=True; self.phase='landing'; self._milestones.append('pie_recontact')
        # Record the A-to-B face transition after the initial supported phase.
        # This intentionally does not require the pie to be airborne or out of
        # contact with the pan.
        if self._armed and not self._flipped and normal[2] < -.5:
            self._flipped=True
            self._milestones.append('pie_flipped')
        inside=np.linalg.norm(local[:2])+self.pie_radius <= self.pan_radius-.003
        flipped=normal[2] < -np.cos(np.deg2rad(20))
        upright=self.pan_mat()[2,2] > np.cos(np.deg2rad(20))
        stable=(self._armed and self._flipped and bottom and inside and flipped and upright
                and abs(clearance)<.004 and np.linalg.norm(rel_v)<.08 and np.linalg.norm(rel_w)<.5)
        self._settle_time=self._settle_time+dt if stable else 0.
        if self._settle_time >= .5:
            self._success=True; self.phase='success'; self._events.append('pie_flipped_and_settled')
        if self.pie_pos()[2] < .15 or np.linalg.norm(local)>1.5:
            self._failure='pie_dropped'

    def _check_success(self): return self._success

    def _task_instruction(self):
        return (
            "Use the pan to turn the rigid pie once. "
            "Flip the pie from golden face A to brown face B, "
            "then leave it supported inside the upright pan. "
            "The pie may remain in contact with the pan; an airborne toss is optional. "
            "It must remain supported in the pan for at least 0.5 seconds. "
            "Start with bottom support for 0.1 seconds, then complete the face transition. "
            "At settlement the pan must be within 20 degrees of upright and face B within 20 degrees of the pan normal, "
            "the entire pie inside with 0.003 m rim clearance, bottom clearance <0.004 m, "
            "relative linear speed <0.08 m/s and angular speed <0.5 rad/s. "
            "Outside-pan pie contact, forbidden robot/tool collision, or dropping the pie "
            "ends the attempt with zero points. "
            f"Success earns 100 and ends the episode; timeout at {self.horizon / self.control_freq:g} seconds earns zero."
        )

    def get_episode_language_prompt(self):
        from react_manip.core.language import task_language_prompt
        return task_language_prompt(
            self, task_id='pan_flip',
            task_instruction=self._task_instruction(),
            success_condition='After a genuine A-to-B face transition, face B points up and the pie stays supported inside the upright pan for 0.5 seconds; airborne flight is optional.',
            constraints=(),
        )

    def reward(self,action=None): return float(self._success)
    @property
    def score(self): return 100. if self._success else 0.

    def step(self,action):
        if self._finished: raise RuntimeError('Episode ended; reset required')
        a=np.asarray(action,float)
        if a.shape != (self.action_dim,) or not np.isfinite(a).all():
            raise ValueError(f'Expected finite action shape {(self.action_dim,)}')
        self._events=[]; self._milestones=[]
        obs,_,done,_=super().step(np.clip(a,-1,1))
        if not np.isfinite(self.sim.data.qpos).all() or not np.isfinite(self.sim.data.qvel).all():
            self._failure='non_finite_state'; self._success=False
        if self._failure: self.phase='failed'
        terminated=bool(self._success or self._failure)
        truncated=bool(done and not terminated)
        self._finished=terminated or truncated
        local,normal,clearance,_,_=self.relative_state()
        info=dict(success_state=self._success,terminal_success=self._success,total_score=self.score,
                  events=list(self._events),milestones=list(self._milestones),phase=self.phase,
                  airborne=self._airborne,landed=self._landed,flipped=self._flipped,
                  face_alignment=float(normal[2]),
                  flipped_in_air=self._flipped_in_air,
                  clearance_m=clearance,flight_duration=self.flight_duration,settle_time=self._settle_time,
                  tool_attachment='fixed',elapsed_sec=float(self.sim.data.time),
                  failure={'recoverable':False,'reason':self._failure} if self._failure else None)
        return obs,float(self._success),terminated,truncated,info
