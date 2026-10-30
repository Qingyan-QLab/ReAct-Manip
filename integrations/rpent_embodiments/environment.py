"""Evaluator-owned native control and public proprioception for RPent ports."""
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from integrations.harness_vla.rgbd import RGBDSensor
from react_manip.core.protocol import normalize_reset, normalize_step
from .catalog import EMBODIMENTS, SUPPORTED_TASKS, camera_ids

SCHEMA = 'react-manip.rpent.v1'


def task_definition(task_id, robot_id):
    """Create an explicitly experimental task variant without changing the bank."""
    from . import register
    from react_manip.builtin import discover_task_definitions
    from react_manip.core.registry import TaskDefinition
    register()
    spec, factory = next((s,f) for s,f in discover_task_definitions() if s.task_id == task_id)
    embodiment = EMBODIMENTS[robot_id]
    if task_id not in SUPPORTED_TASKS[robot_id]:
        raise ValueError(f'{robot_id}/{task_id} has no validated task port; available: {SUPPORTED_TASKS[robot_id]}')
    if spec.required_arms > len(embodiment.arms):
        raise ValueError(f'{task_id} requires {spec.required_arms} arms')
    # Existing car task hardcodes a steering-wheel interaction requiring two
    # arms. A mobile single arm is not automatically compatible with it.
    variant=replace(spec, supported_robots=(robot_id,),
        metadata={**spec.metadata, 'default_robot':robot_id, 'evaluation_track':'rpent_embodiments_experimental'})
    from react_manip.builtin import _camera_factory
    return TaskDefinition(variant, _camera_factory(factory.__wrapped__, variant))


class EmbodimentEnv:
    def __init__(self, task_id, robot_id, *, seed=0, cameras=True, scenario=None, wrapped=None):
        self.definition = task_definition(task_id, robot_id)
        self.robot_id = robot_id
        self._spec = EMBODIMENTS[robot_id]
        self._bundle = None
        if len(self._spec.arms)==2:
            from .models import validate_bundle
            self._bundle=validate_bundle(robot_id)
        self._wrapped = wrapped if wrapped is not None else self.definition.factory(
            seed=seed, robot_id=robot_id, scenario=scenario,has_offscreen_renderer=False)
        self.env = self._wrapped.env
        self._sensor = RGBDSensor(256,256)
        self._cameras = camera_ids(robot_id) if cameras else ()
        self._low = self._high = None
        self._gripper_targets = {arm:1. for arm in self._spec.arms}

    def __getattr__(self, name): return getattr(self._wrapped, name)

    @property
    def action_spec(self): return -np.ones(self.action_dim), np.ones(self.action_dim)

    def _calibration(self):
        robot = self.env.robots[0]
        layout = {k:list(v) for k,v in robot.composite_controller._action_split_indexes.items()}
        low,high=map(lambda a:np.array(a,dtype=float),self.env.action_spec)
        for arm in self._spec.arms:
            part=robot.part_controllers[arm]
            if self._spec.source=='robodojo':
                ids=part.joint_indexes['joints'];limits=np.asarray(self.env.sim.model.jnt_range[ids])
                start,end=layout[arm];low[start:end]=limits[:,0];high[start:end]=limits[:,1]
        if np.any(~np.isfinite(low)) or np.any(high<=low): raise ValueError('Invalid native action bounds')
        self._low,self._high=low,high
        arms={};proprio={}
        for arm in self._spec.arms:
            part=robot.part_controllers[arm];part.update(force=True)
            site=robot.eef_site_id[arm];sim=self.env.sim
            position=np.array(sim.data.site_xpos[site]);rotation=np.array(sim.data.site_xmat[site]).reshape(3,3)
            qids=part.qpos_index;vids=part.qvel_index
            import mujoco
            jp=np.zeros((3,sim.model.nv));jr=np.zeros_like(jp)
            mujoco.mj_jacSite(sim.model._model,sim.data._data,jp,jr,site)
            g=robot.gripper[arm]
            gj=[sim.model.joint_name2id(n) for n in g.joints]
            gq=np.array([sim.data.qpos[sim.model.jnt_qposadr[i]] for i in gj])
            gr=np.array(sim.model.jnt_range[gj])
            if len(self._spec.arms)==2:
                command=self._bundle['gripper']['command_range']
                opening=float(np.clip((gq[0]-command[0])/(command[1]-command[0]),0,1))
            else: opening=float(np.clip(np.sum(np.abs(gq))/.08,0,1))
            proprio[arm]=dict(qpos=np.array(sim.data.qpos[qids]),eef_pos=position,
                eef_quat_xyzw=Rotation.from_matrix(rotation).as_quat(),gripper_qpos=gq,
                gripper_opening=opening,gripper_target_opening=self._gripper_targets[arm],
                jacobian=np.vstack([jp[:,vids],jr[:,vids]]))
            arms[arm]=dict(mode='joint_absolute' if self._spec.source=='robodojo' else 'osc_delta',
                input_ref_frame=getattr(part,'input_ref_frame','world'),
                world_from_control_rotation=np.asarray(getattr(part,'origin_ori',np.eye(3))),
                **{key:np.asarray(getattr(part,key)) for key in ('input_min','input_max','output_min','output_max')})
        contract=dict(schema_version=SCHEMA,robot_id=self.robot_id,action_layout=layout,arms=arms,
                      native_low=low,native_high=high,control_hz=float(self.env.control_freq),
                      gripper_open=-1.,gripper_close=1.)
        if self._spec.mobile:
            contract['base_mode_index']=len(low)-1
        return contract,proprio

    def _enrich(self,obs):
        contract,proprio=self._calibration()
        images={};depths={};masks={};calibrations={}
        for camera in self._cameras:
            images[camera],depths[camera],masks[camera],calibrations[camera]=self._sensor.capture(self.env.sim,camera)
        obs=dict(obs,control_contract=contract,proprio=proprio,images=images,
                 depth_m=depths,depth_valid=masks,camera_calibration=calibrations,
                 observation_schema='react-manip.rgbd.v1',embodiment=self.robot_id)
        if len(self._spec.arms)==2:
            offset=self._bundle['gripper']['tcp_offset']
            obs['tcp_offsets']={arm:offset for arm in self._spec.arms}
        else:
            obs['tcp_offsets']={'right':[0,0,0]}
        # Stable aliases consumed by the existing analytic primitive toolkit.
        for arm,p in proprio.items():
            prefix='robot0_' if len(proprio)==1 else f'robot0_{arm}_'
            obs[prefix+'eef_pos']=p['eef_pos'];obs[prefix+'eef_quat_site']=p['eef_quat_xyzw']
            robot=self.env.robots[0]
            hand_rotation=np.asarray(self.env.sim.data.get_body_xmat(robot.robot_model.eef_name[arm])).reshape(3,3)
            obs[prefix+'eef_quat']=Rotation.from_matrix(hand_rotation).as_quat()
            obs[prefix+'gripper_qpos']=p['gripper_qpos']
        if 'eef_pos' in obs and len(proprio)==1:
            obs['eef_pos']=proprio['right']['eef_pos'];obs['eef_mat']=Rotation.from_quat(proprio['right']['eef_quat_xyzw']).as_matrix()
        if self._spec.mobile:
            robot=self.env.robots[0];sim=self.env.sim
            center=robot.robot_model.base.correct_naming('center')
            pos=np.array(sim.data.get_site_xpos(center));rot=np.array(sim.data.get_site_xmat(center)).reshape(3,3)
            obs['base_pose']=dict(position=pos,rotation=rot)
            obs['robot0_base_pos']=pos;obs['robot0_base_quat']=Rotation.from_matrix(rot).as_quat()
            obs['robot0_base_to_eef_pos']=rot.T@(proprio['right']['eef_pos']-pos)
            hand=Rotation.from_quat(obs['robot0_eef_quat']).as_matrix()
            obs['robot0_base_to_eef_quat']=Rotation.from_matrix(rot.T@hand).as_quat()
        return obs

    def reset(self, **kwargs):
        # Legacy robosuite reset has no seed argument. Construction owns the seed.
        kwargs.pop('seed',None)
        self._gripper_targets = {arm:1. for arm in self._spec.arms}
        obs,info=normalize_reset(self._wrapped.reset(**kwargs))
        return self._enrich(obs),info

    def step(self, action):
        action=np.asarray(action,dtype=float)
        if action.shape!=(self.action_dim,) or not np.isfinite(action).all() or np.any(np.abs(action)>1):
            raise ValueError('Expected finite normalized action matching action_dim')
        native=self._low+(action+1)*.5*(self._high-self._low)
        layout=self.env.robots[0].composite_controller._action_split_indexes
        for arm in self._spec.arms:
            self._gripper_targets[arm]=(1-float(native[layout[arm+'_gripper'][0]]))/2
        result=normalize_step(self._wrapped.step(native))
        return self._enrich(result.observation),result.reward,result.terminated,result.truncated,result.info

    def close(self):
        try:self._sensor.close()
        finally:self._wrapped.close()


class NativeReplayEnv(EmbodimentEnv):
    """RGB-D/native codecs around the same strict saved-instance validator."""
    def __init__(self, instance, *, cameras=True):
        from react_manip.instances.runtime import ReplayEnv
        wrapped=ReplayEnv(instance,cameras=False,native_actions=True)
        try:
            super().__init__(instance['task_id'],instance['robot_id'],cameras=cameras,wrapped=wrapped)
        except BaseException:
            wrapped.close();raise

    def bind(self, episode):
        self._wrapped.bind(episode)
        self.env=self._wrapped.env
