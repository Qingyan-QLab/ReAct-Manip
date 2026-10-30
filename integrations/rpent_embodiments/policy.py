"""Adapters to RPent's existing Pi0.5, LingBot and RLDX service clients.

Only public images / proprioception enter these clients. Checkpoint identity
is supplied by the operator and recorded; it is not inferred from a hostname.
"""
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation

from .catalog import EMBODIMENTS
from .control import decode_native, source_eef, hold_native
from .environment import SCHEMA


def image(obs, name, *, libero=False):
    from PIL import Image
    a=np.asarray(obs['images'][name])
    if a.ndim!=3 or a.shape[-1]!=3 or a.dtype!=np.uint8:
        raise ValueError(f'{name} requires uint8 RGB')
    # Our renderer already reverses MuJoCo's bottom-up rows. RLinf LIBERO
    # reverses both raw axes, so only the horizontal reversal remains here.
    if libero:a=a[:,::-1]
    return np.asarray(Image.fromarray(a).resize((256,256),Image.Resampling.BICUBIC))


def encode_pi05(obs, prompt, source):
    p=obs['proprio']
    if source=='libero':
        # LIBERO uses hand-body quaternion, not the rotated gripper-site frame.
        q=np.asarray(obs['robot0_eef_quat'])
        return dict(main_images=image(obs,'robot0_head_camera',libero=True),
            wrist_images=image(obs,'robot0_wrist_right',libero=True),
            states=np.r_[p['right']['eef_pos'],Rotation.from_quat(q).as_rotvec(),p['right']['gripper_qpos']],
            task_descriptions=prompt)
    return dict(vision={dst:{'color':image(obs,src)} for dst,src in (
        ('cam_head','robot0_head_camera'),('cam_left_wrist','robot0_wrist_left'),
        ('cam_right_wrist','robot0_wrist_right'))},state={
            'left_arm_joint_state':p['left']['qpos'],'right_arm_joint_state':p['right']['qpos'],
            'left_ee_joint_state':np.array([p['left']['gripper_opening']]),
            'right_ee_joint_state':np.array([p['right']['gripper_opening']])},instruction=prompt)


def encode_lingbot(obs,prompt):
    return dict(views={dst:{'rgb':image(obs,src)} for dst,src in (
        ('head','robot0_head_camera'),('left_wrist','robot0_wrist_left'),('right_wrist','robot0_wrist_right'))},
        robot_state=dict(left_eef_pose=source_eef(obs,'left'),right_eef_pose=source_eef(obs,'right'),
            left_gripper=obs['proprio']['left']['gripper_target_opening'],right_gripper=obs['proprio']['right']['gripper_target_opening']),
        task_language=prompt)


def rldx_frame(obs):
    keys=('gripper_qpos','base_pos','base_quat','base_to_eef_pos','base_to_eef_quat')
    mapped=('gripper_qpos','base_position','base_rotation','end_effector_position_relative','end_effector_rotation_relative')
    result={'state.'+dst:np.asarray(obs['robot0_'+src],dtype=np.float32) for src,dst in zip(keys,mapped)}
    for name in ('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand'):
        result['video.'+name]=image(obs,name)
    return result


def rldx_action(values,obs):
    """RoboCasa PandaOmronKeyConverter semantics, assembled by named slices."""
    c=obs['control_contract'];a=hold_native(obs)
    data={'right':np.r_[values['action.end_effector_position'],values['action.end_effector_rotation']],
          'base':np.asarray(values['action.base_motion'])[:3],
          'torso':np.asarray(values['action.base_motion'])[3:4],
          'right_gripper':[-1. if float(np.asarray(values['action.gripper_close']).item())<.5 else 1.]}
    for key,value in data.items():
        start,end=c['action_layout'][key];a[start:end]=value
    a[c['base_mode_index']]=-1. if float(np.asarray(values['action.control_mode']).item())<.5 else 1.
    return a


class NativePolicy:
    def __init__(self, config, *, client=None):
        self.robot_id=config['robot_id'];self.spec=EMBODIMENTS[self.robot_id]
        if not config.get('checkpoint_id') or config.get('weights_frozen') is not True:
            raise ValueError('Declare checkpoint_id and weights_frozen=true')
        self.temporal_adapter=config.get('temporal_adapter','strict')
        if self.temporal_adapter not in ('strict','libero_delta_resample_v1'):raise ValueError('Unknown temporal adapter')
        if self.temporal_adapter!='strict' and self.spec.source!='libero':raise ValueError('Temporal adapter only supports LIBERO')
        self.control_hz=float(config.get('control_hz') or 0)
        if not np.isfinite(self.control_hz) or self.control_hz<=0:
            raise ValueError('Set control_hz to the checkpoint action execution frequency')
        self.provenance=dict(checkpoint_id=config['checkpoint_id'],weights_frozen=True,
            action_schema=SCHEMA,embodiment=self.robot_id,source=self.spec.source,
            native_action=self.spec.native_action,checkpoint_identity_source='operator_config')
        self._rpc=None;self._history=None;self._prompt=None;self._fresh=True
        self.n_action_steps=int(config.get('n_action_steps',8 if self.spec.policy=='rldx' else 50))
        if not 1<=self.n_action_steps<=256:raise ValueError('n_action_steps must be in [1,256]')
        self.provenance.update(control_hz=self.control_hz,n_action_steps=self.n_action_steps,
                               observation_schema='react-manip.rgbd.v1',temporal_adapter=self.temporal_adapter)
        if self.spec.source=='libero':
            self.provenance['source_action_adapter']='libero_clip_unit_v1'
        if client is not None:self.client=client
        elif self.spec.policy=='lingbot':
            from robots.robotwin.vla_client import LingBotVLAClient
            if not config.get('expected_metadata'):raise ValueError('LingBot requires expected_metadata from its pinned model config')
            self.client=LingBotVLAClient(config['host'],int(config['port']))
            self.client.validate_contract(config['expected_metadata'])
        else:
            from rpent.utils.rpc import make_rpc_client,wait_for_ready
            self._rpc=make_rpc_client(config['endpoint'],enable_sessions=self.spec.policy=='rldx')
            try:
                wait_for_ready(self._rpc,timeout_s=10)
                if self.spec.policy=='pi05':
                    from rpent.robots.components.pi05_vla_client import Pi05VLAClient
                    self.client=Pi05VLAClient(self._rpc,embodiment=self.spec.source)
                else:
                    from robots.robocasa.vla_client import RoboCasaVLAClient
                    self.client=RoboCasaVLAClient(self._rpc)
            except BaseException:
                self.close();raise
        if self.spec.policy=='rldx':
            try:
                mod=self.client.get_modality_config();self._indices=np.asarray(mod['video_delta_indices'],dtype=int)
                length=int(mod['hist_maxlen'])
                if (self._indices.ndim!=1 or not len(self._indices) or np.any(self._indices>0)
                        or length<1 or length>256 or np.any(self._indices<1-length)):
                    raise ValueError('Unsupported RLDX history configuration')
                self._history=deque(maxlen=length)
            except BaseException:
                self.close();raise

    def start_episode(self):
        self._prompt=None;self._fresh=True
        if self._history is not None:
            self._history.clear();self.client.reset_session()

    def observe(self,obs):
        if self._history is not None:self._history.append(rldx_frame(obs))

    def predict(self,obs,prompt):
        if obs.get('embodiment')!=self.robot_id:raise ValueError('VLA and environment embodiment mismatch')
        if self.temporal_adapter=='strict' and not np.isclose(obs['control_contract']['control_hz'],self.control_hz):
            raise ValueError('Checkpoint control_hz does not match task; explicit temporal adaptation is required')
        if not prompt.strip():raise ValueError('Instruction is empty')
        if self.spec.policy=='pi05':result=self.client.predict(encode_pi05(obs,prompt,self.spec.source))
        elif self.spec.policy=='lingbot':result=self.client.infer(encode_lingbot(obs,prompt))
        else:
            if prompt!=self._prompt:
                self.client.reset_session();self._history.clear();self._fresh=True
            self._prompt=prompt
            if not self._history:
                frame=rldx_frame(obs)
                self._history.extend([frame]*self._history.maxlen)
            frames=list(self._history);last=frames[-1];payload={}
            for key,value in last.items():
                if key.startswith('video.'):
                    payload[key]=np.stack([frames[max(0,len(frames)+int(d)-1)][key] for d in self._indices])[None]
                else:payload[key]=np.asarray(value)[None,None]
            payload['annotation.human.task_description']=[prompt]
            result=self.client.predict(payload,{'reset_memory':[self._fresh]});self._fresh=False
            shapes={'action.end_effector_position':3,'action.end_effector_rotation':3,
                    'action.gripper_close':1,'action.base_motion':4,'action.control_mode':1}
            arrays={key:np.asarray(result[key]) for key in shapes}
            horizon=None
            for key,a in arrays.items():
                if a.ndim!=3 or a.shape[0]!=1 or a.shape[2]!=shapes[key] or not np.isfinite(a).all():
                    raise ValueError(f'Invalid RLDX output: {key}')
                if horizon is None:horizon=a.shape[1]
                if horizon!=a.shape[1]:raise ValueError('RLDX action horizons disagree')
            result=np.array([rldx_action({k:a[0,i] for k,a in arrays.items()},obs) for i in range(horizon)])
        actions=np.asarray(result,dtype=float)
        dim={'libero_delta7':7,'eef16':16,'qpos14':14,'robocasa12':12}[self.spec.native_action]
        if actions.ndim!=2 or not 1<=len(actions)<=1024 or actions.shape[1]!=dim or not np.isfinite(actions).all():
            raise ValueError(f'Expected nonempty finite native [chunk,{dim}] actions')
        actions=actions[:self.n_action_steps]
        if self.spec.source=='libero':
            # LIBERO's OSC clips unit arm inputs; bounding the gripper keeps
            # its sign unchanged. Apply source bounds before temporal resampling.
            actions=np.clip(actions,-1.,1.)
        if self.temporal_adapter=='libero_delta_resample_v1':
            from .temporal import resample_libero
            actions=resample_libero(actions,self.control_hz,obs['control_contract']['control_hz'])
        return actions

    def decode(self,value,obs):return decode_native(value,obs,self.spec.native_action)

    def close(self):
        if self._rpc is not None:
            self._rpc.close();self._rpc=None


def build(config):return NativePolicy(config)
