"""Native action codecs. All normalization uses the active evaluator contract."""
import numpy as np
from scipy.spatial.transform import Rotation


def normalize(native, contract):
    low, high = np.asarray(contract['native_low']), np.asarray(contract['native_high'])
    native = np.asarray(native, dtype=float)
    if native.shape != low.shape or not np.isfinite(native).all():
        raise ValueError('Native action has wrong shape or nonfinite values')
    if np.any(native < low-1e-8) or np.any(native > high+1e-8):
        raise ValueError('Native policy action exceeds the declared controller range')
    return np.clip(2*(native-low)/(high-low)-1,-1,1)


def hold_native(obs, grippers=None):
    c=obs['control_contract'];action=np.zeros(len(c['native_low']))
    for arm,config in c['arms'].items():
        start,end=c['action_layout'][arm]
        if config['mode']=='joint_absolute':action[start:end]=obs['proprio'][arm]['qpos']
        start,end=c['action_layout'][arm+'_gripper']
        default=0.
        if obs.get('embodiment') in ('aloha_agilex_robotwin','dual_arx_x5_robodojo'):
            default=1-2*obs['proprio'][arm]['gripper_target_opening']
        action[start:end]=(grippers or {}).get(arm,default)
    return action


def pose_command(obs, arm, delta, *, grippers=None):
    c=obs['control_contract'];config=c['arms'][arm];p=obs['proprio'][arm]
    action=hold_native(obs,grippers);start,end=c['action_layout'][arm]
    delta=np.asarray(delta,dtype=float)
    if delta.shape!=(6,) or not np.isfinite(delta).all():raise ValueError('Invalid Cartesian delta')
    if config['mode']=='joint_absolute':
        jac=np.asarray(p['jacobian']);dq=jac.T @ np.linalg.solve(jac@jac.T+1e-4*np.eye(6),delta)
        target=np.asarray(p['qpos'])+np.clip(dq,-.1,.1)
        action[start:end]=np.clip(target,np.asarray(c['native_low'])[start:end],np.asarray(c['native_high'])[start:end])
    else:
        rotation=np.asarray(config['world_from_control_rotation']) if config['input_ref_frame']=='base' else np.eye(3)
        d=np.r_[rotation.T@delta[:3],rotation.T@delta[3:]]
        lo,hi=np.asarray(config['output_min']),np.asarray(config['output_max'])
        action[start:end]=(np.clip(d,lo,hi)-lo)/(hi-lo)*(np.asarray(config['input_max'])-np.asarray(config['input_min']))+np.asarray(config['input_min'])
    return normalize(action,c)


def decode_native(value, obs, kind):
    c=obs['control_contract'];a=np.asarray(value,dtype=float)
    expected={'libero_delta7':7,'qpos14':14,'eef16':16,'robocasa12':12}[kind]
    if a.shape!=(expected,) or not np.isfinite(a).all():raise ValueError(f'{kind} requires finite [{expected}]')
    native=hold_native(obs)
    if kind=='libero_delta7':
        s,e=c['action_layout']['right'];native[s:e]=a[:6]
        s,e=c['action_layout']['right_gripper'];native[s:e]=a[6]
    elif kind=='qpos14':
        for arm,values,opening in [('left',a[:6],a[12]),('right',a[6:12],a[13])]:
            if c['arms'][arm]['mode']!='joint_absolute':raise ValueError('qpos14 requires native joint-position controllers')
            if not 0<=opening<=1:raise ValueError('Gripper opening must be in [0,1]')
            s,e=c['action_layout'][arm];native[s:e]=values
            s,e=c['action_layout'][arm+'_gripper'];native[s:e]=1-2*opening
    elif kind=='eef16':
        for arm,offset in [('left',0),('right',8)]:
            # RoboTwin wire quaternion order is WXYZ. Commands target the
            # source flange EEF, while robosuite controls the TCP grip site.
            q=a[offset+3:offset+7]
            if not np.isclose(np.linalg.norm(q),1,atol=.02):raise ValueError('EEF quaternion must be unit WXYZ')
            # RoboTwin config global_trans_matrix = diag(1,-1,-1).
            # EEF16 is the flange frame rotated 180 degrees about local X.
            target=Rotation.from_matrix(Rotation.from_quat(q[[1,2,3,0]]).as_matrix() @ np.diag([1,-1,-1]))
            tcp=a[offset:offset+3]+target.apply(np.asarray(obs['tcp_offsets'][arm]))
            p=obs['proprio'][arm];rotation=Rotation.from_quat(p['eef_quat_xyzw'])
            delta=np.r_[tcp-p['eef_pos'],(target*rotation.inv()).as_rotvec()]
            command=pose_command(obs,arm,delta)
            lo,hi=np.asarray(c['native_low']),np.asarray(c['native_high'])
            decoded=lo+(command+1)*.5*(hi-lo)
            s,e=c['action_layout'][arm];native[s:e]=decoded[s:e]
            opening=a[offset+7]
            if not 0<=opening<=1:raise ValueError('Gripper opening must be in [0,1]')
            s,e=c['action_layout'][arm+'_gripper'];native[s:e]=1-2*opening
    elif kind=='robocasa12':
        # The upstream key converter has already produced native layout.
        native=a
    return normalize(native,c)


def source_eef(obs, arm):
    p=obs['proprio'][arm];r=Rotation.from_quat(p['eef_quat_xyzw'])
    xyz=p['eef_pos']-r.apply(obs['tcp_offsets'][arm])
    wire_rotation=Rotation.from_matrix(r.as_matrix() @ np.diag([1,-1,-1]))
    return np.r_[xyz,wire_rotation.as_quat()[[3,0,1,2]]]
