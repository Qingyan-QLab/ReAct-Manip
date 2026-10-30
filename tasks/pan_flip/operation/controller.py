"""Privileged action-only baseline. No simulator state or model writes."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class FlipMotion:
    """World-frame EEF offsets (m), pitch targets (rad), and durations (s)."""
    dip_x: float = 0.009957681472241327
    dip_z: float = -0.175937405291424
    dip_pitch: float = 0.4018947881396534
    toss_x: float = -0.08284060500419287
    toss_z: float = 0.34513979546510803
    toss_pitch: float = -1.063469165091196
    toss_duration: float = 0.41732162060737293
    catch_x: float = -0.17275868438259023
    catch_y: float = 0.0
    catch_z: float = -0.21488103818936632
    return_duration: float = 0.19579231296572366
    brake_x: float = .04
    brake_z: float = .08


class PanFlipController:
    requires_geometry_truth = True

    def __init__(self,env,*,motion=None):
        self.env=getattr(env,'env',env)
        self.motion=motion or FlipMotion(**self.env.embodiment.get('motion',{}))
        self.reset()

    def reset(self):
        self.started=float(self.env.sim.data.time)
        self.origin=self.env.pan_pos()
        self.orientation=self.env.eef_mat()
        self.pan_orientation=self.env.pan_mat()
        self.initial_offset=self.origin-self.env.eef_pos()
        self.stage='settle'
        self.layout=self.env.robots[0].composite_controller._action_split_indexes
        self.park={arm:(self.env.eef_pos(arm),self.env.eef_mat(arm))
                   for arm in self.env.robots[0].arms if arm != 'right'}

    def pose_action(self,reference_position,pitch):
        e=self.env
        rot=Rotation.from_rotvec([0,pitch,0]).as_matrix()
        desired=rot @ self.orientation
        offset=self.initial_offset
        target=np.asarray(reference_position)-offset
        translation=np.clip((target-e.eef_pos())/e.position_scale,-1,1)
        rotation=Rotation.from_matrix(desired @ e.eef_mat().T).as_rotvec()
        action=np.zeros(e.action_dim)
        start,end=self.layout['right'];action[start:end]=np.r_[translation,np.clip(rotation/e.rotation_scale,-1,1)]
        start,end=self.layout['right_gripper'];action[start:end]=1.
        for arm,(pos,mat) in self.park.items():
            delta=(pos-e.eef_pos(arm))/e.position_scale
            rot=Rotation.from_matrix(mat@e.eef_mat(arm).T).as_rotvec()/e.rotation_scale
            start,end=self.layout[arm];action[start:end]=np.clip(np.r_[delta,rot],-1,1)
            start,end=self.layout[arm+'_gripper'];action[start:end]=1.
        return action

    def act(self,observation=None,info=None):
        t=float(self.env.sim.data.time)-self.started; m=self.motion
        dip=np.array([m.dip_x,0,m.dip_z]); toss=dip+np.array([m.toss_x,0,m.toss_z])
        catch=np.array([m.catch_x,m.catch_y,m.catch_z])
        if t<.5:
            self.stage='settle'; p=np.zeros(3); pitch=0.
        elif t<1.0:
            self.stage='dip'; u=(t-.5)/.5; u=u*u*(3-2*u)
            p=dip*u; pitch=m.dip_pitch*u
        elif t<1.+m.toss_duration:
            self.stage='toss'; u=(t-1.)/m.toss_duration
            p=dip+(toss-dip)*u; pitch=m.dip_pitch+(m.toss_pitch-m.dip_pitch)*u
        elif t<1.+m.toss_duration+m.return_duration:
            self.stage='catch'; u=(t-1.-m.toss_duration)/m.return_duration
            p=toss+(catch-toss)*u; pitch=m.toss_pitch*(1-u)
        else:
            # Brake forward/up after the rim impulse to clear the robot mount.
            self.stage='hold'; p=catch+np.array([m.brake_x,0,m.brake_z]); pitch=0.
        return self.pose_action(self.origin+p,pitch)
