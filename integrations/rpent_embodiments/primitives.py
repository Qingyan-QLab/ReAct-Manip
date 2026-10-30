"""RPent primitives sharing the benchmark rendezvous, with native VLA decoding."""
import numpy as np
from scipy.spatial.transform import Rotation
from integrations.harness_vla.primitives import Primitives as BasePrimitives
from .control import pose_command, hold_native, normalize


class Primitives(BasePrimitives):
    def pose(self, arm):
        p=self.env.current.observation['proprio'][arm]
        return np.asarray(p['eef_pos']),Rotation.from_quat(p['eef_quat_xyzw']).as_matrix()

    def action(self, arm, delta):
        return pose_command(self.env.current.observation,arm,delta,grippers=self.grippers)

    def vla_act(self, prompt, *, max_chunks=4):
        if self.vla is None or not prompt.strip() or not 1<=max_chunks<=100:
            raise ValueError('A configured VLA, instruction and bounded chunk count are required')
        count=0
        for _ in range(max_chunks):
            actions=self.vla.predict(self.env.current.observation,prompt)
            for value in actions:
                obs=self.env.current.observation
                action=self.vla.decode(value,obs)
                _,_,term,trunc,_=self._step(action);count+=1
                for arm in self.grippers:
                    c=obs['control_contract'];start,_=c['action_layout'][arm+'_gripper']
                    low,high=c['native_low'][start],c['native_high'][start]
                    self.grippers[arm]=float(low+(action[start]+1)*.5*(high-low))
                if term or trunc:return dict(executed_steps=count,episode_ended=True)
        return dict(executed_steps=count,episode_ended=False)

    def move_base(self, velocity, *, torso=0., steps=10):
        c=self.contract
        if 'base' not in c['action_layout']:raise ValueError('This embodiment has no mobile base')
        v=np.asarray(velocity,dtype=float)
        if v.shape!=(3,) or not np.isfinite(v).all() or np.any(np.abs(v)>1) or not np.isfinite(torso) or abs(torso)>1 or not 1<=steps<=200:
            raise ValueError('Use normalized [vx,vy,yaw], torso in [-1,1], steps in [1,200]')
        for index in range(steps+1):
            obs=self.env.current.observation;a=hold_native(obs,self.grippers)
            s,e=c['action_layout']['base'];a[s:e]=v if index<steps else 0
            s,e=c['action_layout']['torso'];a[s:e]=torso if index<steps else 0
            a[c['base_mode_index']]=1
            _,_,term,trunc,_=self._step(normalize(a,c))
            if term or trunc:return dict(executed_steps=index+1,episode_ended=True)
        return dict(executed_steps=steps+1,episode_ended=False)
