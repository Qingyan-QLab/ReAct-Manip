"""Opt-in MuJoCo integration checks: RUN_RPENT_SIM_TESTS=1 (assets required)."""
import os
import numpy as np
import pytest

pytestmark=pytest.mark.skipif(os.environ.get('RUN_RPENT_SIM_TESTS')!='1',reason='Opt-in native simulation tests')
from integrations.rpent_embodiments.catalog import EMBODIMENTS,SUPPORTED_TASKS,camera_ids
from integrations.rpent_embodiments.environment import EmbodimentEnv
from integrations.rpent_embodiments.control import hold_native,normalize,decode_native,source_eef
from integrations.rpent_embodiments.policy import encode_pi05,encode_lingbot,rldx_frame


@pytest.mark.parametrize('robot,task',[(r,t) for r,ts in SUPPORTED_TASKS.items() for t in ts])
def test_supported_task_construction_reset_and_control(robot,task):
    env=EmbodimentEnv(task,robot,cameras=False)
    try:
        obs,_=env.reset()
        for _ in range(3):obs,*_=env.step(normalize(hold_native(obs),obs['control_contract']))
        assert np.isfinite(env.env.sim.data.qpos).all()
        assert obs['control_contract']['robot_id']==robot
        assert len(obs['proprio'])==len(EMBODIMENTS[robot].arms)
    finally:env.close()


@pytest.mark.skipif(os.environ.get('RUN_RPENT_RGBD_TESTS')!='1',reason='Requires OpenGL context')
@pytest.mark.parametrize('robot',EMBODIMENTS)
def test_rgbd_native_action_execution_and_mobile_motion(robot):
    env=EmbodimentEnv('box_insertion',robot)
    try:
        obs,_=env.reset();spec=EMBODIMENTS[robot]
        assert set(obs['images'])==set(camera_ids(robot))
        for name in camera_ids(robot):
            assert obs['images'][name].shape==(256,256,3)
            assert obs['depth_valid'][name].any()
            assert obs['camera_calibration'][name]['intrinsics'].shape==(3,3)
            if len(spec.arms)==2 and 'wrist' in name:
                # Source home flanges point +X. The optical axis must look
                # forward/down at the workspace, not upward toward the wall.
                optical=obs['camera_calibration'][name]['world_from_camera'][:3,2]
                assert optical[0]>.5 and optical[2]<-.2
        before=obs['proprio']['right']['qpos'].copy()
        base=obs.get('robot0_base_pos',np.zeros(3)).copy()
        if spec.policy=='pi05':
            encode_pi05(obs,'test',spec.source)
        elif spec.policy=='lingbot':encode_lingbot(obs,'test')
        else:assert rldx_frame(obs)['state.end_effector_rotation_relative'].shape==(4,)
        for _ in range(12):
            if spec.source=='libero':native=np.array([.1,0,0,0,0,0,-1])
            elif spec.source=='robodojo':
                native=np.r_[obs['proprio']['left']['qpos'],before+[.08,0,0,0,0,0],1,1]
            elif spec.source=='robotwin':
                left=source_eef(obs,'left');right=source_eef(obs,'right');right[0]+=.005
                native=np.r_[left,1,right,1]
            else:
                native=hold_native(obs);s,e=obs['control_contract']['action_layout']['base']
                native[s:e]=[.1,0,0];native[-1]=1
            obs,*_=env.step(decode_native(native,obs,spec.native_action))
        assert np.isfinite(env.env.sim.data.qpos).all()
        if spec.mobile:assert np.linalg.norm(obs['robot0_base_pos']-base)>.001
        else:assert np.linalg.norm(obs['proprio']['right']['qpos']-before)>.001
    finally:env.close()


def test_native_frozen_replay_accepts_x5_absolute_joints_and_rebinds():
    from pathlib import Path
    from integrations.rpent_embodiments.environment import NativeReplayEnv
    from react_manip.instances.candidate_bank import load_bank
    root=Path(__file__).resolve().parents[1]
    _,_,_,groups=load_bank(root/'react_manip/suites/rpent_candidates_v1/bank.json')
    instance,episodes=groups['button_indicator_matching'][0]
    env=NativeReplayEnv(instance,cameras=False)
    try:
        for episode in (episodes[0],episodes[10]):
            env.bind(episode)
            obs,_=env.reset()
            assert env.env is env._wrapped.env
            action=hold_native(obs)
            start,_=obs['control_contract']['action_layout']['right']
            action[start]=1.1  # Valid absolute radians, outside legacy normalized bounds.
            obs,*_=env.step(normalize(action,obs['control_contract']))
            assert np.isfinite(env.env.sim.data.qpos).all()
        with pytest.raises(ValueError,match='bounds'):
            env._wrapped.step(np.full(env.action_dim,100.))
    finally:env.close()
