"""Native action semantics and the four public policy service contracts."""
import copy
import numpy as np
import pytest

from integrations.rpent_embodiments.control import decode_native,normalize,source_eef
from integrations.rpent_embodiments.policy import NativePolicy,encode_pi05,encode_lingbot,rldx_action


def observation(robot_id='dual_arx_x5_robodojo'):
    from integrations.rpent_embodiments.catalog import EMBODIMENTS,camera_ids
    spec=EMBODIMENTS[robot_id];layout={};cursor=0;arms={};proprio={}
    for arm in spec.arms:
        layout[arm]=[cursor,cursor+6];cursor+=6
        arms[arm]=dict(mode='joint_absolute' if spec.source=='robodojo' else 'osc_delta',
            input_ref_frame='world',world_from_control_rotation=np.eye(3),input_min=-np.ones(6),
            input_max=np.ones(6),output_min=-np.array([.05]*3+[.5]*3),output_max=np.array([.05]*3+[.5]*3))
        proprio[arm]=dict(qpos=np.zeros(spec.dof),eef_pos=np.array([.5,0,.8]),
            eef_quat_xyzw=np.array([0,0,0,1]),gripper_qpos=np.array([.02,-.02]),
            gripper_opening=.5,gripper_target_opening=.5,jacobian=np.eye(6,spec.dof))
    if spec.mobile:
        layout.update(torso=[cursor,cursor+1],base=[cursor+1,cursor+4]);cursor+=4
    for arm in spec.arms:layout[arm+'_gripper']=[cursor,cursor+1];cursor+=1
    if spec.mobile:cursor+=1
    lo=-np.ones(cursor);hi=np.ones(cursor)
    if spec.source=='robodojo':lo[:12]=-10;hi[:12]=10
    c=dict(schema_version='react-manip.rpent.v1',robot_id=robot_id,action_layout=layout,arms=arms,
           native_low=lo,native_high=hi,control_hz=20)
    if spec.mobile:c['base_mode_index']=cursor-1
    o=dict(embodiment=robot_id,control_contract=c,proprio=proprio,
        tcp_offsets={arm:[.12,0,0] for arm in spec.arms},
        images={name:np.zeros((256,256,3),dtype=np.uint8) for name in camera_ids(robot_id)},
        robot0_eef_quat=np.array([0,0,0,1]),robot0_gripper_qpos=np.array([.02,-.02]),
        robot0_base_pos=np.zeros(3),robot0_base_quat=np.array([0,0,0,1]),
        robot0_base_to_eef_pos=np.ones(3),robot0_base_to_eef_quat=np.array([0,0,0,1]))
    return o


def test_qpos14_routes_left_then_right_without_cartesian_conversion():
    obs=observation();q=np.r_[np.arange(6)/10,-np.arange(6)/10,0,1]
    action=decode_native(q,obs,'qpos14')
    np.testing.assert_allclose(action[:6],q[6:12]/10,atol=1e-15)
    np.testing.assert_allclose(action[6:12],q[:6]/10,atol=1e-15)
    np.testing.assert_allclose(action[-2:],[-1,1])
    with pytest.raises(ValueError):decode_native(np.r_[q[:12],1.1,0],obs,'qpos14')
    q[0]=11
    with pytest.raises(ValueError):decode_native(q,obs,'qpos14')


def test_eef16_wxyz_flange_offset_and_fresh_observation():
    obs=observation('aloha_agilex_robotwin')
    pose=source_eef(obs,'left')
    np.testing.assert_allclose(pose[3:],[0,1,0,0],atol=1e-12)
    wire=np.r_[pose,1,pose,0]
    action=decode_native(wire,obs,'eef16')
    np.testing.assert_allclose(action[:12],0,atol=1e-12)
    np.testing.assert_allclose(action[-2:],[1,-1])
    moved=copy.deepcopy(obs);moved['proprio']['left']['eef_pos'][0]-=.01
    corrected=decode_native(wire,moved,'eef16')
    assert corrected[6]>.1
    wire[3:7]=0
    with pytest.raises(ValueError):decode_native(wire,obs,'eef16')


def test_libero_state_and_image_direction():
    obs=observation('panda_libero');obs['images']['robot0_head_camera'][:,0]=255
    encoded=encode_pi05(obs,'pick','libero')
    assert encoded['main_images'][:,-1].min()==255
    assert encoded['main_images'][:,0].max()==0
    assert encoded['states'].shape==(8,)
    np.testing.assert_allclose(encoded['states'][-2:],[.02,-.02])


def test_rldx_named_layout_keeps_base_torso_and_mode():
    obs=observation('panda_omron_robocasa')
    a=rldx_action({'action.end_effector_position':[.1,.2,.3],
        'action.end_effector_rotation':[.4,.5,.6],'action.base_motion':[.7,.8,.9,-.2],
        'action.gripper_close':[.49],'action.control_mode':[.5]},obs)
    np.testing.assert_allclose(a,[.1,.2,.3,.4,.5,.6,-.2,.7,.8,.9,-1,1])
    assert decode_native(a,obs,'robocasa12').shape==(12,)


class FakePi05:
    def predict(self,obs):
        self.obs=obs;return np.zeros((3,7))


def test_backend_rejects_wrong_robot_frequency_and_unfrozen_weights():
    config=dict(robot_id='panda_libero',checkpoint_id='test-only',weights_frozen=True,control_hz=20)
    client=FakePi05();p=NativePolicy(config,client=client)
    assert p.predict(observation('panda_libero'),'pick').shape==(3,7)
    with pytest.raises(ValueError):p.predict(observation(),'pick')
    obs=observation('panda_libero');obs['control_contract']['control_hz']=50
    with pytest.raises(ValueError,match='control_hz'):p.predict(obs,'pick')
    with pytest.raises(ValueError):NativePolicy({**config,'weights_frozen':False},client=client)


def test_libero_source_bounds_precede_temporal_resampling():
    class Client:
        def predict(self, payload):
            return np.tile([1.2, -.2, 0, 0, 0, 0, -1.0014484], (4, 1))

    policy=NativePolicy(dict(robot_id='panda_libero',checkpoint_id='test-only',
        weights_frozen=True,control_hz=20,n_action_steps=4,
        temporal_adapter='libero_delta_resample_v1'),client=Client())
    obs=observation('panda_libero')
    actions=policy.predict(obs,'push')
    np.testing.assert_allclose(actions[0],[1,-.2,0,0,0,0,-1])
    decode_native(actions[0],obs,'libero_delta7')
    obs['control_contract']['control_hz']=50
    adapted=policy.predict(obs,'push')
    assert adapted.shape==(10,7)
    np.testing.assert_allclose(adapted.sum(axis=0),actions.sum(axis=0),atol=1e-12)
    assert policy.provenance['source_action_adapter']=='libero_clip_unit_v1'
    class InvalidClient:
        def predict(self, payload):return np.full((4,7),np.nan)
    policy.client=InvalidClient()
    with pytest.raises(ValueError,match='finite'):policy.predict(obs,'push')


class FakeRLDX:
    def __init__(self):self.resets=0;self.calls=[]
    def get_modality_config(self):return dict(video_delta_indices=[-2,0],hist_maxlen=3)
    def reset_session(self):self.resets+=1
    def predict(self,obs,options):
        self.calls.append((obs,options))
        return {key:np.zeros((1,2,dim)) for key,dim in {
            'action.end_effector_position':3,'action.end_effector_rotation':3,
            'action.gripper_close':1,'action.base_motion':4,'action.control_mode':1}.items()}


def test_rldx_history_tracks_steps_and_resets_between_episodes():
    client=FakeRLDX();p=NativePolicy(dict(robot_id='panda_omron_robocasa',checkpoint_id='test-only',
        weights_frozen=True,control_hz=20),client=client)
    obs=observation('panda_omron_robocasa');p.start_episode();p.predict(obs,'pick')
    for value in [10,20]:
        new=copy.deepcopy(obs)
        for im in new['images'].values():im[:]=value
        p.observe(new)
    p.predict(new,'pick')
    payload,options=client.calls[-1]
    frames=payload['video.robot0_agentview_left']
    assert frames.shape==(1,2,256,256,3)
    assert frames[0,0].max()==0 and frames[0,1].min()==20
    assert options=={'reset_memory':[False]}
    p.start_episode();p.predict(obs,'pick')
    assert client.calls[-1][0]['video.robot0_agentview_left'].max()==0
    assert client.calls[-1][1]=={'reset_memory':[True]}


def test_cli_requires_model_configuration_before_simulation(tmp_path):
    from integrations.harness_vla.evaluation.evaluate_rpent_embodiment import main
    with pytest.raises(SystemExit):main(['--robot','panda_libero','--task','box_insertion','--out',str(tmp_path/'x.json')])
    assert not list(tmp_path.iterdir())
