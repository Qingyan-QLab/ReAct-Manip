"""Real RPent encoders over a deterministic transport; no model claim."""
import numpy as np
import pytest

pytest.importorskip('rpent.robots.components.pi05_vla_client')
from rpent.robots.components.pi05_vla_client import Pi05VLAClient
from robots.robotwin.vla_client import LingBotVLAClient
from robots.robocasa.vla_client import RoboCasaVLAClient
from integrations.rpent_embodiments.policy import NativePolicy
from test_rpent_embodiments import observation


class Transport:
    def __init__(self, kind):self.kind=kind;self.calls=[]
    def call(self, method, args=(), **kwargs):
        self.calls.append((method,args,kwargs))
        if method=='vla.get_modality_config':return {'hist_maxlen':2,'video_delta_indices':[-1,0]}
        if method=='vla.reset_session':return {}
        assert method=='vla.predict'
        if self.kind=='rldx':
            return {k:np.zeros((1,2,d)) for k,d in {
                'action.end_effector_position':3,'action.end_effector_rotation':3,
                'action.gripper_close':1,'action.base_motion':4,'action.control_mode':1}.items()}
        if self.kind=='lingbot':return {'action':np.zeros((2,16))}
        return np.zeros((1,2,7 if self.kind=='libero' else 14))


@pytest.mark.parametrize('robot,kind',[
    ('panda_libero','libero'),('dual_arx_x5_robodojo','robodojo'),
    ('aloha_agilex_robotwin','lingbot'),('panda_omron_robocasa','rldx')])
def test_real_clients_wire_payloads(robot,kind):
    transport=Transport(kind)
    if kind=='lingbot':
        client=LingBotVLAClient('unused',8000);client._client=transport
    elif kind=='rldx':client=RoboCasaVLAClient(transport)
    else:client=Pi05VLAClient(transport,embodiment=kind)
    policy=NativePolicy(dict(robot_id=robot,checkpoint_id='transport-test-only',weights_frozen=True,control_hz=20),client=client)
    obs=observation(robot)
    result=policy.predict(obs,'pick block')
    assert result.shape==(2,{'libero':7,'robodojo':14,'lingbot':16,'rldx':12}[kind])
    method,args,_=transport.calls[-1];payload=args[0]
    if kind in ('libero','robodojo'):
        assert payload['states'].shape==(1,8 if kind=='libero' else 14)
        assert payload['main_images'].shape==(1,256,256,3)
        assert payload['task_descriptions']==['pick block']
    elif kind=='lingbot':
        assert payload['observation.state'].shape==(16,)
        np.testing.assert_allclose(payload['observation.state'][3:7],[0,1,0,0])
    else:
        assert payload['state.base_rotation'].shape==(1,1,4)
        assert payload['video.robot0_agentview_left'].shape==(1,2,256,256,3)

@pytest.mark.parametrize('robot',['panda_libero','aloha_agilex_robotwin','dual_arx_x5_robodojo','panda_omron_robocasa'])
def test_native_policy_through_actual_rpent_toolkit(tmp_path,monkeypatch,robot):
    from types import SimpleNamespace
    import rpent.planner.base
    from integrations.harness_vla import rpent_session
    from integrations.rpent_embodiments.catalog import EMBODIMENTS
    from integrations.rpent_embodiments.control import decode_native,source_eef
    seen=[]
    class Backend:
        provenance=dict(weights_frozen=True,checkpoint_id='toolkit-test-only',action_schema='react-manip.rpent.v1')
        def start_episode(self):seen.append('start')
        def observe(self,obs):seen.append('observe')
        def predict(self,obs,prompt):
            spec=EMBODIMENTS[robot]
            if spec.source=='robotwin':return np.array([np.r_[source_eef(obs,'left'),1,source_eef(obs,'right'),1]])
            return np.zeros((1,{'libero':7,'robodojo':14,'robocasa':12}[spec.source]))
        def decode(self,value,obs):return decode_native(value,obs,EMBODIMENTS[robot].native_action)
    class Planner:
        def solve(self,*,toolkit,**kwargs):
            names={t.name for t in toolkit.list_tools()}
            assert ('move_base' in names)==EMBODIMENTS[robot].mobile
            result=toolkit.execute_tool('vla_act',{'prompt':'test','max_chunks':1})
            assert not result.is_error
            seen.append('inferred')
            return SimpleNamespace(finish_result={'status':'stuck'},stats={},messages=[],error=None)
    monkeypatch.setattr(rpent_session,'load_factory',lambda _:lambda _:Backend())
    monkeypatch.setattr(rpent.planner.base,'build_planner',lambda *a,**k:Planner())
    policy=rpent_session.build(dict(vla_factory='test:build',planner_model='test'),tmp_path)
    obs=observation(robot)
    obs['depth_m']={name:np.ones((256,256),dtype=np.float32) for name in obs['images']}
    obs['camera_calibration']={name:{'intrinsics':np.eye(3),'world_from_camera':np.eye(4)} for name in obs['images']}
    policy.set_action_dim(len(obs['control_contract']['native_low']))
    try:
        policy.begin_episode(phase='practice',episode_index=0)
        first=policy.act(obs,{'language_prompt':'test'})
        assert np.isfinite(first).all()
        policy.act(obs,{})
        assert seen==['start','observe','inferred']
        policy.end_episode(observation=obs,info={},truncated=True)
    finally:policy.shutdown()
