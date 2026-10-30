"""Camera mounts, public RGB selection, replay identity and multiview recording."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from react_manip.builtin import discover_task_definitions
from react_manip.robots.sensors import policy_camera_ids


def test_task_camera_declarations():
    for spec,_ in discover_task_definitions():
        cameras=policy_camera_ids(spec.supported_robots[0])
        assert spec.metadata['camera_contract']['policy_cameras']==list(cameras)
        assert spec.metadata['camera_contract']['version']=='3.0'
        assert len(cameras)==(2 if spec.supported_robots[0]=='piperx' else 3)
        assert not any('debug' in name or 'workspace' in name for name in cameras)


@pytest.fixture(params=['box_insertion','button_indicator_matching','laptop_into_drawer'])
def camera_env(request):
    spec,factory=next((s,f) for s,f in discover_task_definitions() if s.task_id==request.param)
    kwargs=dict(seed=0,has_offscreen_renderer=False)
    if request.param=='button_indicator_matching':
        from react_manip.core.scenarios import sample_scenario
        path=Path(__file__).resolve().parents[1]/'configs/button_indicator_matching/profile_default.json'
        kwargs['scenario']=sample_scenario(json.loads(path.read_text()),seed=0)
    env=factory(**kwargs)
    try:yield spec,env
    finally:env.close()


def test_mount_following_reset_and_replay_identity(camera_env):
    import mujoco
    from react_manip.instances.runtime import runtime_identity
    from react_manip.instances.records import validate_runtime
    spec,env=camera_env
    robot=spec.supported_robots[0];names=policy_camera_ids(robot)
    env.reset()
    m,d=env.sim.model._model,env.sim.data._data
    ids=[m.camera(n).id for n in names]
    assert all(m.cam_bodyid[i]!=0 for i in ids)
    for arm in ('right',) if robot=='piperx' else ('right','left'):
        c=m.camera('robot0_wrist_'+arm).id
        assert m.body(m.cam_bodyid[c]).name==f'gripper0_{arm}_eef'
        np.testing.assert_allclose(m.cam_pos[c],[-.1,0,-.08])
        assert m.cam_fovy[c]==65
        from scipy.spatial.transform import Rotation
        local_rotation=Rotation.from_quat(m.cam_quat[c][[1,2,3,0]]).as_matrix()
        target=np.array([0.,0.,.06])-m.cam_pos[c]
        np.testing.assert_allclose(-local_rotation[:,2],target/np.linalg.norm(target),atol=1e-10)
        # The camera is mounted above/rearward of the eef.  In its local
        # frame optical -Z points at the tool target and image-up (+Y) is the
        # local -X direction projected onto the image plane.
        view = target/np.linalg.norm(target)
        up = np.array([-1.,0.,0.])
        expected_up = up - np.dot(up, view)*view
        expected_up /= np.linalg.norm(expected_up)
        np.testing.assert_allclose(local_rotation[:,1],expected_up,atol=1e-10)
        np.testing.assert_allclose(-local_rotation[:,2],target/np.linalg.norm(target),atol=1e-10)
        np.testing.assert_allclose(d.cam_xmat[c].reshape(3,3),
                                  d.xmat[m.cam_bodyid[c]].reshape(3,3)@local_rotation,atol=1e-10)
        # Rendering names must share extrinsics; neither path adds a roll.
        alias=m.camera(f'gripper0_{arm}_eye_in_hand').id
        np.testing.assert_allclose(m.cam_pos[c],m.cam_pos[alias],atol=1e-10)
        np.testing.assert_allclose(m.cam_quat[c],m.cam_quat[alias],atol=1e-10)
    local=m.cam_pos[ids].copy();world=d.cam_xpos[ids].copy()
    j=m.joint('robot0_right_joint1').id
    d.qpos[m.jnt_qposadr[j]]+=.15;mujoco.mj_forward(m,d)
    assert np.linalg.norm(d.cam_xpos[ids[1]]-world[1])>.005
    np.testing.assert_allclose(d.cam_xpos[ids[0]],world[0],atol=1e-10)
    np.testing.assert_allclose(m.cam_pos[ids],local)
    if robot=='mobile_dual_piperx':
        before=d.cam_xpos[ids].copy();j=m.joint('robot0_mobile_free').id
        d.qpos[m.jnt_qposadr[j]]+=.1;mujoco.mj_forward(m,d)
        np.testing.assert_allclose(d.cam_xpos[ids]-before,np.tile([.1,0,0],(len(ids),1)),atol=1e-10)
    env.reset()
    m=env.sim.model._model
    assert len([m.camera(n).id for n in names])==len(names)
    runtime=runtime_identity(env,spec.task_id)
    assert runtime['cameras']==list(names)
    validate_runtime(runtime)


def test_runner_delivers_every_camera_and_can_disable_rgb(monkeypatch):
    from react_manip.core.runner import BenchmarkRunner
    spec,factory=next((s,f) for s,f in discover_task_definitions() if s.task_id=='box_insertion')
    env=factory(seed=0,has_offscreen_renderer=False)
    names=policy_camera_ids('piperx')
    monkeypatch.setattr(env,'get_camera_observation',lambda camera_ids,**kw:
                        {n:np.zeros((8,8,3),np.uint8) for n in camera_ids})
    class Spy:
        def set_action_dim(self,n):self.n=n
        def act(self,observation,**kw):
            self.observation=observation
            return np.zeros(self.n)
    try:
        policy=Spy();runner=BenchmarkRunner()
        result=runner.evaluate(spec,factory,policy,env=env,horizon_sec=.05)
        assert tuple(policy.observation['images'])==names
        assert result.metrics['camera_ids']==list(names)
        result=runner.evaluate(spec,factory,policy,env=env,horizon_sec=.05,camera_observations=False)
        assert 'images' not in policy.observation
        assert result.metrics['observation_mode']=='state_only_smoke'
    finally:env.close()


def test_viewer_records_all_raw_views_and_rejects_camera_changes(tmp_path,monkeypatch):
    imageio=pytest.importorskip('imageio.v2')
    from react_manip.evaluation import viewer as module
    monkeypatch.setattr(module,'check_dependencies',lambda *args:None)
    writers={}
    class Writer:
        def __init__(self):self.frames=[];self.closed=False
        def append_data(self,x):self.frames.append(x.copy())
        def close(self):self.closed=True
    def writer(path,**kwargs):
        writers[Path(path).name]=Writer();return writers[Path(path).name]
    monkeypatch.setattr(imageio,'get_writer',writer)
    names=policy_camera_ids('dual_piperx')
    frames={n:np.full((24,32,3),40+70*i,np.uint8) for i,n in enumerate(names)}
    with module.EvaluationViewer(video_dir=tmp_path/'video') as viewer:
        viewer.begin_episode('test','episode','exam',0,20)
        viewer.observe(None,{'images':frames},{})
        viewer.observe(None,{'images':frames},{})
        with pytest.raises(ValueError,match='changed during an episode'):
            viewer.observe(None,{'images':{names[0]:frames[names[0]]}},{})
    assert len(writers)==4 and all(w.closed and len(w.frames)==2 for w in writers.values())
    for name in names:np.testing.assert_array_equal(writers[f'episode.{name}.mp4'].frames[0],frames[name])
    metadata=json.loads((tmp_path/'video/episode.json').read_text())
    assert metadata['cameras']==list(names) and metadata['frames']==2
    assert writers['episode.mp4'].frames[0].shape==(480,960,3)


@pytest.mark.parametrize('robot',['piperx','dual_piperx','mobile_dual_piperx'])
def test_cameras_are_installed_without_task_scene_helpers(robot):
    from react_manip.robots.agilex_adapter import AgileXAdapter
    model=AgileXAdapter(robot).load_mujoco_model()
    ids=[model.camera(n).id for n in policy_camera_ids(robot)]
    assert len(ids)==len(set(ids))
    assert all(model.cam_bodyid[i]!=0 for i in ids)
