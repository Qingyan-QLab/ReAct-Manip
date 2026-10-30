"""Physical invariants of the downloaded PiPER-X replacement bundle."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets/robots/agilex'


def test_eight_canonical_ids_are_independent():
    from react_manip.robots.catalog import normalize_robot_id, ALL_ROBOT_IDS
    assert set(ALL_ROBOT_IDS) == {'panda','sawyer','piperx','dual_piperx','mobile_dual_piperx',
                                  'legacy_piperx','aloha','aloha_mobile'}
    for key in ALL_ROBOT_IDS:
        assert normalize_robot_id(key) == key


@pytest.fixture(scope='module')
def manifest():
    from react_manip.robots.agilex_robosuite import validate_bundle
    validate_bundle()
    return json.loads((ASSETS/'provenance.json').read_text())


def test_official_model_identity_and_source_accuracy(manifest):
    assert manifest['schema'] == 'piperx-family.v1'
    source = ET.parse(ASSETS/'source/arm/piper_x/urdf/piper_x_description.urdf')
    compiled = ET.parse(ASSETS/'piperx.xml')
    for i in range(1,7):
        original = source.find(f"joint[@name='joint{i}']")
        joint = compiled.find(f".//joint[@name='right_joint{i}']")
        limit = original.find('limit')
        np.testing.assert_allclose(np.fromstring(joint.get('range'),sep=' '),
                                   [float(limit.get('lower')),float(limit.get('upper'))])
        np.testing.assert_allclose(np.fromstring(joint.get('axis'),sep=' '),
                                   np.fromstring(original.find('axis').get('xyz'),sep=' '))
    for name in ('gripper_link1','gripper_link2'):
        r=manifest['geometry'][name]
        assert r['sampled_surface_error_m'] <= .001
        assert r['convex_parts'] > 1


def test_compiled_gripper_visual_and_physical_surfaces_match(manifest):
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(ASSETS/'gripper.xml'))
    data=mujoco.MjData(model)
    for opening in (0.,.025,.05):
        data.qpos[:]=[opening,-opening]
        mujoco.mj_forward(model,data)
        for i in range(model.ngeom):
            name=model.geom(i).name
            assert not name.endswith(('_pad','_flat_pad'))
            if '_collision' not in name:continue
            j=model.geom(name.replace('_collision','_visual')).id
            assert model.geom_contype[i] and model.geom_conaffinity[i]
            assert not model.geom_contype[j] and not model.geom_conaffinity[j]
            assert model.geom_dataid[i]==model.geom_dataid[j]
            np.testing.assert_allclose(data.geom_xpos[i],data.geom_xpos[j],atol=1e-12)
            np.testing.assert_allclose(data.geom_xmat[i],data.geom_xmat[j],atol=1e-12)
            assert model.geom_margin[i]==0


def test_bundle_tampering_is_rejected(manifest,tmp_path,monkeypatch):
    import shutil
    from react_manip.robots.agilex_robosuite import validate_bundle
    # Copy XML only: the manifest must reject a missing mesh too.
    for name in ['provenance.json','piperx.xml','dual_piperx.xml','mobile_dual_piperx.xml','gripper.xml']:
        shutil.copy2(ASSETS/name,tmp_path/name)
    monkeypatch.setenv('AGILEX_ASSET_DIR',str(tmp_path))
    with pytest.raises(ValueError,match='Stale or modified'):
        validate_bundle()


@pytest.mark.parametrize('robot_id,arms',[('piperx',1),('dual_piperx',2),('mobile_dual_piperx',2)])
def test_models_compile_and_keep_all_contact_parts(robot_id,arms,manifest):
    from react_manip.robots.agilex_adapter import AgileXAdapter
    model=AgileXAdapter(robot_id).load_mujoco_model()
    names=[model.geom(i).name for i in range(model.ngeom)]
    for side in ('right','left')[:arms]:
        for parts in manifest['gripper_collision_names'].values():
            assert all(f'gripper0_{side}_{part}' in names for part in parts)
    assert all(not n.endswith(('_pad','_flat_pad')) for n in names if n)


@pytest.mark.parametrize('robot_id', ['aloha', 'aloha_mobile'])
def test_legacy_saved_states_are_not_silently_relabelled(robot_id):
    from react_manip.instances.records import VERSION, digest, validate_record
    # A restored robot ID does not make an incomplete historical record valid.
    record = dict(
        schema_version=VERSION, kind='task_instance', task_id='curling_push',
        task_instance_id='legacy_test', task_spec_digest='0' * 64,
        contract_digest='0' * 64, robot_id=robot_id, scenario={}, runtime={},
        provenance=dict(sampler=VERSION, seed=0),
    )
    record['content_digest'] = digest(record)
    with pytest.raises(ValueError,match='runtime: expected exactly'):
        validate_record(record)


def test_contact_starts_at_visible_finger_surface(manifest):
    """An independent sphere probe must touch where the rendered hull is."""
    import mujoco
    import trimesh
    root=ET.parse(ASSETS/'gripper.xml').getroot()
    for mesh in root.findall('asset/mesh'):
        mesh.set('file',str(ASSETS/mesh.get('file')))
    probe=ET.SubElement(root.find('worldbody'),'body',name='probe',mocap='true')
    ET.SubElement(probe,'geom',name='probe_geom',type='sphere',size='.001',margin='0')
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    data=mujoco.MjData(model)
    probe_id=model.geom('probe_geom').id
    for finger in ('finger7','finger8'):
        ids=[model.geom(name).id for name in manifest['gripper_collision_names'][finger]]
        meshes=[]
        for geom_id in ids:
            mesh_id=model.geom_dataid[geom_id]
            start=model.mesh_vertadr[mesh_id];count=model.mesh_vertnum[mesh_id]
            # Compiled vertices are transformed by geom pose, which includes
            # MuJoCo's mesh recentering. Read the actual compiled mesh.
            vertices=model.mesh_vert[start:start+count]
            first=model.mesh_faceadr[mesh_id];n=model.mesh_facenum[mesh_id]
            faces=model.mesh_face[first:first+n]
            meshes.append((geom_id,vertices.copy(),faces.copy()))
        model.geom_contype[:]=0;model.geom_conaffinity[:]=0
        model.geom_contype[ids+[probe_id]]=1;model.geom_conaffinity[ids+[probe_id]]=1
        for opening in (0.,.025,.05):
            data.qpos[:]=[opening,-opening]
            mujoco.mj_forward(model,data)
            body=model.body(finger).id
            rotation=data.xmat[body].reshape(3,3)
            origin=data.xpos[body]+rotation@np.array([0.,-.01,-.1])
            direction=rotation@np.array([0.,0,1.])
            parts=[trimesh.Trimesh(vertices@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g],faces,process=False)
                   for g,vertices,faces in meshes]
            shape=trimesh.util.concatenate(parts)
            hits,_,_=shape.ray.intersects_location([origin],[direction],multiple_hits=False)
            assert len(hits)==1
            for gap,contact_expected in ((.003,False),(-.0002,True)):
                data.mocap_pos[0]=hits[0]-direction*(.001+gap)
                mujoco.mj_forward(model,data)
                probe_contacts=[c for c in data.contact if probe_id in (c.geom1,c.geom2)]
                assert bool(probe_contacts)==contact_expected


def test_source_download_and_cache_are_checksum_locked(tmp_path,monkeypatch):
    import importlib.util
    import io
    spec=importlib.util.spec_from_file_location('piperx_builder',ROOT/'scripts/build_agilex_assets.py')
    builder_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder_module)
    key='arm/LICENSE'
    content=(ASSETS/'source'/key).read_bytes()
    monkeypatch.setattr(builder_module.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(content))
    builder=builder_module.Builder(tmp_path)
    downloaded=builder.fetch(key)
    assert downloaded.read_bytes()==content
    downloaded.write_bytes(b'modified upstream or corrupted cache')
    with pytest.raises(ValueError,match='checksum mismatch'):
        builder.fetch(key)
    downloaded.unlink()
    with pytest.raises(ValueError,match='checksum mismatch'):
        monkeypatch.setattr(builder_module.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(b'wrong asset'))
        builder.fetch(key)
    builder.offline=True
    with pytest.raises(FileNotFoundError,match='Offline source missing'):
        builder.fetch(key)


def test_task_contact_override_is_rejected(manifest):
    from react_manip import register_builtin_tasks
    from react_manip.core.registry import get_task_registry
    from react_manip.robots.agilex_robosuite import validate_compiled_grippers
    register_builtin_tasks()
    definition=next(d for d in get_task_registry().list() if d.spec.task_id=='box_insertion')
    env=definition.factory(seed=0,robot_id='piperx',has_offscreen_renderer=False)
    try:
        env.reset()
        raw=getattr(env,'env',env)
        gripper=raw.robots[0].gripper['right']
        name=gripper.naming_prefix+manifest['gripper_collision_names']['finger7'][0]
        geom_id=raw.sim.model.geom_name2id(name)
        raw.sim.model.geom_contype[geom_id]=0
        with pytest.raises(ValueError,match='contact surface'):
            validate_compiled_grippers(env)
    finally:
        env.close()
