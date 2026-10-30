"""Resolved variations must change physics without changing the question or leaking it."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from react_manip.core.scenarios import Scenario
from react_manip.instances.records import digest, seal, validate_pair
from react_manip.instances.runtime import _construct, runtime_identity, capture_state, spec_for, contract_for, ReplayEnv
from react_manip.instances.variation import TASKS, sample_variation, effective_parameters, episode_scenario

ROOT = Path(__file__).resolve().parents[1]

def baseline(task):
    from react_manip.core.scenarios import sample_scenario
    from react_manip.instances.records import VERSION
    spec = spec_for(task)
    profile = json.loads((ROOT/'configs'/task/'profile_default.json').read_text())
    if task == 'leveling_board1':
        from tasks.leveling_board1.instances import sample_task_instance
        scenario = Scenario('1.0', task, f'{task}_0', 0,
                            sample_task_instance(0))
    else:
        scenario = sample_scenario(profile, seed=0)
    if task == 'conveyor_grasp':
        from tasks.conveyor_grasp.instances import sample_layout
        parameters = scenario.parameters
        parameters['inventory'] = sample_layout(
            scenario.seed, parameters['task']['n_objects'],
            parameters['task']['category_count'],
            parameters.get('basket', {}).get('example_counts'))
    return dict(task_id=task, task_instance_id=f'{task}_source_default',
                robot_id=spec.supported_robots[0], scenario=scenario.to_dict(),
                provenance={'sampler': VERSION, 'seed': 0})

@pytest.mark.parametrize('task',TASKS)
def test_resolved_variations_are_unique_bounded_deterministic_and_private(task):
    base = baseline(task)['scenario']['parameters']; saved = deepcopy(base)
    values = [sample_variation(task,base,17,i) for i in range(15)]
    assert len({digest(v) for v in values}) == 15
    assert values == [sample_variation(task,base,17,i) for i in range(15)]
    assert base == saved
    for v in values:
        p = effective_parameters(task,base,v)
        with pytest.raises(ValueError):
            effective_parameters(task,base,dict(v, arbitrary_override=1))
        if task == 'leveling_board1':
            assert sorted(p['shim_dimensions'][n][a] for n,a in p['solution_axes'].items())==[.03,.04,.05]
        if task == 'box_insertion':
            assert p['block_density']==base['block_density']
            assert p['wood_block']==base['wood_block']
    if task in ('car_drive2','conveyor_grasp','box_insertion'):
        key,lo,hi={'car_drive2':('nominal_speed_m_s',.9,1.15),'conveyor_grasp':('belt_speed_m_s',.08,.2),'box_insertion':('mass_scale',.7,1.3)}[task]
        for start in (0,5,10):
            assert sorted(int(5*(v[key]-lo)/(hi-lo)) for v in values[start:start+5])==list(range(5))

@pytest.mark.parametrize('task', ['leveling_board1','conveyor_grasp','box_insertion','car_drive2'])
def test_real_models_and_exact_replay(task):
    old = baseline(task)
    scenario = Scenario(**{k:v for k,v in old['scenario'].items() if k!='digest'})
    env = _construct(task,scenario,old['robot_id'])
    try:
        env.reset()
        task_fields = {k:deepcopy(v) for k,v in old.items() if k not in ('kind','schema_version','content_digest')}
        task_fields.update(runtime=runtime_identity(env,task),task_spec_digest=digest(spec_for(task).to_dict()),contract_digest=digest(contract_for(task)))
        instance = seal('task_instance',**task_fields)
        if task=='box_insertion':
            bid=env.env.block_body_id
            original=(env.env.sim.model.body_mass[bid],env.env.sim.model.body_inertia[bid].copy(),env.env.sim.model.body_ipos[bid].copy())
    finally:env.close()
    records=[]
    for index in (0,14):
        variation=sample_variation(task,scenario.parameters,17,index)
        resolved=episode_scenario(scenario,variation); scene=None
        if task=='leveling_board1':
            from tasks.leveling_board1.instances import sample_episode_instance
            scene=sample_episode_instance(resolved.parameters,index)
        env=_construct(task,resolved,old['robot_id'],episode=scene)
        try:
            env.reset(); raw=env.env
            if task=='box_insertion':
                bid=raw.block_body_id; m=raw.sim.model; scale=variation['mass_scale']
                np.testing.assert_allclose(m.body_mass[bid], original[0]*scale,rtol=1e-10)
                np.testing.assert_allclose(m.body_inertia[bid],original[1]*scale,rtol=1e-9)
                np.testing.assert_allclose(m.body_ipos[bid],original[2],atol=1e-10)
                np.testing.assert_allclose(raw._block_patch_friction,[variation['sliding_friction'][k] for k in ('left','right')])
                from react_manip.core.observation import policy_observation,policy_info
                obs,info=env.reset()
                assert not set(raw.privileged_observation_keys)&set(policy_observation(obs,spec_for(task),env))
                info=env.step(np.zeros(env.action_dim))[-1]
                assert not set(raw.privileged_info_keys)&set(policy_info(info,SimpleNamespace(),env))
                env.reset()
            if task=='conveyor_grasp':
                from react_manip.category_objects import _variant_spec
                assert raw._v_belt==variation['belt_speed_m_s']
                for r,s in zip(raw.records,variation['target_size_scales']):
                    ext=_variant_spec(r['category'],r['variant'])['extent']*s
                    np.testing.assert_allclose(r['extent'],ext)
                    assert 2*min(ext[:2])<.08
                    assert np.linalg.norm(ext[:2])<.14
            if task=='car_drive2': assert raw._nominal_speed==variation['nominal_speed_m_s']
            ep=dict(task_id=task,task_instance_id=instance['task_instance_id'],task_instance_digest=instance['content_digest'],episode_id=f'test_{task}_{index}',initial_state=capture_state(env),round_plan=None,provenance=old['provenance'],variation=variation,runtime=runtime_identity(env,task))
            if scene is not None:ep['scene']=scene
            records.append(seal('episode_instance',**ep))
        finally:env.close()
    replay=ReplayEnv(instance,cameras=False)
    try:
        for record in records:
            replay.bind(record)
            for _ in range(2):
                replay.reset()
                assert capture_state(replay)==record['initial_state']
                replay.step(np.zeros(replay.action_dim))
        bad=deepcopy(records[0]); bad['runtime']['model_digest']='0'*64
        bad=seal('episode_instance',**{k:v for k,v in bad.items() if k not in ('schema_version','kind','content_digest')})
        with pytest.raises(ValueError,match='compiled model'):
            replay.bind(bad)
    finally:replay.close()


def test_leveling_useful_blocks_move_between_identities_and_physically_support():
    from tasks.leveling_board1.instances import sample_task_instance
    from tasks.leveling_board1.feasibility import certify_dimensions
    base=sample_task_instance(0)
    sets=set()
    for i in (0,4,14):
        p=effective_parameters('leveling_board1',base,sample_variation('leveling_board1',base,0,i))
        sets.add(tuple(sorted(p['solution_axes'])))
        assert certify_dimensions(p)['passed']
    assert len(sets)>1


def test_task_default_practice_exam_also_varies_car_gain():
    from react_manip.builtin import discover_task_definitions
    from react_manip.evaluation.protocols import EvaluationContext, load_protocol
    spec,factory=next((s,f) for s,f in discover_task_definitions() if s.task_id=='car_drive2')
    class Policy:
        def begin_episode(self,**kwargs): pass
        def end_episode(self,**kwargs): pass
        def set_action_dim(self,action_dim): self.dim=action_dim
        def act(self,observation,info): return np.zeros(self.dim)
    result=load_protocol(spec).run(EvaluationContext(spec=spec,factory=factory,policy_factory=Policy,
        runner_options={'horizon_sec':.05,'camera_observations':False}))
    assert result['valid']
    rows=result['episodes']
    assert len(rows)==15
    assert len({r['episode_variation']['nominal_speed_m_s'] for r in rows})==15
    assert sum(r['phase']=='practice' for r in rows)==10


@pytest.mark.parametrize('task',TASKS)
def test_declared_domains_and_protocol_match_sampler(task):
    from react_manip.instances.variation import DOMAINS
    assert contract_for(task)['episode_randomization']['domains']==DOMAINS[task]
    assert spec_for(task).metadata['episode_randomization']['domains']==DOMAINS[task]
    assert spec_for(task).evaluation['practice_episodes']==10
    assert spec_for(task).evaluation['exam_episodes']==5
