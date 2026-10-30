"""Strict JSON contracts: no sampling, simulator imports, or implicit overrides."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

VERSION = 'cl-manip.instances.v1'
SUPPORTED_TASKS = ('curling_push', 'button_indicator_matching', 'phone_lean', 'leveling_board1', 'rail_ball_to_cup', 'pitch_pot', 'conveyor_grasp', 'turntable_gamble1', 'ring_toss1', 'box_insertion', 'car_drive2', 'rope_hang', 'pan_flip', 'laptop_into_drawer', 'find_object_in_drawer2', 'tilte_maze')
ID = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$')


def digest(value):
    """SHA256 over UTF-8, sorted keys, compact JSON, finite numbers only."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _keys(value, names, where):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ValueError(f'{where}: expected exactly {names}')


def _id(value, where):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError(f'{where}: invalid identifier')


def _hash(value, where):
    if not isinstance(value, str) or not re.fullmatch('[a-f0-9]{64}', value):
        raise ValueError(f'{where}: expected SHA256')


def _int(value, where, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{where}: expected integer >= {minimum}')


def validate_parameters(parameters, contract):
    """Reject unowned fields, rule overrides, and out-of-domain resolved values."""
    def flatten(value, prefix=''):
        result = {}
        for key, item in value.items():
            name = prefix + key
            if isinstance(item, dict):
                result.update(flatten(item, name + '.'))
            else:
                result[name] = item
        return result
    if contract['task_id'] == 'leveling_board1':
        from tasks.leveling_board1.instances import validate_task_instance
        validate_task_instance(parameters, contract['sampling_profile'])
        return
    if contract['task_id'] == 'phone_lean':
        from tasks.phone_lean.instances import validate_task_instance
        validate_task_instance(parameters, contract['sampling_profile'])
        return
    if contract['task_id'] == 'ring_toss1':
        from tasks.ring_toss1.instances import validate_resolved
        parameters = validate_resolved(parameters, contract['sampling_profile'])
    if contract['task_id'] == 'turntable_gamble1':
        from tasks.turntable_gamble1.instances import validate_resolved
        parameters = validate_resolved(parameters, contract['sampling_profile'])
    if contract['task_id'] == 'conveyor_grasp':
        from tasks.conveyor_grasp.instances import validate_layout
        parameters = deepcopy(parameters)
        layout = parameters.pop('inventory', None)
        validate_layout(layout, parameters.get('task', {}).get('n_objects'), parameters.get('task', {}).get('category_count'))
    if contract['task_id'] == 'find_object_in_drawer2':
        from tasks.find_object_in_drawer2.instances import validate_inventory
        parameters = deepcopy(parameters)
        validate_inventory(parameters.pop('inventory', None))
    profile = contract['sampling_profile']
    fixed = flatten(profile.get('fixed', {}))
    domains = {}
    for group in ('geometry', 'physics', 'appearance', 'runtime'):
        domains.update(profile.get(group, {}))
    actual = flatten(parameters)
    extra = set()
    if contract['task_id'] == 'button_indicator_matching':
        extra = {'button_indicator_assignment', 'layout.seed', 'layout.button_xy', 'layout.indicator_xy'}
        n = parameters.get('button_count')
        _int(n, 'button_count', 3)
        assignment = parameters.get('button_indicator_assignment')
        if n > 6 or not isinstance(assignment, list) or any(type(x) is not int for x in assignment) or sorted(assignment) != list(range(n)):
            raise ValueError('instance assignment must be a permutation of button indices')
        _int(actual.get('layout.seed'), 'layout.seed')
        for name, bounds in (('button_xy', (-.065, .025)), ('indicator_xy', (.34, .46))):
            rows = actual.get('layout.' + name)
            if not isinstance(rows, list) or len(rows) != n:
                raise ValueError('resolved layout must contain N coordinate pairs')
            for i, row in enumerate(rows):
                center = -.48 + i * .96 / (n-1)
                if (not isinstance(row, list) or len(row) != 2 or any(type(v) not in (int, float) for v in row)
                        or not bounds[0] <= row[0] <= bounds[1] or abs(row[1]-center) > .045001):
                    raise ValueError('resolved layout outside stratified sampling domain')
    if set(actual) != set(fixed) | set(domains) | extra:
        raise ValueError('parameters contain missing or unowned fields')
    for name, value in fixed.items():
        if name not in domains and actual[name] != value:
            raise ValueError(f'{name}: cannot override frozen task parameter')
    for name, domain in domains.items():
        value = actual[name]
        kind = domain['distribution']
        if kind == 'choice':
            valid = value in domain['values']
        elif kind in ('uniform', 'exponential'):
            valid = type(value) in (int, float) and domain['low'] <= value <= domain['high']
        else:
            raise ValueError(f'unsupported pilot distribution: {kind}')
        if not valid:
            raise ValueError(f'{name}: outside frozen sampling domain')

    for rule in profile.get('constraints', []):
        if rule['type'] != 'not_equal' or len(rule['paths']) != 2:
            raise ValueError('unsupported instance constraint')
        left, right = rule['paths']
        if actual[left] == actual[right]:
            raise ValueError(f'{left} and {right} must differ')


def validate_record(record):
    if not isinstance(record, dict):
        raise ValueError('instance must be a JSON object')
    try:
        digest(record)
    except (TypeError, ValueError) as exc:
        raise ValueError('instance must contain finite JSON values') from exc
    kind = record.get('kind')
    common = 'schema_version kind content_digest '
    fields = {
        'task_instance': 'task_id task_instance_id task_spec_digest contract_digest robot_id scenario runtime provenance',
        'episode_instance': 'task_id task_instance_id task_instance_digest episode_id initial_state round_plan provenance',
        'manifest': 'suite_id protocol groups',
    }
    if kind not in fields:
        raise ValueError('unknown instance kind')
    if kind == 'episode_instance' and record.get('task_id') in ('phone_lean','leveling_board1','rope_hang','laptop_into_drawer'):
        fields[kind] += ' scene'
    if kind == 'episode_instance' and ('variation' in record or 'runtime' in record):
        from .variation import TASKS
        if record.get('task_id') not in TASKS:
            raise ValueError('this task does not permit Episode model variation')
        fields[kind] += ' variation runtime'
    _keys(record, common + fields[kind], kind)
    if record['schema_version'] != VERSION:
        raise ValueError('unsupported instance schema version')
    _hash(record['content_digest'], 'content_digest')
    if digest({k: v for k, v in record.items() if k != 'content_digest'}) != record['content_digest']:
        raise ValueError('content digest mismatch')
    if kind != 'manifest':
        if record['task_id'] not in SUPPORTED_TASKS:
            raise ValueError('task has not passed the instance adapter gate')
        _id(record['task_instance_id'], 'task_instance_id')
        _keys(record['provenance'], 'sampler seed', 'provenance')
        if record['provenance']['sampler'] != VERSION:
            raise ValueError('unsupported sampler')
        _int(record['provenance']['seed'], 'seed')
    if kind == 'task_instance':
        _hash(record['task_spec_digest'], 'task_spec_digest')
        _hash(record['contract_digest'], 'contract_digest')
        # Validate the record shape before resolving the robot capability.
        # This keeps malformed legacy records diagnosable (and prevents an
        # unsupported robot label from masking a missing runtime contract).
        validate_runtime(record['runtime'])
        from .runtime import spec_for
        from react_manip.core.compatibility import resolve_task_robot
        resolve_task_robot(spec_for(record['task_id']), record['robot_id'])
        from react_manip.robots.sensors import policy_camera_ids
        if record['runtime']['cameras'] != list(policy_camera_ids(record['robot_id'])):
            raise ValueError('camera contract changed; regenerate instances with main and wrist views')
        if ('left' in record['runtime']['action_layout']) != (record['robot_id'] not in ('piperx', 'legacy_piperx', 'panda', 'sawyer', 'panda_libero', 'panda_omron_robocasa')):
            raise ValueError('action layout does not match the instance robot')
        s = record['scenario']
        _keys(s, 'schema_version task_id scenario_id seed parameters assets metadata digest', 'scenario')
        if s['task_id'] != record['task_id'] or not isinstance(s['parameters'], dict):
            raise ValueError('scenario task/parameters mismatch')
        from react_manip.core.scenarios import Scenario
        scenario = Scenario(**{k: v for k, v in s.items() if k != 'digest'})
        if scenario.digest != s['digest']:
            raise ValueError('scenario digest mismatch')
    elif kind == 'episode_instance':
        if 'variation' in record:
            if not isinstance(record['variation'], dict):
                raise ValueError('Episode variation must be explicit')
            validate_runtime(record['runtime'])
        _id(record['episode_id'], 'episode_id')
        _hash(record['task_instance_digest'], 'task_instance_digest')
        state = record['initial_state']
        _keys(state, 'format state_spec integration_state', 'initial_state')
        if state['format'] != 'mujoco.integration.v1':
            raise ValueError('unsupported initial state format')
        _int(state['state_spec'], 'state_spec', 1)
        if not isinstance(state['integration_state'], list) or not state['integration_state'] or any(type(v) not in (int, float) for v in state['integration_state']):
            raise ValueError('integration_state must be a nonempty numeric list')
        plan = record['round_plan']
        if record['task_id'] in ('curling_push', 'phone_lean', 'leveling_board1', 'rail_ball_to_cup', 'pitch_pot', 'conveyor_grasp', 'turntable_gamble1', 'ring_toss1', 'box_insertion', 'car_drive2', 'rope_hang', 'pan_flip', 'laptop_into_drawer', 'tilte_maze'):
            if plan is not None:
                raise ValueError('this task has no rounds')
            if record['task_id'] in ('phone_lean','leveling_board1','rope_hang','laptop_into_drawer'):
                if not isinstance(record['scene'], dict): raise ValueError('task scene must be explicit')
        else:
            _keys(plan, 'advance_on exhaustion targets', 'round_plan')
            if plan['advance_on'] != 'instruction_completed' or plan['exhaustion'] != 'terminate':
                raise ValueError('unsupported round transition')
            if not isinstance(plan['targets'], list) or not plan['targets']:
                raise ValueError('round targets must be nonempty')
            for t in plan['targets']:
                _int(t, 'target')
            if any(a == b for a, b in zip(plan['targets'], plan['targets'][1:])):
                raise ValueError('adjacent targets must differ')
    else:
        _id(record['suite_id'], 'suite_id')
        expected = {'name': 'practice_exam', 'practice_episodes': 10, 'exam_episodes': 5,
                    'exam_adaptation': 'online', 'invalid': 'abort_group',
                    'clock': 'synchronous_simulation', 'score': 'mean_exam_task_score',
                    'policy_track': 'public'}
        if record['protocol'] != expected:
            raise ValueError('unsupported manifest protocol; historical frozen suites require their original source revision')
        if not isinstance(record['groups'], list) or not record['groups']:
            raise ValueError('manifest groups must be nonempty')
        for group in record['groups']:
            _keys(group, 'task_instance episodes', 'manifest group')
            if not isinstance(group['episodes'], list) or len(group['episodes']) != 15:
                raise ValueError('each group requires exactly 15 episodes')
            for ref in [group['task_instance'], *group['episodes']]:
                _keys(ref, 'path digest', 'file reference')
                _hash(ref['digest'], 'file reference digest')
                if not isinstance(ref['path'], str) or not ref['path'] or Path(ref['path']).is_absolute() or '..' in Path(ref['path']).parts:
                    raise ValueError('references must be suite-relative paths without ..')
    return record


def seal(kind, **fields):
    result = dict(schema_version=VERSION, kind=kind, **deepcopy(fields))
    result['content_digest'] = digest(result)
    return validate_record(result)


def save_record(record, path):
    validate_record(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite published/generated instances silently.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load_record(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result
    return validate_record(json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs))


def validate_pair(instance, episode):
    validate_record(instance); validate_record(episode)
    if instance['kind'] != 'task_instance' or episode['kind'] != 'episode_instance':
        raise ValueError('expected task instance and episode instance')
    if any(episode[k] != instance[k] for k in ('task_id', 'task_instance_id')) or episode['task_instance_digest'] != instance['content_digest']:
        raise ValueError('episode refers to a different task instance')
    parameters = instance['scenario']['parameters']
    if instance['scenario']['metadata'].get('episode_variation_version') == 1 and 'variation' not in episode:
        raise ValueError('this Task requires explicit Episode variation and runtime')
    if 'variation' in episode:
        from .variation import effective_parameters
        parameters = effective_parameters(instance['task_id'], parameters, episode['variation'])
        # Geometry/physics can change the model, never the robot/controller/runtime.
        for key, value in instance['runtime'].items():
            if key != 'model_digest' and episode['runtime'][key] != value:
                raise ValueError('Episode changed a frozen runtime field: '+key)
    if instance['task_id'] == 'laptop_into_drawer':
        from tasks.laptop_into_drawer.instances import validate_episode_instance
        validate_episode_instance(instance['scenario']['parameters'], episode['scene'])
    if instance['task_id'] == 'find_object_in_drawer2':
        if sorted(episode['round_plan']['targets']) != sorted(instance['scenario']['parameters']['inventory']['target_set']):
            raise ValueError('Episode targets must permute the fixed five-object set')
    if instance['task_id'] == 'rope_hang':
        from tasks.rope_hang.instances import validate_episode_instance
        validate_episode_instance(instance['scenario']['parameters'], episode['scene'])
    if instance['task_id'] == 'leveling_board1':
        from tasks.leveling_board1.instances import validate_episode_instance
        validate_episode_instance(parameters, episode['scene'])
    if instance['task_id'] == 'phone_lean':
        from tasks.phone_lean.instances import validate_episode_instance
        validate_episode_instance(parameters, episode['scene'])
    if instance['task_id'] == 'button_indicator_matching':
        n = instance['scenario']['parameters']['button_count']
        targets = episode['round_plan']['targets']
        # At most N awards per tick; one extra target for each transition.
        minimum = 5
        if len(targets) != minimum or any(t >= n for t in targets):
            raise ValueError('round plan must contain exactly five in-range targets')


def load_suite(path):
    path = Path(path).resolve()
    manifest = load_record(path)
    if manifest['kind'] != 'manifest':
        raise ValueError('expected manifest')
    def read(ref):
        target = (path.parent / ref['path']).resolve()
        if not target.is_relative_to(path.parent):
            raise ValueError('reference escapes suite directory')
        record = load_record(target)
        if record['content_digest'] != ref['digest']:
            raise ValueError('manifest file digest mismatch')
        return record
    groups, instance_ids, episode_ids = [], set(), set()
    for group in manifest['groups']:
        instance = read(group['task_instance'])
        episodes = [read(ref) for ref in group['episodes']]
        if instance['task_instance_id'] in instance_ids:
            raise ValueError('duplicate task instance ID')
        instance_ids.add(instance['task_instance_id'])
        for episode in episodes:
            validate_pair(instance, episode)
            if episode['episode_id'] in episode_ids:
                raise ValueError('duplicate episode ID')
            episode_ids.add(episode['episode_id'])
        groups.append((instance, episodes))
    return manifest, groups


def validate_runtime(runtime):
    _keys(runtime, 'python numpy mujoco robosuite source_digest control_hz horizon_sec physics_timestep model_digest action_layout controller_digest cameras', 'runtime')
    for name in ('source_digest', 'model_digest', 'controller_digest'):
        _hash(runtime[name], name)
    for name in ('control_hz', 'horizon_sec', 'physics_timestep'):
        value = runtime[name]
        if type(value) not in (int, float) or value <= 0:
            raise ValueError(f'{name} must be positive')
    layout = runtime['action_layout']
    if layout not in ({'right': [0, 6], 'right_gripper': [6, 7]},
                      {'right': [0, 6], 'left': [6, 12], 'right_gripper': [12, 13], 'left_gripper': [13, 14]}):
        raise ValueError('runtime requires a 7D single-arm or 14D dual-arm action layout')
    if (not isinstance(runtime['cameras'], list) or not runtime['cameras']
            or any(not isinstance(n, str) or not n for n in runtime['cameras'])
            or len(set(runtime['cameras'])) != len(runtime['cameras'])):
        raise ValueError('cameras must be a nonempty list of unique camera IDs')
