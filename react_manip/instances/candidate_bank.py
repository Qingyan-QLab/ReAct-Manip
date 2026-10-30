"""Validate and draw an explicitly supplied external candidate bank."""
import json
from pathlib import Path
import shutil
from .records import digest, load_suite, save_record, seal
from .suite import PROTOCOL
ALGORITHM = 'sha256-rejection-v1'

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def checked_path(parent, name):
    path = (parent / name).resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError('file reference escapes bank directory')
    return path

def choices(groups):
    result = {}
    for instance, episodes in groups:
        result.setdefault(instance['task_id'], []).append((instance, episodes))
    for task, items in result.items():
        if len(items) != 3 or len({digest(i['scenario']['parameters']) for i, _ in items}) != 3:
            raise ValueError(f'{task}: requires three distinct resolved parameter sets')
        items.sort(key=lambda x: x[0]['task_instance_id'])
    return result

def load_bank(path):
    path = Path(path).resolve(); bank = json.loads(path.read_text())
    if set(bank) != {'schema_version','manifest','manifest_digest','selection','content_digest'}:
        raise ValueError('invalid bank fields')
    if bank['schema_version'] != 'cl-manip.candidate-bank.v1' or bank['selection'] != {
        'algorithm': ALGORITHM, 'candidates_per_task': 3, 'draws_per_task': 3,
        'policy_state_between_tasks': 'fresh_policy'}:
        raise ValueError('unsupported bank protocol')
    if bank['content_digest'] != digest({k:v for k,v in bank.items() if k!='content_digest'}):
        raise ValueError('bank digest mismatch')
    manifest_path = checked_path(path.parent, bank['manifest'])
    manifest, groups = load_suite(manifest_path)
    if manifest['content_digest'] != bank['manifest_digest'] or manifest['protocol'] != PROTOCOL:
        raise ValueError('bank manifest/protocol mismatch')
    return bank, manifest_path, manifest, choices(groups)

def choose_index(seed, task, count=3):
    if type(seed) is not int or seed < 0:
        raise ValueError('draw seed must be a nonnegative integer')
    limit = (1 << 256) - (1 << 256) % count
    counter = 0
    while True:
        value = int(digest([ALGORITHM, seed, task, counter]), 16)
        if value < limit:
            return value % count
        counter += 1

def copy_group(group, source, target):
    for ref in [group['task_instance'], *group['episodes']]:
        src = checked_path(source, ref['path']); dst = checked_path(target, ref['path'])
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise ValueError('duplicate output reference')
        shutil.copyfile(src, dst)

def draw(path, *, seed, out):
    bank, source_path, manifest, by_task = load_bank(path)
    if type(seed) is not int or seed < 0:
        raise ValueError('draw seed must be a nonnegative integer')
    selected=[]; records=[]
    for task in sorted(by_task):
        # The bank owns all three resolved instances for every task.  The draw
        # seed remains in provenance for reproducibility/CLI compatibility, but
        # it must not discard two instances: each task contributes 3*15=45
        # episodes to a complete candidate-bank evaluation.
        for instance, episodes in by_task[task]:
            group = next(g for g in manifest['groups'] if g['task_instance']['digest']==instance['content_digest'])
            selected.append(group)
            records.append(dict(task_id=task, task_instance_id=instance['task_instance_id'],
                                task_instance_digest=instance['content_digest'],
                                episode_digests=[e['content_digest'] for e in episodes]))
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    for group in selected:
        copy_group(group, source_path.parent, out)
    result=seal('manifest',suite_id=out.name,protocol=manifest['protocol'],groups=selected)
    save_record(result,out/'manifest.json')
    provenance=dict(schema_version='cl-manip.draw.v1', algorithm=ALGORITHM, draw_seed=seed,
                    draws_per_task=3,
                    bank_digest=bank['content_digest'], candidate_manifest_digest=manifest['content_digest'],
                    selected_manifest_digest=result['content_digest'], selected=records)
    provenance['content_digest']=digest(provenance)
    write_json(out/'selection.json',provenance)
    return out/'manifest.json', provenance

def select_task(path, *, task, seed, out):
    """Copy all candidate groups for one task into a standalone manifest."""
    bank, source_path, manifest, by_task = load_bank(path)
    if task not in by_task:
        raise ValueError(
            f'Unknown task {task!r}; available: {", ".join(sorted(by_task))}')
    if type(seed) is not int or seed < 0:
        raise ValueError('draw seed must be a nonnegative integer')
    selected = []
    records = []
    for instance, episodes in by_task[task]:
        group = next(
            group for group in manifest['groups']
            if group['task_instance']['digest'] == instance['content_digest'])
        selected.append(group)
        records.append(dict(
            task_id=task,
            task_instance_id=instance['task_instance_id'],
            task_instance_digest=instance['content_digest'],
            episode_digests=[episode['content_digest'] for episode in episodes],
        ))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for group in selected:
        copy_group(group, source_path.parent, out)
    result = seal('manifest', suite_id=out.name,
                  protocol=manifest['protocol'], groups=selected)
    save_record(result, out / 'manifest.json')
    provenance = dict(
        schema_version='cl-manip.draw.v1', algorithm=ALGORITHM, draw_seed=seed,
        draws_per_task=3, bank_digest=bank['content_digest'],
        candidate_manifest_digest=manifest['content_digest'],
        selected_manifest_digest=result['content_digest'], selected=records,
    )
    provenance['content_digest'] = digest(provenance)
    write_json(out / 'selection.json', provenance)
    return out / 'manifest.json'

def selection_for(path):
    """Validate optional draw provenance when replaying a saved selected manifest."""
    path=Path(path); sidecar=path.with_name('selection.json')
    if not sidecar.exists():
        return None
    record=json.loads(sidecar.read_text())
    if (record.get('schema_version')!='cl-manip.draw.v1' or record.get('algorithm')!=ALGORITHM
            or record.get('draws_per_task') != 3):
        raise ValueError('invalid selection provenance')
    if record.get('content_digest')!=digest({k:v for k,v in record.items() if k!='content_digest'}):
        raise ValueError('selection digest mismatch')
    manifest,groups=load_suite(path)
    expected=[dict(task_id=i['task_id'],task_instance_id=i['task_instance_id'],
                   task_instance_digest=i['content_digest'],episode_digests=[e['content_digest'] for e in es])
              for i,es in groups]
    if record.get('selected_manifest_digest')!=manifest['content_digest'] or record.get('selected')!=expected:
        raise ValueError('selection does not match execution manifest')
    return record
