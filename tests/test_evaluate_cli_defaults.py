"""The evaluation CLI requires an explicit task or external suite source."""
import pytest
from scripts import evaluate_policy as cli
from react_manip.core.timeouts import (
    USE_DEFAULT_TASK_WALL_TIMEOUT,
    default_task_wall_timeout_sec,
    resolve_task_wall_timeout_sec,
)
class EvaluationReached(Exception):
    pass


def test_no_implicit_candidate_bank(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(['--out', str(tmp_path / 'result.json')])
    assert exc.value.code == 2
    assert 'provide --task, --manifest or --candidate-bank' in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(('task_id', 'expected'), [
    ('box_insertion', 3600.0),
    ('conveyor_grasp', 7200.0),
    ('find_object_in_drawer2', 14400.0),
])
def test_release_task_wall_timeout_defaults(task_id, expected):
    assert default_task_wall_timeout_sec(task_id) == expected
    assert resolve_task_wall_timeout_sec(task_id, USE_DEFAULT_TASK_WALL_TIMEOUT) == expected


@pytest.mark.parametrize('task_id', [
    'box_insertion', 'conveyor_grasp', 'find_object_in_drawer2',
])
def test_explicit_task_wall_timeout_overrides_release_default(task_id):
    assert resolve_task_wall_timeout_sec(task_id, 123.0) == 123.0
    assert resolve_task_wall_timeout_sec(task_id, None) is None


@pytest.mark.parametrize('seed_args', [[], ['--draw-seed', '-1']])
def test_explicit_bank_requires_nonnegative_seed(seed_args, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(['--candidate-bank', str(tmp_path / 'external-bank.json'),
                  *seed_args, '--out', str(tmp_path / 'result.json')])
    assert exc.value.code == 2
    assert '--candidate-bank requires nonnegative --draw-seed' in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_explicit_task_reaches_task_registration(monkeypatch):
    def stop():
        raise EvaluationReached

    monkeypatch.setattr(cli, 'register_default_robots', stop)
    with pytest.raises(EvaluationReached):
        cli.main(['--task', 'box_insertion'])

@pytest.mark.parametrize('option,value', [
    ('--horizon-sec','0'), ('--horizon-sec','-1'), ('--horizon-sec','nan'),
    ('--horizon-sec','inf'), ('--control-hz','0'), ('--control-hz','nan'),
    ('--render-hold-sec','-1'), ('--render-hold-sec','inf'),
])
def test_invalid_timing_rejected_before_environment(option, value, monkeypatch):
    monkeypatch.setattr(cli,'register_default_robots',lambda: pytest.fail('environment setup reached'))
    with pytest.raises(SystemExit) as exc:
        cli.main(['--task','box_insertion',option,value])
    assert exc.value.code == 2


def test_mismatched_control_rate_rejected(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(['--task','box_insertion','--control-hz','10'])
    assert exc.value.code == 2
    assert 'must match the task physics rate' in capsys.readouterr().err


@pytest.mark.parametrize('task',['conveyor_grasp','car_drive2'])
def test_direct_task_practice_exam_reaches_real_runner(task,tmp_path):
    # Exercises robot capability checks and all 10/5 lifecycle transitions.
    out=tmp_path/(task+'.json')
    assert cli.main(['--task',task,'--state-only-smoke','--horizon-sec','.05','--out',str(out)])==0
    import json
    result=json.loads(out.read_text())
    assert result['valid'] and len(result['episodes'])==15
    assert all('error' not in row for row in result['episodes'])
    assert [r['phase'] for r in result['episodes']]==['practice']*10+['exam']*5
