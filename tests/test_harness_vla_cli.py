import json
from types import SimpleNamespace
import pytest

from integrations.harness_vla.evaluation import evaluate_harness_vla as cli


def test_no_model_config_fails_before_outputs(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main(['--draw-seed', '42', '--out', str(tmp_path/'missing.json')])
    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())


def test_smoke_cannot_publish_score(tmp_path, monkeypatch):
    from react_manip.instances import suite
    def evaluate(path, factory, **kwargs):
        assert kwargs['cameras'] is True
        policy = factory()
        assert policy.policy_id == 'rgbd-bridge-diagnostic-zero'
        return dict(valid=True, protocol={}, protocol_score=100, groups=[dict(score=100, exam_scores=[100]*5)], episodes=[])
    monkeypatch.setattr(suite, 'evaluate', evaluate)
    out = tmp_path/'smoke.json'
    assert cli.main(['--manifest', str(tmp_path/'manifest.json'), '--smoke-steps', '1', '--out', str(out)]) == 0
    result = json.loads(out.read_text())
    assert result['protocol_score'] is result['rgbd_protocol_score'] is result['official_score'] is None
    assert result['groups'][0]['score'] is None
    assert result['mode'] == 'rgbd_bridge_smoke'
    assert result['extension']['schema_version'] == 'react-manip.rgbd.v1'
    assert result['protocol']['policy_track'] == 'public_rgbd_extension'


def test_trace_header_declares_rgbd_track(tmp_path):
    directory = tmp_path/'episode'
    directory.mkdir()
    path = directory/'trajectory.jsonl'
    path.write_text('{"type":"header","policy_track":"public"}\n{"type":"policy_input","tick":0}\n')
    cli.annotate_traces(tmp_path, provenance_digest='extension-hash')
    header, record = map(json.loads, path.read_text().splitlines())
    assert header['policy_track'] == 'public_rgbd_extension'
    assert header['extension_source_digest'] == 'extension-hash'
    assert record == dict(type='policy_input', tick=0)
