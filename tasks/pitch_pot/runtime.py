"""Task-local entry point; pass a concrete Scenario to the unchanged factory."""
import json
from pathlib import Path
from react_manip.core.scenarios import Scenario, sample_scenario


def make_env(*, seed=0, scenario=None, profile=None, robot_id="panda", **kwargs):
    from react_manip.builtin import discover_task_definitions
    if isinstance(scenario, (str, Path)):
        scenario = Scenario.load(scenario)
    if scenario is None:
        path = Path(profile) if profile else Path(__file__).resolve().parents[2] / 'configs/pitch_pot/profile_default.json'
        scenario = sample_scenario(json.loads(path.read_text()), seed=seed)
    spec, factory = next(x for x in discover_task_definitions() if x[0].task_id == 'pitch_pot')
    return spec, factory(seed=scenario.seed, robot_id=robot_id, scenario=scenario, **kwargs)
