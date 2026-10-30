<p align="center"><img src="docs/imgs/react-title.png" alt="ReAct-Manip" width="800"></p>

[简体中文](README.zh-CN.md)

ReAct-Manip is a MuJoCo environment for testing how robot policies use feedback to adapt across manipulation episodes. It provides 16 task implementations, a common policy interface, task-default evaluation, and replay support for user-supplied manifests.

<p align="center"><img src="docs/imgs/react_demo.gif" alt="Preview of the 16 manipulation tasks" width="600"></p>

> **Public release scope:** This repository does not include the evaluation question bank, fixed benchmark instances, or reported baseline scores. Task-default runs and explicit external manifests are development evaluations; their results must not be presented as scores from a bundled official benchmark.

## Install

ReAct-Manip requires Python 3.10. For source development:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/build_agilex_assets.py
```

The asset builder downloads checksum-pinned upstream PiPER-X files on its first run. RGB evaluation needs an OpenGL backend; headless Linux systems generally need EGL or OSMesa.

Install the complete package from PyPI with `python -m pip install 'react-manip[assets,demo]'`, then run `react-manip-assets prepare` once to prepare the PiPER-X assets.

## Protocol and scoring

Each Task Instance fixes its task-level parameters. A fresh policy sees the practice and exam Episodes defined by the selected task protocol; it may retain state and adapt when the protocol permits. The next Task Instance starts with a new policy. An Episode's parameters stay fixed during that Episode, and an explicit replay manifest restores its saved initial state.

Four tasks vary physical conditions between Episodes:

| Task | Episode-level variation |
| --- | --- |
| `leveling_board1` | Shim dimensions and thin-axis orientation |
| `conveyor_grasp` | Belt speed and target sizes |
| `box_insertion` | Contact friction and block mass |
| `car_drive2` | Vehicle speed gain |

Task rules, sampling domains, scoring and instance contracts are defined by each task's `task_spec.json` and `instance_contract.json`. The public package does not bundle a question bank; any manifest or candidate bank must be supplied explicitly by the caller.

## Evaluate a policy

Put `my_policy.py` in the repository root or on the Python import path, then run an installed task-default evaluation:

```bash
react-manip-evaluate \
  --task box_insertion \
  --policy my_policy:build \
  --out outputs/box_result.json
```

From a source checkout, use `python scripts/evaluate_policy.py` with the same arguments. Use `--state-only-smoke` and a short `--horizon-sec` for infrastructure checks; without `--policy`, the evaluator uses random actions.

The CLI has no implicit data source. To replay data maintained outside this repository, pass an external manifest or candidate bank explicitly:

```bash
react-manip-evaluate \
  --manifest /path/to/suite/manifest.json \
  --policy my_policy:build \
  --out outputs/replay.json

react-manip-evaluate \
  --candidate-bank /path/to/external-bank.json \
  --draw-seed 42 \
  --policy my_policy:build \
  --out outputs/bank_result.json
```

External manifests and candidate banks are validated before execution. Output files and generated selection directories must be new paths. The evaluator enforces wall-clock limits; timeouts invalidate the affected result. Simulation time advances with control steps, not model inference time.

## Policy interface

This zero-action policy shows the required interface; it does not solve tasks:

```python
# my_policy.py
import numpy as np

class ModelPolicy:
    policy_id = "my-model"

    def set_action_dim(self, action_dim):
        self.action_dim = action_dim

    def reset(self, *, seed=None, task_spec=None):
        pass

    def begin_episode(self, *, phase, episode_index, seed, task_spec):
        pass  # Keep useful state across practice and exam Episodes.

    def act(self, observation, info):
        return np.zeros(self.action_dim, dtype=np.float32)

    def end_episode(self, **feedback):
        pass

def build():
    return ModelPolicy()
```

`build()` must return a fresh policy. Actions are finite vectors of length `action_dim` in `[-1, 1]`; invalid actions invalidate the group. Read instructions from `info["language_prompt"]` and policy images from `observation["images"]`. The recorded robot determines the action layout. Instance files and privileged geometry Features are not policy inputs; the public track rejects `requires_geometry_truth=True`.

## Tasks

| Task ID | Goal | Task ID | Goal |
| --- | --- | --- | --- |
| `curling_push` | Push a curling stone | `button_indicator_matching` | Match buttons and indicators |
| `phone_lean` | Lean a phone against a support | `leveling_board1` | Level a board with shims |
| `rail_ball_to_cup` | Roll a ball into a cup | `pitch_pot` | Toss into a pot |
| `conveyor_grasp` | Sort conveyor objects | `turntable_gamble1` | Push a turntable |
| `ring_toss1` | Toss rings | `box_insertion` | Insert a block |
| `car_drive2` | Drive with two joysticks | `rope_hang` | Rebalance a suspended chain |
| `pan_flip` | Flip food in a pan | `laptop_into_drawer` | Place a laptop in a drawer |
| `find_object_in_drawer2` | Find an object in drawers | `tilte_maze` | Control a tilting maze |

Each task's directory under [`tasks/`](tasks/) contains its rules, score implementation, and instance contract. Runtime language instructions are supplied to the policy at reset and after each step.

## Visualization and manual debugging

PiPER-X policies receive 640×480 RGB images through `observation["images"]`. Camera IDs are `robot0_head_camera` and `robot0_wrist_right`; dual-arm robots also provide `robot0_wrist_left`. The selected robot determines the action layout.

Add `--visualize` to show policy images and `--video-dir outputs/new_videos` to record them. A desktop display is required for the viewer; headless video needs an offscreen OpenGL backend. For manual control or a temporary unscored task-default scene:

```bash
react-manip-demo --task phone_lean --seed 0
# or, from a source checkout
python scripts/pygame_demo.py --task phone_lean --seed 0
```

These commands do not read or replay a bundled question bank and do not produce an official benchmark result.

## Files and reproducibility

- `react_manip/`: evaluator, policy interface, robot support, instance validation, and replay.
- `tasks/` and `configs/`: task implementations, contracts, and sampling profiles.
- `scripts/evaluate_policy.py`: task-default and explicit external-suite evaluation.
- `scripts/pygame_demo.py`: unscored task-default manual control.
- `scripts/build_agilex_assets.py`: PiPER-X asset builder.

The public repository intentionally omits the evaluation question bank and fixed benchmark instances. Saved instances and candidate banks must be maintained outside this repository and supplied explicitly when replay is needed. Legacy `benchmark.*` imports must be updated to `react_manip.*`.

## Geometry ground-truth Features (debugging only)

[`GeometryFeatureAdapter`](react_manip/core/geometry.py) reads simulator geometry for local diagnostics. It is a privileged, read-only channel: **never provide its output to a benchmark policy or use it for benchmark scoring**. Extract a new snapshot after each step and keep it separate from public observations. The schema has `entities`, `relations`, `task_state`, and `validity_mask`; available fields depend on the task. The task's final `info[score_key]`, not a Geometry Feature, is the scoring authority.

Adapters currently cover `box_insertion`, `laptop_into_drawer`, `find_object_in_drawer2`, `conveyor_grasp`, `phone_lean`, `pitch_pot`, and `button_indicator_matching`. Other tasks may have no adapter. See [`geometry.py`](react_manip/core/geometry.py) for registration and field definitions.

## Robot embodiments and asset updates

The active robots are `piperx`, `dual_piperx`, and `mobile_dual_piperx`, built from pinned official PiPER-X sources; see [`piperx_assets.lock.json`](scripts/piperx_assets.lock.json). Generated PiPER-X assets come from pinned upstream sources and are built locally rather than committed to this repository. Optional HarnessVLA and RPent embodiments are under `integrations/`; they remain public source integrations but are not included in the PyPI wheel.

## License

Project-authored code, records, documentation, and original assets use the [PolyForm Noncommercial License 1.0.0](LICENSE). This is a noncommercial, source-available license, not an OSI-approved open-source license. Third-party dependencies and assets retain their own licenses.
