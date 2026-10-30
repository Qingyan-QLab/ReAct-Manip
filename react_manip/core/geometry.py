"""Read-only geometry truth adapters for CL_Coach data collection.

The adapters deliberately live outside the policy observation path.  They read
the compiled MuJoCo model/data and task-owned read-only properties; no qpos or
qvel values are changed.  A task can register a specialised adapter without
changing the benchmark runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np


def _pose(env: Any, body_id: int) -> dict[str, np.ndarray]:
    d = env.sim.data
    return {
        "position": np.asarray(d.body_xpos[body_id], dtype=float).copy(),
        "quaternion": np.asarray(d.body_xquat[body_id], dtype=float).copy(),
    }


def _eef_pose(env: Any) -> dict[str, np.ndarray] | None:
    try:
        return {"position": np.asarray(env.eef_pos(), float).copy(), "quaternion": np.asarray(env.eef_quat(), float).copy()}
    except Exception:
        return None


@dataclass
class GeometryFeatureAdapter:
    """Base adapter and task registry.

    ``for_task(env)`` selects an adapter from the task module name.  The
    returned mapping is intentionally stable at the top level: entities,
    relations, task_state, validity_mask and schema_version.
    """

    env: Any
    schema_version: ClassVar[str] = "geometry.v1"
    _registry: ClassVar[dict[str, type["GeometryFeatureAdapter"]]] = {}

    @classmethod
    def register(cls, task_id: str):
        def deco(adapter: type["GeometryFeatureAdapter"]):
            cls._registry[task_id] = adapter
            return adapter

        return deco

    @classmethod
    def for_task(cls, env: Any) -> "GeometryFeatureAdapter":
        task_id = getattr(env, "task_id", None) or env.__class__.__module__.split(".")[-2]
        adapter = cls._registry.get(str(task_id))
        # Be convenient for callers that keep the benchmark's legacy wrapper.
        if adapter is None and hasattr(env, "env"):
            env = env.env
            task_id = getattr(env, "task_id", None) or env.__class__.__module__.split(".")[-2]
            adapter = cls._registry.get(str(task_id))
        if adapter is None:
            raise KeyError(f"no geometry adapter registered for task {task_id!r}")
        return adapter(env)

    def extract(self) -> dict[str, Any]:  # pragma: no cover - abstract contract
        raise NotImplementedError

    def _result(self, entities=None, relations=None, task_state=None, validity_mask=None):
        entities = entities or {}
        relations = relations or {}
        task_state = task_state or {}
        if validity_mask is None:
            validity_mask = {name: True for name in entities}
        return {
            "schema_version": self.schema_version,
            "entities": entities,
            "relations": relations,
            "task_state": task_state,
            "validity_mask": validity_mask,
        }


@GeometryFeatureAdapter.register("box_insertion")
class BoxInsertionGeometry(GeometryFeatureAdapter):
    """Geometric truth for the passive block / active EEF insertion task."""

    def extract(self):
        e = self.env
        block_pose = e._block_pose()
        eef_pos = np.asarray(e.sim.data.site_xpos[e.eef_site_id], dtype=float).copy()
        entities = {
            "block": {"pose": {"position": block_pose, "quaternion": np.asarray(e.sim.data.xquat[e.block_body_id], dtype=float).copy()}, "velocity": e._block_velocity(), "width_m": np.asarray([e._block_width_m], dtype=float), "patch_friction": e._block_patch_friction.copy(), "patch_width_m": e._block_patch_width_m.copy(), "friction_split_y_m": np.asarray([e._block_friction_split_y_m], dtype=float), "friction_split_from_left_m": np.asarray([e._block_friction_split_from_left_m], dtype=float)},
            "box": {"entry_x": np.asarray([e._slot_bounds[0]], dtype=float), "bounds_xy": np.asarray(e._slot_bounds, dtype=float), "opening_width_m": np.asarray([e._slot_opening_width_m], dtype=float), "clearance_m": np.asarray([e._slot_clearance_m], dtype=float), "target_depth": np.asarray([e._TARGET_DEPTH], dtype=float)},
            "table": {"top_z": np.asarray([0.78], dtype=float)},
            "eef": {"pose": {"position": eef_pos, "matrix": np.asarray(e.sim.data.site_xmat[e.eef_site_id], dtype=float).copy()}},
        }
        relations = {
            "eef_to_push_face": np.asarray(block_pose - eef_pos + np.array([-e._block_half_length, 0, 0]), dtype=float),
            "remaining_insertion_distance": np.asarray([max(0.0, e._TARGET_DEPTH - e._block_depth())], dtype=float),
            "contact_force_N": np.asarray([e._contact_force()], dtype=float),
        }
        state = {"insertion_depth_m": float(e._block_depth()), "block_contained": bool(e._block_contained()), "bottom_reached": bool(e._bottom_reached()), "success_blockers": e._success_blockers(), "success": bool(e._events), "failure": e._failure_reason}
        return self._result(entities, relations, state, {"block": True, "box": True, "table": True, "eef": True})


@GeometryFeatureAdapter.register("laptop_into_drawer")
class LaptopIntoDrawerGeometry(GeometryFeatureAdapter):
    def extract(self):
        e = self.env
        entities = {"laptop": {"pose": _pose(e, e.laptop_body_id)}, "handle": {"pose": {"position": np.asarray(e.sim.data.site_xpos[e.handle_site_id], float).copy()}}}
        entities["drawer"] = {"slide_position": np.asarray([e.sim.data.qpos[e.drawer_qpos_addr]], float)}
        if hasattr(e, 'geometry'):
            entities['laptop']['size'] = np.asarray(e.laptop_size,float).copy()
            entities['drawer'].update(inner_size=np.array([e.geometry.depth,e.geometry.width,e.geometry.height]),
                                     max_open_m=e.geometry.opening, floor_z=e.geometry.floor)
        ee = _eef_pose(e)
        if ee:
            entities["eef"] = {"pose": ee}
            entities["eef"]["velocity"] = np.asarray(getattr(e, "eef_vel", lambda: np.zeros(3))(), float).copy()
        bounds = np.asarray(e._drawer_inner_bounds(), float)
        drawer_center = np.asarray([(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, e.drawer_floor_z + e.geometry.height/2], float)
        relations = {"drawer.inner_bounds": bounds, "laptop_to_drawer": drawer_center - entities["laptop"]["pose"]["position"]}
        if ee:
            relations["eef_to_handle"] = entities["handle"]["pose"]["position"] - ee["position"]
        return self._result(entities, relations, {"success": bool(e._check_success())}, {"laptop": True, "drawer": True})


@GeometryFeatureAdapter.register("conveyor_grasp")
class ConveyorGraspGeometry(GeometryFeatureAdapter):
    def extract(self):
        e, d = self.env, self.env.sim.data
        objects = []
        for i, bid in enumerate(e.obj_body_ids):
            objects.append({"id": i, "pose": _pose(e, int(bid)), "velocity": np.asarray(e.obj_vel(i), float).copy(),
                            "category_id": e.records[i]["category"], "variant_id": e.records[i]["variant"],
                            "extent": e.records[i]["extent"].copy(), "phase": str(e.phases[i])})
        entities = {"objects": objects, "eef": {"pose": {"position": np.asarray(e.eef_pos(), float).copy()}}}
        entities["belt"] = {"phase": np.asarray([d.qpos[e.belt_qpos_addr]], float), "speed": np.asarray([getattr(e, "_v_belt", 0.0)], float)}
        from tasks.conveyor_grasp.env import BIN_HALF
        entities["baskets"] = [{"id": i, "position": c.copy(), "half_size": BIN_HALF.copy(),
                                "bounds": np.stack([c + [-BIN_HALF[0], -BIN_HALF[1], .01], c + [BIN_HALF[0], BIN_HALF[1], 2 * BIN_HALF[2]]]),
                                "category_id": int(e.bin_categories[i]), "fixed_category_id": int(e.fixed_categories[i])}
                               for i, c in enumerate(e.bin_centers)]
        entities["examples"] = [{"pose": _pose(e, int(b)), "category_id": r["category"], "basket_id": r["basket"]}
                                for b,r in zip(e.example_body_ids,e.examples)]
        relations = {"object_basket": e._bin_ids.copy(), "correct_mask": e._counted.copy(), "mixed_baskets": e._mixed.copy()}
        return self._result(entities, relations, {"score": e.total_score, "time": float(d.time), "examples_retained": e._examples_retained})






@GeometryFeatureAdapter.register("phone_lean")
class PhoneLeanGeometry(GeometryFeatureAdapter):
    """Read-only geometry snapshot for the phone-lean scene."""

    def extract(self):
        e, d = self.env, self.env.sim.data
        def body_pose(body_id):
            pose = _pose(e, body_id)
            pose["matrix"] = np.asarray(d.body_xmat[body_id], float).reshape(3, 3).copy()
            return pose
        entities = {name: {"pose": body_pose(getattr(e, name+'_body_id'))}
                    for name in ('phone', *e.support_targets)}
        ee_by_arm = {}
        for arm in ('right', 'left'):
            if arm not in e.robots[0].eef_site_id:
                continue
            site = e.robots[0].eef_site_id[arm]
            ee_by_arm[arm] = {'position': d.site_xpos[site].copy(),
                              'matrix': d.site_xmat[site].reshape(3, 3).copy()}
            entities[f"eef_{arm}"] = {"pose": ee_by_arm[arm]}
        if 'right' in ee_by_arm:
            entities["eef"] = {"pose": ee_by_arm['right']}
        phone = entities["phone"]["pose"]["position"]
        supports = {name: entities[name]["pose"]["position"] for name in e.support_targets}
        relations = {
            "eef_to_phone": (phone - ee_by_arm['right']["position"]
                             if 'right' in ee_by_arm else np.zeros(3)),
            "eef_to_phone_by_arm": {
                arm: phone - pose["position"] for arm, pose in ee_by_arm.items()
            },
            "phone_to_supports": {name: position - phone for name, position in supports.items()},
        }
        distance, height = e._phone_metrics()
        state = {
            "phone_lean_xy_error": float(distance),
            "phone_height": float(height),
            "success": bool(distance < 0.06 and height > 0.72),
        }
        validity = {name: True for name in entities}
        return self._result(entities, relations, state, validity)
@GeometryFeatureAdapter.register("pitch_pot")
class PitchPotGeometry(GeometryFeatureAdapter):
    def extract(self):
        e = self.env
        from tasks.pitch_pot.env import TRAY_POS
        velocity = np.asarray(e.sim.data.qvel[e.obj_qvel_addr:e.obj_qvel_addr + 3], float).copy()
        entities = {"projectile": {"pose": _pose(e, e.obj_body_id), "velocity": velocity}, "pot": {"pose": _pose(e, e.pot_body_id)}, "tray": {"position": np.asarray(TRAY_POS, float).copy()}}
        in_state = bool(e._obj_in_pot())
        return self._result(entities, {"projectile_to_pot": entities["pot"]["pose"]["position"] - entities["projectile"]["pose"]["position"]}, {"in_state": in_state, "release_state": "in_pot" if in_state else "free", "attempts": int(e.attempts), "hits": int(e.hits)})




@GeometryFeatureAdapter.register("button_indicator_matching")
class ButtonIndicatorMatchingGeometry(GeometryFeatureAdapter):
    """Read-only geometry truth for the button / indicator event task.

    The button-to-indicator permutation is included for oracle and benchmark
    logs.  Consumers exposing policy observations should filter this field
    according to the task specification's ``geometry_view`` contract.
    """

    def extract(self):
        e = self.env
        d = e.sim.data
        pressed = np.asarray(e._pressed(), dtype=bool)
        indicator_on = np.zeros(int(e.n_buttons), dtype=bool)
        indicator_on[np.asarray(e.assignment, dtype=int)[pressed]] = True
        button_positions = np.asarray(d.qpos[e._button_qadr], dtype=float).copy()
        button_entities = []
        for i, position in enumerate(button_positions):
            body_id = e.sim.model.body_name2id(f"button_{i}")
            button_entities.append({
                "id": i,
                "pose": _pose(e, body_id),
                "deflection_m": float(position),
                "pressed": bool(pressed[i]),
                # Buttons are intentionally unlabeled; labels belong to the
                # indicator side of the random wiring.
                "label": None,
            })
        indicator_entities = []
        for i in range(int(e.n_buttons)):
            gid = e._indicator_gids[i]
            indicator_position = np.asarray(d.geom_xpos[gid], dtype=float).copy()
            indicator_entities.append({
                "id": i,
                "label": str(e.labels[i]),
                "color": str(e.labels[i]) if getattr(e, "label_mode", "") == "colors" else "green",
                "on": bool(indicator_on[i]),
                "pose": {"position": indicator_position,
                         "matrix": np.asarray(d.geom_xmat[gid], dtype=float).copy()},
                "position": indicator_position,
            })
        entities = {
            "buttons": button_entities,
            "indicators": indicator_entities,
        }
        ee = _eef_pose(e)
        if ee:
            entities["eef"] = {"pose": ee}
        assignment = np.asarray(e.assignment, dtype=np.int32).copy()
        relations = {
            "button_to_indicator": assignment,
            "target_indicator": np.asarray([int(e._target_indicator)], dtype=np.int32),
            "target_label": str(e.labels[e._target_indicator]),
        }
        state = {
            "score": int(e._score),
            "events": list(getattr(e, "_events", ())),
            "elapsed_sec": float(d.time),
            "label_mode": str(e.label_mode),
        }
        validity = {"buttons": True, "indicators": True, "eef": ee is not None}
        return self._result(entities, relations, state, validity)
