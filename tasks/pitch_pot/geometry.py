"""Task-owned geometry adapter registered through the existing extension point."""
from react_manip.core.geometry import GeometryFeatureAdapter, _pose


@GeometryFeatureAdapter.register("pitch_pot")
class PhysicalPitchPotGeometry(GeometryFeatureAdapter):
    def extract(self):
        e = getattr(self.env, "env", self.env)
        entities = {
            "projectile": {"pose": _pose(e, e.obj_body_id), "velocity": e.ball_velocity()},
            "pot": {"pose": _pose(e, e.pot_body_id)},
            "tray": {"position": e.tray_pos.copy()},
        }
        relation = entities["pot"]["pose"]["position"] - e.ball_pos()
        return self._result(entities, {"projectile_to_pot": relation},
                            {"in_state": bool(e._obj_in_pot()), "release_state": e.phase,
                             "attempts": e.attempts, "hits": int(e._check_success())})
