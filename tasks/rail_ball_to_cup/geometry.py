"""Privileged read-only data, never injected into policy observations."""
from react_manip.core.geometry import GeometryFeatureAdapter, _pose

@GeometryFeatureAdapter.register('rail_ball_to_cup')
class RailBallGeometry(GeometryFeatureAdapter):
    def extract(self):
        e=getattr(self.env,'env',self.env)
        return self._result(
            {'ball':{'pose':_pose(e,e.obj_body_id),'velocity':e.ball_velocity()},
             'cup':{'pose':_pose(e,e.pot_body_id),'base_height_m':e.cup.base_height_m},
             'track':{'start_height':e.track.start_height,'exit_angle_deg':e.track.exit_angle_deg}},
            {'ball_to_cup':e.sim.data.body_xpos[e.pot_body_id]-e.ball_pos()},
            {'phase':e.phase,'rail_contact_seen':e._rail_seen,'valley_seen':e._valley_seen,
             'launched':e._flight,'success':e._success})
