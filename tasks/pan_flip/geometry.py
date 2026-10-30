"""Internal geometry only; no hidden dynamics in policy observations."""
from react_manip.core.geometry import GeometryFeatureAdapter, _pose


@GeometryFeatureAdapter.register('pan_flip')
class PanFlipGeometry(GeometryFeatureAdapter):
    def extract(self):
        e=getattr(self.env,'env',self.env)
        local,normal,clearance,velocity,omega=e.relative_state()
        return self._result(
            {'pan':{'pose':_pose(e,e.pan_body_id)},
             'pie':{'pose':_pose(e,e.pie_body_id),'velocity':e.pie_velocity(),'angular_velocity':e.pie_angular_velocity()},
             'eef':{'pose':{'position':e.eef_pos(),'matrix':e.eef_mat()}}},
            {'pie_in_pan_frame':local,'face_normal_in_pan':normal,'clearance_m':clearance,
             'relative_velocity':velocity,'relative_angular_velocity':omega},
            {'phase':e.phase,'airborne':e._airborne,'landed':e._landed,
             'flipped':e._flipped,'success':e._success})
