"""Internal geometry snapshots; never add mass/friction to policy observations."""
from react_manip.core.geometry import GeometryFeatureAdapter
from .model import BAR, BAR_RADIUS


@GeometryFeatureAdapter.register('rope_hang')
class RopeHangGeometry(GeometryFeatureAdapter):
    def extract(self):
        e=getattr(self.env,'env',self.env)
        return self._result({'rope':{'nodes':e.rope_points()},
                             'bar':{'position':BAR.copy(),'radius':BAR_RADIUS},
                             'eef':{'position':e.eef_pos()}},
                            {'bar_contact':e._bar_contact,'external_contact':e._external_contact},
                            {'phase':e.phase,'stable_seconds':e.stability.elapsed,
                             'success':e._check_success()})
