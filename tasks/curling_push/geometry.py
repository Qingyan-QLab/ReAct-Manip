"""Evaluator/CL_Coach-only geometry; never injected into policy observations."""
import numpy as np
from react_manip.core.geometry import GeometryFeatureAdapter


@GeometryFeatureAdapter.register('curling_push')
class CurlingPushGeometry(GeometryFeatureAdapter):
    def extract(self):
        e = self.env
        stone, target = e.stone_position(), e.sim.data.body_xpos[e.target_body_id].copy()
        return self._result(
            entities={
                'stone': {'pose': {'position':stone,'quaternion':e.sim.data.body_xquat[e.stone_body_id].copy()},
                          'velocity':e.stone_velocity()},
                'target': {'pose': {'position':target,'quaternion':np.array([1.,0.,0.,0.])},'radius_m':e._parameters['target']['radius_m']},
                'eef': {'pose': {'position':e.eef_pos(),'matrix':e.eef_mat()}},
            },
            relations={'stone_to_target':target-stone,'planar_error_m':float(np.linalg.norm(target[:2]-stone[:2]))},
            task_state={'phase':e._state.phase,'success':e._state.success,'failure':e._state.failure})
