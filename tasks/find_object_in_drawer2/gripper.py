"""Task-local stock PiPER finger meshes; no laptop-specific oversized pads."""
from pathlib import Path
import numpy as np
from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.models.grippers.gripper_model import GripperModel


class CabinetSearchGripper(GripperModel):
    def __init__(self, idn=0):
        super().__init__(str(Path(__file__).parent/'assets/search_gripper.xml'),idn=idn)
    @property
    def dof(self):return 1
    @property
    def init_qpos(self):return np.array([.035,-.035])
    def format_action(self, action):
        self.current_action=np.clip(np.asarray(action),-1,1)
        return np.array([-self.current_action[0], self.current_action[0]])
    @property
    def _important_geoms(self):
        return {'left_finger':['finger7_collision0'],'right_finger':['finger8_collision0'],
                'left_fingerpad':['finger7_collision0'],'right_fingerpad':['finger8_collision0']}

GRIPPER_MAPPING['CabinetSearchGripper']=CabinetSearchGripper
