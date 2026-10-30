"""Explicit model and policy identities; these are not aliases of legacy robots."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Embodiment:
    source: str
    model: str
    arms: tuple[str, ...]
    dof: int
    policy: str
    native_action: str
    mobile: bool = False


EMBODIMENTS = {
    'panda_libero': Embodiment('libero', 'LiberoPanda', ('right',), 7,
                               'pi05', 'libero_delta7'),
    'aloha_agilex_robotwin': Embodiment('robotwin', 'RoboTwinAloha', ('right', 'left'), 6,
                                      'lingbot', 'eef16'),
    'dual_arx_x5_robodojo': Embodiment('robodojo', 'RoboDojoX5', ('right', 'left'), 6,
                                     'pi05', 'qpos14'),
    'panda_omron_robocasa': Embodiment('robocasa', 'RoboCasaPandaOmron', ('right',), 7,
                                     'rldx', 'robocasa12', mobile=True),
}

# Construction/reset/control checks, not task-success or reachability claims.
# Task-specific PiPER-X/controller restrictions remain explicit until ported.
COMMON_TASKS = ('button_indicator_matching', 'conveyor_grasp', 'box_insertion', 'laptop_into_drawer')
SUPPORTED_TASKS = {
    'panda_libero': COMMON_TASKS + ('curling_push', 'pitch_pot', 'rope_hang', 'pan_flip', 'rail_ball_to_cup', 'ring_toss1', 'tilte_maze', 'turntable_gamble1'),
    'aloha_agilex_robotwin': COMMON_TASKS + ('curling_push', 'pitch_pot', 'rope_hang', 'find_object_in_drawer2', 'leveling_board1', 'car_drive2'),
    'dual_arx_x5_robodojo': COMMON_TASKS + ('find_object_in_drawer2',),
    'panda_omron_robocasa': COMMON_TASKS + ('rope_hang',),
}


def camera_ids(robot_id):
    spec = EMBODIMENTS[robot_id]
    if spec.mobile:
        return ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')
    return ('robot0_head_camera', 'robot0_wrist_right') + (
        ('robot0_wrist_left',) if len(spec.arms) == 2 else ())
