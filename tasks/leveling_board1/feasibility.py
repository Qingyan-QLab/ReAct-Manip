"""Check a released stack on a separate simulation; not a robot trajectory."""
from types import SimpleNamespace
import numpy as np
from robosuite.utils.binding_utils import MjSim
from .instances import SHIM_NAMES, SLOTS, validate_task_instance
from .scoring import LevelScore


def certify_dimensions(task):
    from .env import LevelingBoard1Arena, LevelingBoard1
    validate_task_instance(task)
    sim = MjSim.from_xml_string(LevelingBoard1Arena(task['shim_dimensions']).get_xml())
    model, data = sim.model, sim.data
    ids = {n:model.body_name2id(n) for n in SHIM_NAMES}
    gids = {n:model.geom_name2id(n+'_collision') for n in SHIM_NAMES}
    hinge = model.joint_name2id('board_hinge')
    qh, vh = int(model.jnt_qposadr[hinge]), int(model.jnt_dofadr[hinge])
    fake = SimpleNamespace(sim=sim, shim_geom_ids=gids, board_body_id=model.body_name2id('board'))
    initial = {}
    for n, xy in zip(SHIM_NAMES,SLOTS):
        q = int(model.jnt_qposadr[model.body_jntadr[ids[n]]])
        data.qpos[q:q+7] = [*xy,.68+task['shim_dimensions'][n][2]/2+.001,1,0,0,0]
    # The board underside is 12 cm above the table at zero hinge angle.
    # The 3/4/5 cm sides form a centered stack, widest footprint first.
    axes = task['solution_axes']
    names = sorted(axes, key=lambda n:np.prod([x for j,x in enumerate(task['shim_dimensions'][n]) if j!=axes[n]]), reverse=True)
    top = .68
    for n in names:
        axis = axes[n]; height = task['shim_dimensions'][n][axis]
        quat = ([1.,0,0,0] if axis==2 else [2**-.5,0,2**-.5,0] if axis==0 else [2**-.5,2**-.5,0,0])
        pose = [.30,-.10,top+height/2, *quat]
        q = int(model.jnt_qposadr[model.body_jntadr[ids[n]]])
        data.qpos[q:q+7] = pose
        initial[n] = pose
        top += height
    data.qpos[qh] = 0
    sim.forward()
    score = LevelScore(20)
    try:
        for tick in range(201):
            for _ in range(round(.05/model.opt.timestep)):
                sim.step()
            released,supported,shims = LevelingBoard1._support_state(fake)
            report = score.update(angle=float(np.degrees(data.qpos[qh])), released=released,
                supported=supported, board_speed=float(data.qvel[vh]),
                positions={n:data.body_xpos[ids[n]].copy() for n in shims},
                linear_speed=max((np.linalg.norm(data.get_body_xvelp(n)) for n in shims),default=0),
                angular_speed=max((np.linalg.norm(data.get_body_xvelr(n)) for n in shims),default=0))
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise ValueError('Non-finite leveling witness')
            if report['terminal_success']:
                return dict(passed=True, type='isolated_released_stack', initial_poses=initial,
                            angle_deg=float(np.degrees(data.qpos[qh])), evaluation_sec=(tick+1)*.05,
                            score=report, robot_trajectory_certified=False)
        raise ValueError('No stable leveling stack witness for Episode dimensions')
    finally:
        sim.free()
