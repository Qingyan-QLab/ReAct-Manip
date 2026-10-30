"""Robot-free calibration. State is assigned only at trial initialization."""
from functools import lru_cache
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .track import Track, Cup, arena_root, BALL_RADIUS, BALL_MASS, RELEASE_RANGE, values


def build_model(track, timestep=.002, cup=None):
    cup = cup or Cup()
    root = arena_root()
    root.find('option').set('timestep', str(timestep))
    track.add_to(root, cup=cup)
    body = ET.SubElement(root.find('worldbody'), 'body', name='ball')
    ET.SubElement(body, 'freejoint', name='ball_joint')
    ET.SubElement(body, 'geom', name='ball_g0', type='sphere', size=str(BALL_RADIUS),
                  mass=str(BALL_MASS), rgba='.9 .2 .15 1', friction='.8 .005 .0001', solref='.01 1')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def trial(model, track, release_x, duration=4, cup=None):
    cup = cup or Cup()
    data = mujoco.MjData(model)
    data.qpos[:3] = track.ball_position(release_x)
    data.qpos[3:7] = [1,0,0,0]
    mujoco.mj_forward(model, data)
    ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'ball_g0')
    rail_ids = {i for i in range(model.ngeom) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith('rail_')}
    last_contact = None
    launch = crossing = launch_sample = None
    settle = 0.
    prev = data.qpos[:3].copy()
    for _ in range(int(duration/model.opt.timestep)):
        mujoco.mj_step(model, data)
        p = data.qpos[:3].copy()
        contacts = [c.geom2 if c.geom1 == ball else c.geom1 for c in data.contact
                    if (c.geom1 == ball and c.geom2 in rail_ids) or (c.geom2 == ball and c.geom1 in rail_ids)]
        touched = bool(contacts)
        if touched:
            launch_sample = None
            if any(int(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g).rsplit('_',1)[1]) >= track.segments-3 for g in contacts):
                last_contact = p.copy()
        elif launch is None and last_contact is not None:
            if launch_sample is None:
                launch_sample = (p.copy(),data.qvel[:3].copy(),float(data.time))
            if data.time-launch_sample[2] >= .008 and p[0] > .875 and launch_sample[1][2] > 0:
                launch = {'position':launch_sample[0].tolist(), 'velocity':launch_sample[1].tolist()}
        mouth = cup.mouth_height
        if crossing is None and launch is not None and prev[2] > mouth >= p[2] and data.qvel[2] < 0:
            t = (mouth-prev[2])/(p[2]-prev[2])
            crossing = (prev+t*(p-prev)).tolist()
        inside = np.linalg.norm(p[:2]-cup.center[:2]) < cup.radius-BALL_RADIUS and cup.base_height_m < p[2] < mouth-BALL_RADIUS
        settle = settle+model.opt.timestep if inside and np.linalg.norm(data.qvel[:3]) < .08 else 0.
        if settle >= .5 or p[2] < BALL_RADIUS+.005:
            break
        prev = p
    return dict(release_x=float(release_x), success=bool(settle>=.5), launch=launch, crossing=crossing, final=p.tolist())


def scan(track, count=17, timestep=.002, cup=None):
    cup = cup or Cup()
    model = build_model(track, timestep, cup)
    return [trial(model,track,x,cup=cup) for x in np.linspace(*RELEASE_RANGE,count)]


@lru_cache(maxsize=128)
def require_feasible(track, cup=None):
    """Reject scenes without a numerically demonstrated route into the cup.

    This uses a separate simulator, never the live robot state. It is a
    geometric feasibility filter, not a guarantee of arm-controller success.
    """
    cup = cup or Cup()
    if not any(r['success'] and r['launch'] for r in scan(track, count=25, cup=cup)):
        raise ValueError('Track has no demonstrated feasible release; adjust the scenario')
