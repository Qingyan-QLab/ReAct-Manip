"""Privileged demo planning using physical rehearsals in an independent world.

This is not a visual policy. No trial changes the displayed environment, ball
state, target pose, gravity or contacts. Every trial executes ordinary actions.
"""
from dataclasses import asdict, replace
import numpy as np
from .arm import PickThrowController, ThrowMotion


def motion_for_target(env):
    """Ballistic initial guess; the robot's tracking lag is calibrated separately."""
    e = getattr(env, 'env', env)
    if 'base_distance_m' not in e.scenario.parameters.get('pot', {}):
        return ThrowMotion()
    target = e.sim.data.body_xpos[e.pot_body_id]
    horizontal = max(.1, float(np.linalg.norm(target[:2] - [.65, 0])))
    height = .325 - 1.0
    root = np.hypot(horizontal, height)
    speed = np.sqrt(9.81 * (root + height))
    angle = np.arctan2(root + height, horizontal)
    return ThrowMotion(swing_duration=.4, open_at=.20 + .06 * min(horizontal / 10, 1),
                       swing_dx=float(speed * np.cos(angle) * .56),
                       swing_dz=float(speed * np.sin(angle) * .56),
                       lift_x=.2, lift_z=.85, swing_pitch=1.5,
                       swing_dy=float(target[1]) * .4)


def calibrate_throw(env, *, max_trials=24, progress=None):
    """Select the smallest observed mouth-plane miss, then validate that throw.

    Budget is bounded. Failure to find a hit is reported, not called unreachable
    in principle. A replay uses the same Scenario and deterministic reset.
    """
    from tasks.pitch_pot.env import PitchPot
    e = getattr(env, 'env', env)
    initial = motion_for_target(e)
    if 'base_distance_m' not in e.scenario.parameters.get('pot', {}):
        return initial, {'status': 'near_calibration', 'trials': 0}
    if max_trials < 1:
        raise ValueError('max_trials must be positive')
    target = e.sim.data.body_xpos[e.pot_body_id].copy()
    sim_env = PitchPot(robots=e.robot_id, seed=e.scenario.seed, scenario=e.scenario, hard_reset=False,
                       control_freq=e.control_freq)
    candidates = [initial]
    # Broad seeds cover acceleration / loft tradeoffs before local refinement.
    for dx, dz, opening in ((1.5, .6, .18), (1.5, 1.2, .24),
                            (2.43, 1.38, .22), (2.15, 1.52, .24),
                            (.9, .6, .18), (1.2, .8, .22)):
        candidates.append(replace(initial, swing_dx=dx, swing_dz=dz, open_at=opening))
    best, best_motion = None, initial
    tried = set()
    refinements = 0
    def rollout(motion):
        sim_env.reset()
        controller = PickThrowController(sim_env, motion=motion)
        landing = None
        previous = None
        for _ in range(450):
            _, _, done, truncated, info = sim_env.step(controller.act())
            p = sim_env.ball_pos()
            if sim_env._released and previous is not None and previous[2] > .325 >= p[2]:
                fraction = (previous[2] - .325) / (previous[2] - p[2])
                landing = previous[:2] + fraction * (p[:2] - previous[:2])
            previous = p
            if done or truncated:
                break
        valid = sim_env._released and sim_env.flight_platform_contacts == 0 and landing is not None
        failure = info.get('failure') or {}
        valid = valid and failure.get('reason') not in ('ball_slipped', 'released_without_throw', 'non_finite_state')
        miss = float(np.linalg.norm(landing - target[:2])) if valid else None
        return dict(success=bool(sim_env._check_success()), miss_m=miss,
                    crossing_xy=None if landing is None else landing.tolist(),
                    release_velocity=None if sim_env.release_velocity is None else sim_env.release_velocity.tolist(),
                    failure=info.get('failure'))
    try:
        for index in range(max_trials):
            if not candidates:
                # Contract around the current best; include lateral correction.
                factor = .24 * .65**refinements
                refinements += 1
                for key in ('swing_dx', 'swing_dz'):
                    for sign in (-1, 1):
                        candidates.append(replace(best_motion, **{key: max(.1, getattr(best_motion, key) * (1 + sign * factor))}))
                for sign in (-1, 1):
                    candidates.append(replace(best_motion, open_at=float(np.clip(best_motion.open_at + sign * .02, .06, .32))))
                if best and best['crossing_xy']:
                    correction = float(target[1] - best['crossing_xy'][1])
                    candidates.insert(0, replace(best_motion, swing_dy=best_motion.swing_dy + correction))
            motion = candidates.pop(0)
            key = tuple(asdict(motion).values())
            if key in tried:
                # A repeated parameter vector provides no new evidence.
                motion = replace(motion, swing_dz=motion.swing_dz * 1.01)
            tried.add(tuple(asdict(motion).values()))
            result = rollout(motion)
            if result['miss_m'] is not None and (best is None or result['success'] or result['miss_m'] < best['miss_m']):
                best, best_motion = result, motion
            if progress:
                progress(index + 1, max_trials, best)
            if result['success']:
                break
        report = dict(status='validated_hit' if best and best['success'] else 'best_effort',
                      trials=len(tried), target_xy=target[:2].tolist(),
                      motion=asdict(best_motion), **(best or {'success': False, 'miss_m': None}))
        return best_motion, report
    finally:
        sim_env.close()
