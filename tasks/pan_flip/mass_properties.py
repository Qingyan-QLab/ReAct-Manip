"""Mass properties of a rigid disk with a positive linear density gradient.

In the body frame, rho(x,y,z) = rho_mean * (1 + (1-u)*dot(d, [x,y])/R),
where d = [cos(phi), sin(phi)] and u is uniformity in [0, 1]. Density is
uniform through the thickness. The integral stays equal to the specified mass;
only the center of mass and the inertia about that center change. The beveled
collision hull is a contact approximation to this ideal cylindrical mass model.
"""
import math
import numpy as np


def pie_mass_properties(radius, half_thickness, mass, uniformity=1.0,
                        density_direction_deg=0.0):
    """Return COM (m) and full COM inertia tensor (kg m²), in body axes."""
    values = [radius, half_thickness, mass, uniformity, density_direction_deg]
    if (not np.isfinite(values).all() or min(radius, half_thickness, mass) <= 0
            or not 0 <= uniformity <= 1):
        raise ValueError('Positive finite dimensions/mass and uniformity in [0, 1] required')
    angle = math.radians(density_direction_deg % 360)
    offset = (1 - uniformity) * radius / 4
    com = np.array([offset * math.cos(angle), offset * math.sin(angle), 0.])
    transverse = mass * (3 * radius**2 + (2 * half_thickness)**2) / 12
    origin_inertia = np.diag([transverse, transverse, mass * radius**2 / 2])
    # The linear gradient has zero contribution to second moments about the
    # geometric origin. Translate those moments to the displaced COM.
    inertia = origin_inertia - mass * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
    return com, inertia
