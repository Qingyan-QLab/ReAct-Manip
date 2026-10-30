"""Final-position score, independent of historical maximum insertion depth."""
import numpy as np

T1 = 10.0
T2 = 40.0
T3 = 60.0


def score_state(*, center, corners, bounds, z_bounds, half_length, elapsed, success, failed=False):
    center, corners = np.asarray(center), np.asarray(corners)
    xmin,xmax,ymin,ymax=bounds
    zmin,zmax=z_bounds
    tol=.0005
    valid=bool(not failed and np.isfinite(center).all() and np.isfinite(corners).all()
               and center[2]>=.65 and np.all(corners[:,0]<=xmax+tol)
               and np.all(corners[:,1]>=ymin-tol) and np.all(corners[:,1]<=ymax+tol)
               and np.all(corners[:,2]>=zmin-tol) and np.all(corners[:,2]<=zmax+tol))
    # Entry X is deliberately unbounded: partial insertion can earn points.
    full_x=xmax-half_length
    distance=50.*float(np.clip((center[0]-xmin)/(full_x-xmin),0.,1.)) if valid else 0.
    time=50.*float(np.clip((T2-elapsed)/(T2-T1),0.,1.)) if success and not failed else 0.
    return {'distance_score':distance,'time_score':time,'total_score':distance+time,
            'score_position_valid':valid,'score_entry_x_m':float(xmin),'score_full_x_m':float(full_x)}
