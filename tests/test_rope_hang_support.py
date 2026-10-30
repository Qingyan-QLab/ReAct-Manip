"""Slip-based scoring must distinguish a swinging drape from a moving support."""
import numpy as np
from tasks.rope_hang.model import BAR, BAR_RADIUS, BAR_HALF_LENGTH
from tasks.rope_hang.state import Stability, hanging_geometry, support_coordinates


def drape():
    return BAR + np.array([[-.08,0,-.35],[-.045,0,-.05],
                           [-.025,0,.04],[.025,0,.04],[.045,0,-.05],[.08,0,-.35]])


def test_swinging_ends_do_not_reset_support_window():
    s=Stability()
    previous=None;peak_speed=0
    for i in range(501):
        p=drape()
        # Large lateral endpoint swings, including beyond the bar's axial margin.
        p[[0,-1],1]+=.26*np.sin(i*.01*5)
        if previous is not None:peak_speed=max(peak_speed,np.max(np.linalg.norm(p-previous,axis=1))/.01)
        previous=p
        assert hanging_geometry(p,.012)
        q=support_coordinates(p,.025,.012,{2})
        success=s.update(q,True,.01)
    assert success and peak_speed>.08
    np.testing.assert_allclose(s.range,0)


def test_continuous_material_and_axial_sliding_never_succeed():
    for axis in (0,1):
        s=Stability()
        for i in range(1500):
            q=np.array([.4,-.25]);q[axis]+=.003*i*.01
            assert not s.update(q,True,.01)


def test_trend_rejects_motion_even_inside_range_limit():
    s=Stability()
    for i in range(500):
        assert not s.update([.4+(-.0045 if i<250 else .0045),-.25],True,.01)
    assert not s.update([.4045,-.25],True,.01)


def test_support_loss_requires_fresh_full_window():
    s=Stability()
    assert not s.update([.4,-.25],True,4.9)
    assert not s.update(None,False,.01)
    assert s.elapsed==0
    assert not s.update([.4,-.25],True,4.9)
    assert s.update([.4,-.25],True,.1)


def test_crossing_requires_local_contact_and_end_clearance():
    p=drape()
    assert support_coordinates(p,.025,.012,{2}) is not None
    assert support_coordinates(p,.025,.012,set()) is None
    assert support_coordinates(p,.025,.012,{5}) is None
    p[:,1]=BAR[1]+BAR_HALF_LENGTH-.05
    assert support_coordinates(p,.025,.012,{2}) is None
    p=drape();p[2:4,2]+=.1
    assert support_coordinates(p,.025,.012,{2}) is None


def test_ambiguous_crossings_and_nonfinite_data_are_rejected():
    p=drape();p=np.insert(p,3,[BAR[0]+.01,BAR[1],BAR[2]+.04],axis=0)
    p=np.insert(p,4,[BAR[0]-.01,BAR[1],BAR[2]+.04],axis=0)
    assert support_coordinates(p,.025,.012,{2,3,4}) is None
    p=drape();p[0,0]=np.nan
    assert support_coordinates(p,.025,.012,{2}) is None
    assert not Stability().update([np.nan,0],True,5)
