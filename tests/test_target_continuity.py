import pytest
from rob2nav.target_continuity import assess


def test_target_speed_is_normalized_in_named_joint_order():
    trace=[dict(time=.02,target=[0,0]),dict(time=.04,target=[.1,.1])]
    r=assess(trace,['slow','fast'],[5.,20.])
    assert r['passed'] and r['max_target_slew_urdf_ratio']==1.
    trace[1]['target'][0]=.2
    r=assess(trace,['slow','fast'],[5.,20.])
    assert not r['passed'] and r['joint']=='slow' and r['sample_pairs']==1
    assert r['recorded_control_step_end_s']==.04


def test_missing_or_nonfinite_trace_is_not_a_pass():
    assert not assess([],['joint'],[5.])['passed']
    with pytest.raises(ValueError):assess([dict(time=0,target=[0]),dict(time=.02,target=[float('nan')])],['joint'],[5.])
