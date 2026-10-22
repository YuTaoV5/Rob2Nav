import math
import numpy as np
from rob2nav.in_place_validation import assess_in_place_steps,nonfoot_ground_contact


def trace(mode='stepping'):
    rows=[]
    for i in range(150):
        t=2.+i*.02;forces={k:0. for k in ('FR','FL','RR','RL')};clearance={k:0. for k in forces}
        for leg,offset in [('RR',0.),('RL',.5 if mode!='hopping' else 0.)]:
            phase=((t-2.)*2.+offset)%1.
            swing=(phase-.65)/.35
            air=0.<swing<1. and mode!='static'
            forces[leg]=0. if air else 60.
            clearance[leg]=.065*math.sin(math.pi*swing) if air and mode!='sliding' else 0.
        rows.append(dict(time=t,position=[0.,0.,.48],upright_walk_error=.05,speed=.03,
                         angular_speed=.2,target=np.full(12,.05*math.sin(t*2.)).tolist(),
                         stepping_foot_forces_N=forces,foot_clearance_m=clearance))
    return rows


def test_actual_alternating_clearance_and_load_cycles_pass():
    result=assess_in_place_steps(trace())
    assert result['passed'] and min(result['rear_loaded_step_cycles'].values())>=3


def test_static_hold_cannot_count_as_stepping():
    assert not assess_in_place_steps(trace('static'))['passed']


def test_contact_unloading_without_clearance_is_not_a_step():
    assert not assess_in_place_steps(trace('sliding'))['passed']


def test_simultaneous_hops_cannot_count_as_alternation():
    result=assess_in_place_steps(trace('hopping'))
    assert not result['passed'] and result['alternating_landings']==0


def test_drift_and_target_spike_each_fail():
    rows=trace()
    for i,row in enumerate(rows):row['position'][0]=i*.003
    assert not assess_in_place_steps(rows)['passed']
    rows=trace();rows[80]['target'][0]+=.5
    assert not assess_in_place_steps(rows)['passed']


def test_missing_exact_foot_measurement_and_sample_gaps_fail():
    rows=trace();del rows[4]['foot_clearance_m']
    assert not assess_in_place_steps(rows)['passed']
    rows=trace();del rows[30]
    assert not assess_in_place_steps(rows)['passed']


def test_leg_or_knee_bracing_cannot_masquerade_as_foot_contact():
    floors={0};feet={20,33,46,59}
    assert nonfoot_ground_contact(0,21,floors,feet)
    assert nonfoot_ground_contact(21,0,floors,feet)
    assert not nonfoot_ground_contact(0,20,floors,feet)
    assert not nonfoot_ground_contact(20,0,floors,feet)
    assert not nonfoot_ground_contact(20,21,floors,feet)
    assert not nonfoot_ground_contact(0,0,floors,feet)
