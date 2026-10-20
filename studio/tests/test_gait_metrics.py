import pytest
from moe_perceptive.gait_metrics import GaitMetrics


def test_integrals_contact_mask_and_completed_episode_exclusion():
    m=GaitMetrics(2,.1)
    args=[[[[3.,4.,0.]],[[99.,99.,0.]]],[[[0.,0.,6.]],[[0.,0.,6.]]],[[2.],[10.]],[[3.],[10.]],[[1.,0.,0.],[5.,0.,0.]],[.8,.1]]
    m.update([True,False],*args)
    r=m.episode(0)
    assert r['mean_stance_foot_speed_m_s']==pytest.approx(5.)
    assert r['mean_stance_foot_speed_squared_m2_s2']==pytest.approx(25.)
    assert r['mean_stance_motion_reward_raw_m2_s2']==pytest.approx(25.)
    assert r['peak_stance_foot_speed_m_s']==pytest.approx(5.)
    assert r['absolute_joint_work_J']==pytest.approx(.6)
    assert r['minimum_upright']==.8 and m.episode(1)['samples']==0
    args[1][0][0][2]=0
    m.update([True,False],*args)
    assert m.episode(0)['stance_foot_seconds']==pytest.approx(.1)
    assert m.episode(0)['mean_stance_motion_reward_raw_m2_s2']==pytest.approx(12.5)
    assert m.episode(1)['mean_stance_motion_reward_raw_m2_s2'] is None
    assert m.episode(1)['mean_stance_foot_speed_m_s'] is None
