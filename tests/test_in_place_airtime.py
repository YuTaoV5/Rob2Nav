import copy
import torch
import pytest
import sys
from types import SimpleNamespace as NS
from skills.upright_walk.airtime import step_events
from studio.registry import registry
from studio.config import RunConfig,command


def test_actual_reward_wrapper_does_not_disable_steps_for_zero_commands(monkeypatch):
    from skills.upright_walk.airtime import qualifying_airtime
    import skills.fixstand
    monkeypatch.setattr(skills.fixstand,'phase_time',lambda env:torch.tensor([2.5]))
    robot=NS(data=NS(body_pos_w=torch.tensor([[[0.,0.,.022],[0.,0.,.022]]])))
    loaded=torch.tensor([[True,True]])
    monkeypatch.setitem(sys.modules,'skills.upright_walk.task',NS(
        rear_gait_state=lambda env:(robot,None,loaded,torch.tensor([False])),
        upright=lambda env:torch.ones(1)))
    class Scene(dict):pass
    scene=Scene(robot=robot);scene.env_origins=torch.zeros(1,3)
    env=NS(scene=scene,_upright_rear_ids=[0,1],step_dt=.02)
    assert qualifying_airtime(env).item()==0.
    loaded[0,0]=False;robot.data.body_pos_w[0,0,2]=.082
    for _ in range(4):assert qualifying_airtime(env).item()==0.
    loaded[0,0]=True;robot.data.body_pos_w[0,0,2]=.022
    assert qualifying_airtime(env).item()==1.


def tick(loaded,height,state=None,active=True):
    return step_events(torch.tensor([loaded]),torch.tensor([height],dtype=torch.float32),torch.tensor([active]),.02,state)


def cycle(leg,state,height=.05,frames=4):
    load=[True,True];load[leg]=False
    for _ in range(frames):_,state=tick(load,[height,height],state)
    return tick([True,True],[0.,0.],state)


def test_true_rear_foot_cycles_work_without_any_velocity_command():
    _,state=tick([True,True],[0.,0.])
    for leg in [0,1,0,1]:
        reward,state=cycle(leg,state)
        assert reward.item()==1.


@pytest.mark.parametrize('height,frames',[(0.,4),(.02,4),(.05,1)])
def test_unloading_sliding_or_short_flight_cannot_earn_step_reward(height,frames):
    _,state=tick([True,True],[0.,0.])
    reward,state=cycle(0,state,height,frames)
    assert reward.item()==0.


def test_static_hold_synchronous_hops_and_repeated_same_leg_fail():
    _,state=tick([True,True],[0.,0.])
    for _ in range(20):
        reward,state=tick([True,True],[0.,0.],state);assert reward.item()==0.
    for _ in range(4):_,state=tick([False,False],[.06,.06],state)
    reward,state=tick([True,True],[0.,0.],state);assert reward.item()==0.
    reward,state=cycle(0,state);assert reward.item()==1.
    reward,state=cycle(0,state);assert reward.item()==0.


def test_reset_and_hold_window_do_not_leak_previous_support_or_rewrite_inputs():
    loaded=torch.tensor([[True,False]]);clearance=torch.tensor([[0.,.06]])
    before=loaded.clone(),clearance.clone()
    reward,state=step_events(loaded,clearance,torch.tensor([False]),.02)
    assert reward.item()==0. and torch.equal(loaded,before[0]) and torch.equal(clearance,before[1])
    reward,state=tick([True,True],[0.,0.],state);assert reward.item()==0.


def test_measured_airtime_is_a_web_configurable_opt_in(tmp_path):
    cfg=copy.deepcopy(next(r for r in registry()['recipes'] if r['id']=='upright_walk_ppo')['defaults'])
    cfg['recipe_options'].update(transfer_motors=True,fixstand_roundtrip=True,stepping_gait=True,alternating_rear_gait=True,in_place_stepping=True,in_place_airtime=True,in_place_airtime_weight=70.)
    out=tmp_path/'outputs/studio/runs/preview'
    cmd=command(RunConfig.model_validate(cfg),tmp_path,out,tmp_path/'preview.json','isaac')
    assert '--in-place-airtime' in cmd and cmd[cmd.index('--in-place-airtime-weight')+1]=='70.0'
    cfg['recipe_options']['in_place_stepping']=False
    with pytest.raises(ValueError,match='in_place_airtime'):RunConfig.model_validate(cfg)


def test_live_recipe_metadata_exposes_the_same_typed_options_as_validation():
    recipe=next(r for r in registry()['recipes'] if r['id']=='upright_walk_ppo')
    assert recipe['defaults']['recipe_options']['in_place_airtime'] is False
    assert recipe['option_schema']['in_place_airtime']['type']=='boolean'
    assert recipe['option_schema']['in_place_airtime_weight']['min']==0.
    assert recipe['option_schema']['in_place_airtime_weight']['max']==200.
