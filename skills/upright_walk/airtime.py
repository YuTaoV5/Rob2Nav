"""Measured rear-foot steps for a zero-velocity task; reward memory only."""
import torch


def step_events(loaded,clearance,active,dt,state=None):
    """Count loaded -> real flight -> opposite-foot landing, not unloading/sliding."""
    if state is None:
        n=loaded.shape[0]
        state=dict(air=torch.zeros_like(clearance),peak=torch.zeros_like(clearance),
            seen=torch.zeros_like(loaded),previous=loaded.clone(),
            clock=torch.zeros(n,device=clearance.device,dtype=clearance.dtype),
            last_clock=torch.full((n,),-float('inf'),device=clearance.device,dtype=clearance.dtype),
            last_leg=torch.full((n,),-1,device=clearance.device,dtype=torch.long))
    inactive=~active
    for key in ('air','peak','seen'):state[key][inactive]=0
    state['clock'][inactive]=0;state['last_clock'][inactive]=-float('inf');state['last_leg'][inactive]=-1
    state['clock']+=active*dt
    air=torch.where(loaded,state['air'],state['air']+dt)
    peak=torch.where(loaded,state['peak'],torch.maximum(state['peak'],clearance))
    valid=loaded&~state['previous']&state['seen']&(air>=.06-1e-6)&(peak>=.04)&active[:,None]
    leg=valid.long().argmax(-1)
    qualifies=(valid.sum(-1)==1)&(leg!=state['last_leg'])&((state['clock']-state['last_clock'])>=.08-1e-6)
    state['last_clock'][qualifies]=state['clock'][qualifies];state['last_leg'][qualifies]=leg[qualifies]
    state['air']=torch.where(loaded,torch.zeros_like(air),air)
    state['peak']=torch.where(loaded,torch.zeros_like(peak),peak)
    state['seen']|=loaded&active[:,None];state['previous']=loaded.clone()
    return qualifies.to(clearance.dtype),state


def qualifying_airtime(env):
    from skills.fixstand import phase_time
    from skills.upright_walk.task import rear_gait_state,upright
    robot,sensor,loaded,_moving=rear_gait_state(env)
    t=phase_time(env);active=(t>=2.)&(t<5.)
    # The fixed Go2 foot sphere is 0.022m; use clearance, not body-origin height.
    clearance=torch.clamp(robot.data.body_pos_w[:,env._upright_rear_ids,2]-env.scene.env_origins[:,2:3]-.022,min=0.)
    reward,state=step_events(loaded,clearance,active,env.step_dt,getattr(env,'_in_place_step_events',None))
    env._in_place_step_events=state
    return reward*upright(env)
