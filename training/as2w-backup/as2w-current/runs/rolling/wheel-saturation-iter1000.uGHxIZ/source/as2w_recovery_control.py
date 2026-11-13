"""Backend-independent staged recovery and fixed-stand target offsets."""
import torch


def staged_target(initial, action, age, *, fast_entry=None):
    entry = ((age-.22)/.50).clamp(0.,1.)
    if fast_entry is not None:
        earlier = ((age-.06)/.34).clamp(0.,1.)
        entry = torch.where(fast_entry, earlier, entry)
    plant = ((age-.22)/1.1).clamp(0.,1.)
    extend = ((age-1.45)/1.2).clamp(0.,1.)
    crouch = action.new_tensor([0.]*4+[.46]*4+[-.72]*4)/.35
    prior = initial.lerp(crouch,plant[:,None])*(1.-extend[:,None])
    return prior + .42*entry[:,None]*action


def latch_handoff(steady, frames, height, up, dt, *, min_height=.28):
    begin = steady*dt >= .45
    viable = (height>min_height) & (up>.90)
    frames = torch.where(viable & (begin | (frames>0)), frames+1, 0)
    return frames, (frames*dt/.85).clamp(0.,1.)


def stand_integral(bias, measured_error, blend, height, up, dt, *, min_height=.29):
    enabled = (blend >= .80) & (up>.93) & (height>min_height)
    next_bias = ((1.-.04*dt)*bias - 1.10*dt*measured_error).clamp(-.34,.34)
    return torch.where(enabled[:,None], next_bias, 0.)


def mirror_legs(frame):
    order = (1,0,3,2,5,4,7,6,9,8,11,10)
    sign = frame.new_tensor((-1.,)*4+(1.,)*8)
    return frame[:,order]*sign


def mirror_observation(frame):
    imu_sign = frame.new_tensor((-1.,1.,-1.,1.,-1.,1.))
    return torch.cat((frame[:,:6]*imu_sign,mirror_legs(frame[:,6:18]),
                      mirror_legs(frame[:,18:30]),mirror_legs(frame[:,30:42])),dim=1)


def augment_recovery(env, obs=None, actions=None):
    if obs is not None:
        reflected = obs.clone()
        reflected["policy"] = mirror_observation(obs["policy"])
        obs = torch.cat((obs,reflected),dim=0)
    if actions is not None:
        actions = torch.cat((actions,mirror_legs(actions)),dim=0)
    return obs,actions


def recovery_roll_brace(gravity_y, age):
    """Extend the low-side legs during the early lateral recovery phase."""
    side = gravity_y.new_tensor((1.,-1.,1.,-1.))
    ramp = ((age-.10)/.25).clamp(0.,1.)*((1.35-age)/.45).clamp(0.,1.)
    low_side = ((gravity_y[:,None]*side-.12)/.50).clamp(0.,1.)
    calf = .38*low_side*ramp[:,None]
    return torch.cat((torch.zeros_like(calf),-.5*calf,calf),dim=1)/.35
