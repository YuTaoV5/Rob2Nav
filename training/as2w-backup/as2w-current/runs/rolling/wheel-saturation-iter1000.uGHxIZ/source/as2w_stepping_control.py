"""Backend-independent guarded stepping action mapping."""
import torch


def sequence_swing(clock):
    slot_order = clock.new_tensor((0.,2.,3.,1.))
    local = (clock[:,None]*12. - 3.*slot_order).remainder(12.)
    active = local < 3.
    lift = ((local-.85)/.65).clamp(0.,1.) * ((2.48-local)/.65).clamp(0.,1.) * active
    shift = (local/.85).clamp(0.,1.) * ((3.-local)/.52).clamp(0.,1.) * active
    return local, lift, shift


def calibrated_sequence(command, clock, *, guarded=False, stride_gain=.10, lateral_shift_scale=1.,
                        front_lift_m=0., front_reach_scale=1., body_raise_m=0., symmetric_front_support=False):
    local, lift, shift = sequence_swing(clock)
    active = command.abs().amax(dim=1,keepdim=True) > .015
    lift *= active
    shift *= active
    side = command.new_tensor((1.,-1.,1.,-1.))
    fore = command.new_tensor((1.,1.,-1.,-1.))
    shift_x = command.new_tensor((.04,.04,.08,.08))
    shift_y = command.new_tensor((.09,.09,.06,.06))
    support = command.new_tensor((.042,.018,.030,.030))
    if guarded:
        shift_x = command.new_tensor((.02,.02,.08,.08))
        support = command.new_tensor((.030,.015,.030,.030))
        if symmetric_front_support:
            support = command.new_tensor((.0225, .0225, .030, .030))
    extension = (shift*support).sum(dim=1,keepdim=True) * (local >= 3.)
    swing_forward = ((local-.85)/1.35).clamp(0.,1.)
    return_to_rear = ((local-2.48)/9.52).clamp(0.,1.)
    stride = (2.*(swing_forward-return_to_rear)-1.)*active
    vx = command[:,:1]-command[:,2:3]*side*.1654
    vy = command[:,1:2]+command[:,2:3]*fore*.2119
    reach = stride_gain*vx*stride
    if front_reach_scale != 1.:
        forward_fore_reach = (fore > 0.) & (command[:,:1] > .015) & (reach > 0.)
        reach = torch.where(forward_fore_reach,front_reach_scale*reach,reach)
    x = (shift*shift_x*fore).sum(dim=1,keepdim=True) + reach
    y = side*.1054 + lateral_shift_scale*(shift*shift_y*side).sum(dim=1,keepdim=True) + stride_gain*vy*stride
    z = -.424*torch.cos(command.new_tensor(.8)) + .105*lift - extension
    z = z - body_raise_m * active
    if front_lift_m:
        forward = (command[:,:1]/.16).clamp(0.,1.)
        z = z + front_lift_m * forward * lift * (fore > 0.)
    planar = -(y.square()+z.square()-.1054**2).clamp_min(.01).sqrt()
    hip = torch.atan2(z,y)-torch.atan2(planar,side*.1054)
    calf = -torch.acos(((x.square()+planar.square())/(2.*.212**2)-1.).clamp(-.99,.99))
    thigh = torch.atan2(-x,-planar)-torch.atan2(calf.sin(),1.+calf.cos())
    return torch.cat((hip,thigh-.8,calf+1.6),dim=1)


def guarded_sequence_action(command, age, action, delayed_rear_thigh_error, motion_end, *,
                            period=12., stride_gain=.10, gravity=None, gyro=None, lateral_shift_scale=1.,
                            front_lift_m=0., front_reach_scale=1., damp_lateral_shift=False, body_raise_m=0., symmetric_front_support=False):
    phase = (age-2.).clamp_min(0.)/period
    if gravity is not None:
        up = -gravity
        if damp_lateral_shift:
            roll_load = (up[:,1].abs() + .08*gyro[:,0].abs())/.35
            lateral_shift_scale = lateral_shift_scale*(1.-.65*roll_load.clamp(0.,1.))[:,None]
        pitch = torch.atan2(-up[:,0],up[:,2].clamp_min(.75))
        pitch_error = (pitch+.12*gyro[:,1]).clamp(-.34,.34)
        stride_gain = stride_gain*(1.-.55*(pitch_error.abs()/.25).clamp(0.,1.))[:,None]
    prior = calibrated_sequence(command,phase,guarded=True,stride_gain=stride_gain,
                                lateral_shift_scale=lateral_shift_scale,front_lift_m=front_lift_m,
                                front_reach_scale=front_reach_scale, body_raise_m=body_raise_m,
                                symmetric_front_support=symmetric_front_support)
    if gravity is not None:
        _,lift,_ = sequence_swing(phase)
        fore = action.new_tensor((1.,1.,-1.,-1.))
        calf = (.85*pitch_error[:,None]*fore)*(1.-.75*lift)
        prior = prior + torch.cat((torch.zeros_like(calf),-.5*calf,calf),dim=1)
    activity = (command.abs().amax(dim=1,keepdim=True)/.08).clamp(0.,1.)
    target = (prior + .12*action)*activity
    rear_q = .8 + delayed_rear_thigh_error
    support_bias = (.65*(.53-rear_q)).clamp(0.,.23)
    target[:,6:8] = (target[:,6:8] + activity*support_bias).clamp(-.25,.48)
    blend = (age-1.).clamp(0.,1.)*(motion_end+2.-age).clamp(0.,1.)
    return blend[:,None]*target/.30
