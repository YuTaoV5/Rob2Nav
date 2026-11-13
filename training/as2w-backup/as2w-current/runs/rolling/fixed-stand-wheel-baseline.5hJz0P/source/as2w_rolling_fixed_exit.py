"""Rolling residual action mapping with command-gated fixed stand."""
def rolling_fixed_exit(raw, command, *, pure_turn_common_scale=1., gyro_z=None, wheel_dq=None):
    raw = raw.new_zeros(raw.shape)
    vx, yaw = command[:, :1], command[:, 2:3]
    turning = (yaw.abs() / .20).clamp(max=1.)
    motion = ((vx.abs() / .10).maximum(turning)).clamp(max=1.)
    side = raw.new_tensor((1., -1., 1., -1.))
    wheel = raw[:, 12:].clamp(-1., 1.)
    common = wheel.mean(dim=1, keepdim=True)
    steering = (wheel * side).mean(dim=1, keepdim=True)
    legs = .18 * motion * raw[:, :12].clamp(-1., 1.)
    turn_common = common
    if pure_turn_common_scale != 1.:
        forward_blend = (vx.abs() / .04).clamp(max=1.)
        turn_common = common * (pure_turn_common_scale + (1.-pure_turn_common_scale)*forward_blend)
    yaw_drive = yaw
    common_drive = .10 * turning * turn_common
    if gyro_z is not None:
        engagement = (yaw.abs() / .035).clamp(max=1.)
        correction = (.8 * (yaw - gyro_z[:, None])).clamp(-.10, .10)
        yaw_drive = yaw + engagement * correction
        common_drive = common_drive.clamp(-.015, .015)
    linear = vx + common_drive - .66 * (1. + .25 * steering) * yaw_drive * side
    wheels = linear / (.086 * 16.) + .035 * motion * (1. - turning) * (wheel - common)
    if wheel_dq is not None:
        error = ((wheels - wheel_dq / 16.) * side).mean(dim=1, keepdim=True)
        engagement = (yaw.abs() / .06).clamp(0., 1.)
        trim = (.8 * error).clamp(-.08, .08) * engagement
        wheels = wheels + trim * side
    return legs, wheels.clamp(-6., 6.)


def mirror_rolling_joints(frame):
    pairs = (1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14)
    reflected = frame[:,pairs].clone()
    reflected[:,:4] *= -1.
    return reflected


def mirror_rolling_observation(frame):
    reflected = frame.clone()
    reflected[:,[1,2,3,5,7]] *= -1.
    for start in (10,26,42):
        reflected[:,start:start+16] = mirror_rolling_joints(frame[:,start:start+16])
    if frame.shape[-1] == 65:
        reflected[:,60:64] = frame[:,[61,60,63,62]]
    return reflected


def rolling_stand_bias(bias, command, q_error, gyro, up, wheel_speed, dt, *, enabled=None):
    """Compensate static joint deflection only around centered-stick standing."""
    import torch

    activity = (command[:,:1].abs()/.08).maximum(command[:,2:3].abs()/.15)
    quiet = ((gyro.norm(dim=1)<.30) & (up>.90)
             & (wheel_speed.abs().amax(dim=1)<1.2))
    integrate = (activity[:,0]<.05) & quiet
    if enabled is not None:
        integrate &= enabled
    candidate = ((1.-.04*dt)*bias - 2.*dt*q_error).clamp(-.44,.44)
    bias = torch.where(integrate[:,None],candidate,bias)
    bias = torch.where((activity<1.) & (up[:,None]>.85),bias,0.)
    return bias, (1.-activity).clamp(0.,1.)*bias/.35


def rolling_wheel_hold(reference, frames, command, wheel_angle, wheel_speed, gyro, dt, *, common_feedback=False, leg_speed=None, arm=None, common_gain_scale=1., common_speed_damping=0.):
    """Latch wheel position after braking, with wrapped angle feedback."""
    import torch

    centered = command.abs().amax(dim=1) < .005
    quiet = (wheel_speed.abs().amax(dim=1)<.35) & (gyro.norm(dim=1)<.12) & (gyro[:,2].abs()<.06)
    latch_time = .12
    if leg_speed is not None:
        quiet &= leg_speed.abs().amax(dim=1)<.50
        latch_time = .35
    if arm is not None:
        quiet &= arm
    latched = frames*dt>=latch_time
    frames = torch.where(centered & (latched | quiet),frames+1,0)
    capture = (frames*dt>=latch_time) & ~latched
    reference = torch.where(capture[:,None],wheel_angle,reference)
    error = (wheel_angle-reference+2.*torch.pi).remainder(4.*torch.pi)-2.*torch.pi
    if common_feedback:
        side = error.new_tensor((1.,-1.,1.,-1.))
        longitudinal = error.mean(dim=1,keepdim=True)
        heading = (error*side).mean(dim=1,keepdim=True)
        target_speed = common_gain_scale*(-2.5*longitudinal-1.4*heading*side)
        if common_speed_damping:
            target_speed = target_speed - common_speed_damping * wheel_speed.mean(dim=1,keepdim=True)
        target = target_speed.clamp(-.75,.75)/16.
    else:
        target = (-6.*error).clamp(-1.5,1.5)/16.
    return reference,frames,target*(frames*dt>=latch_time)[:,None]


def rolling_precalibrated_stand(bias, command, q_error, gyro, up, wheel_speed, frames, age, dt, *, continuous=False, straight_only=False, recalibrate=False):
    """Calibrate under initial support, then reuse the frozen standing offset."""
    import torch

    activity = (command[:,:1].abs()/.08).maximum(command[:,2:3].abs()/.15)
    quiet = ((gyro.norm(dim=1)<.20) & (up>.92) & (wheel_speed.abs().amax(dim=1)<.55))
    calibrate = ((age < 4.) | recalibrate) & quiet & (frames*dt>=.30) & (activity[:,0]<.05)
    correction = (-1.40*dt*q_error).clamp(-.004,.004)
    candidate = ((1.-.03*dt)*bias + correction).clamp(-.38,.38)
    bias = torch.where(calibrate[:,None],candidate,bias)
    release_blend = ((frames*dt-.45)/1.05).clamp(0.,1.)
    blend = torch.where(age<4.,torch.ones_like(release_blend),release_blend)
    scale = blend[:,None]*(1.-activity).clamp(0.,1.)
    if continuous:
        upright_scale = ((up-.82)/.10).clamp(0.,1.)[:,None]
        if straight_only:
            straight = (1.-command[:,2:3].abs()/.12).clamp(0.,1.)
            scale = scale.lerp(upright_scale,straight)
        else:
            scale = upright_scale
    output = scale*bias/.35
    output[:, :4] = output[:, :4].clamp(-.12/.35,.12/.35)
    output[:, 4:8] = 0.
    neutral_blend = (1. - activity/.6).clamp(0.,1.)
    output[:, 8:] = neutral_blend * (bias[:,8:].clamp(0.,.06)/.35)
    return bias,output


def rolling_loaded_stance(bias, command, up):
    """Preserve calibrated leg extension and distribute contact through a turn."""
    import torch

    turning = (command[:,2:3].abs()/.20).clamp(0.,1.)
    moving = (command[:,:1].abs()/.08).maximum(command[:,2:3].abs()/.15).clamp(0.,1.)
    x = up.new_tensor((.2119,.2119,-.2119,-.2119))
    y = up.new_tensor((.1654,-.1654,.1654,-.1654))
    projected = (up[:,:1]*x+up[:,1:2]*y)/up[:,2:3].clamp_min(.85)
    calf = (projected/.152).clamp(-.15,.15)*turning
    correction = torch.cat((torch.zeros_like(calf),-.5*calf,calf),dim=1)
    return (.55*moving*bias + correction)/.35


def parking_odometry(wheel_angle, leg_error, up):
    """Estimate chassis longitudinal displacement from rolling contact kinematics."""
    import torch

    _,thigh_error,calf_error = leg_error.split(4,dim=1)
    thigh,calf = .8+thigh_error,-1.6+calf_error
    pitch = torch.atan2(-up[:,:1],up[:,1:].norm(dim=1,keepdim=True))
    x = up.new_tensor((.2119,.2119,-.2119,-.2119))
    x = x-.212*(thigh.sin()+(thigh+calf).sin())
    z = -.212*(thigh.cos()+(thigh+calf).cos())
    hub_x = pitch.cos()*x+pitch.sin()*z
    return wheel_angle+thigh_error+calf_error+pitch-hub_x/.086


def rolling_park_policy_trim(action, command, frames, dt, max_rad):
    """Limit policy authority to loaded rear joints around locked-wheel stand."""
    neutral = (1. - command.abs().amax(dim=1)/.005).clamp(0.,1.)
    ramp = ((frames*dt - .12)/.60).clamp(0.,1.)
    trim = action * 0.
    for start in (6, 10):
        trim[:, start:start+2] = action[:,start:start+2].clamp(-1.,1.)
    return trim * (neutral * ramp)[:, None] * max_rad / .35
