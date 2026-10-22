"""Pinned backflip controller parameters shared by export and native playback."""
import math
import numpy as np

JOINT_NAMES=tuple(f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','calf'))


def validate_entry_contract(env,contract):
    if contract and contract.get('action_parameterization') in ('backflip_entry_v2','backflip_entry_v3'):
        if env.get('episode_length_s')!=5. or env.get('entry_delay_seconds')!=1.:
            raise ValueError('Training environment does not match delayed-entry export contract')
        if 'entry_hold_scale' in env and env['entry_hold_scale']!=contract.get('entry_hold_scale',.4):
            raise ValueError('Training entry hold scale differs from export contract')


def control_contract(raw):
    names=tuple(raw.get('joint_names',JOINT_NAMES))
    if names!=JOINT_NAMES:raise ValueError('Backflip control joint order differs from pinned actor')
    result=dict(schema_version=1,joint_names=list(names))
    for key in ('action_scale','action_clip','stiffness','damping','motor_velocity_x1','motor_torque_y1','motor_torque_y2'):
        value=float(raw[key])
        if not math.isfinite(value) or value<=0:raise ValueError('Invalid backflip control value: '+key)
        result[key]=value
    for key in ('joint_velocity_limits','target_velocity_limits'):
        value=np.asarray(raw[key],dtype=float)
        if value.shape!=(12,) or not np.isfinite(value).all() or np.any(value<=0):raise ValueError('Invalid '+key)
        result[key]=value.tolist()
    if min(result['joint_velocity_limits'])<=result['motor_velocity_x1']:
        raise ValueError('Motor no-load limits must exceed torque plateau speed')
    return result


def configure_playback(module,raw):
    contract=control_contract(raw)
    if tuple(module.JOINT_NAMES)!=JOINT_NAMES:raise ValueError('Playback joint order differs from control contract')
    module.ACTION_SCALE=contract['action_scale']
    module.ACTION_CLIP=contract['action_clip']
    module.KP=contract['stiffness'];module.KD=contract['damping']
    module.MOTOR_VELOCITY_X1=contract['motor_velocity_x1']
    module.MOTOR_VELOCITY_X2=np.array(contract['joint_velocity_limits'])
    module.JOINT_VELOCITY_LIMITS=module.MOTOR_VELOCITY_X2.copy()
    module.TARGET_VELOCITY_LIMIT=np.array(contract['target_velocity_limits'])
    module.MOTOR_TORQUE_Y1=contract['motor_torque_y1'];module.MOTOR_TORQUE_Y2=contract['motor_torque_y2']
    return contract


def motor_torque(raw,velocity,contract,joint_names=JOINT_NAMES):
    """The native training/playback motoring and braking curve, in caller order."""
    limits=np.array([contract['joint_velocity_limits'][JOINT_NAMES.index(n)] for n in joint_names])
    fraction=np.clip((limits-np.abs(velocity))/(limits-contract['motor_velocity_x1']),0.,1.)
    peak=np.where(np.asarray(velocity)*raw>0.,contract['motor_torque_y1'],contract['motor_torque_y2'])
    return np.clip(raw,-peak*fraction,peak*fraction)
