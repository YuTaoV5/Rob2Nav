"""Sagittal reflection of Go2 state; angular vectors are axial, not polar."""
import numpy as np


def swapped(name):
    for a,b in [('FL_','FR_'),('FR_','FL_'),('RL_','RR_'),('RR_','RL_')]:
        if name.startswith(a):return b+name[len(a):]
    return name


def joint_map(names):
    expected={f'{leg}_{part}_joint' for leg in ['FL','FR','RL','RR'] for part in ['hip','thigh','calf']}
    if len(names)!=12 or set(names)!=expected:raise ValueError('Expected all twelve named Go2 joints')
    return np.array([names.index(swapped(n)) for n in names]),np.array([-1. if '_hip_' in n else 1. for n in names])


def observation_map(names):
    p,s=joint_map(names);indices=np.arange(69);sign=np.ones(69)
    for start in [0,12,33,45,57]:indices[start:start+12]=start+p;sign[start:start+12]=s
    sign[24:30]=[1.,-1.,-1.,1.,1.,-1.]
    sign[30:33]=[-1.,1.,-1.]
    return indices,sign


def motion(data,joints,bodies):
    p,s=joint_map(joints);bp=[bodies.index(swapped(n)) for n in bodies]
    result={k:np.array(v,copy=True) for k,v in data.items()}
    for k in ['joint_pos','joint_vel']:result[k]=data[k][...,p]*s
    for k in ['body_pos_w','body_lin_vel_w']:result[k]=data[k][:,bp,:]*[1.,-1.,1.]
    result['body_quat_w']=data['body_quat_w'][:,bp,:]*[1.,-1.,1.,-1.]
    result['body_ang_vel_w']=data['body_ang_vel_w'][:,bp,:]*[-1.,1.,-1.]
    return result
