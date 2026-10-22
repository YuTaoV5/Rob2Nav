"""Validate an observed-entry diagnostic snapshot without modifying a simulator."""
import numpy as np


def validate(raw,names):
    if raw.get('schema_version')!=1:raise ValueError('Unsupported observed-entry schema')
    order=raw.get('joint_names',[])
    if len(order)!=12 or len(set(order))!=12 or set(order)!=set(names):raise ValueError('Observed joint mapping is not a bijection')
    result={}
    for key,size in [('joint_position',12),('joint_velocity',12),('root_position',3),('root_quaternion',4),
                     ('root_linear_velocity_world',3),('root_angular_velocity_body',3)]:
        v=np.asarray(raw[key],dtype=float)
        if v.shape!=(size,) or not np.isfinite(v).all():raise ValueError('Invalid observed '+key)
        result[key]=v[[order.index(n) for n in names]] if key.startswith('joint_') else v
    if abs(np.linalg.norm(result['root_quaternion'])-1.)>1e-5:raise ValueError('Observed quaternion is not normalized')
    return result
