"""Generate Noetic deployment configuration from measured, reviewed extrinsics.

T_A_B maps B coordinates into A coordinates. Never populates unknown calibration.
Run from this directory or load with importlib; ROS is not required here.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
import yaml

SCAN_PIN='348e8a590a50a5a6bbab8d8c6dcfd171f009be26'
LIO_PIN='1d79af77f3d9747ea57ef52a9b01d326a8ec561a'

def transform(value):
    t=np.asarray(value,dtype=float)
    if t.shape!=(4,4) or not np.isfinite(t).all():raise ValueError('Expected finite 4x4 transform')
    if not np.allclose(t[3],[0,0,0,1],atol=1e-6):raise ValueError('Invalid homogeneous bottom row')
    r=t[:3,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=1e-5) or abs(np.linalg.det(r)-1)>1e-5:raise ValueError('Rotation must be orthonormal and right handed')
    return t

def load_calibration(path):
    c=json.loads(Path(path).read_text())
    if c.get('reviewed') is not True:raise ValueError('Fill measured calibration and set reviewed=true after independent checks')
    c['T_base_lidar']=transform(c.get('T_base_lidar'))
    c['T_imu_lidar']=transform(c.get('T_imu_lidar'))
    h=c.get('initial_base_height_m')
    if not isinstance(h,(float,int)) or not math.isfinite(h) or not .15<h<.7:raise ValueError('Measure initial standing base height in metres')
    return c

def shift_cloud_z(data,width,height,point_step,row_step,z_offset,dz,bigendian=False):
    if width<1 or height<1 or point_step<12 or not 0<=z_offset<=point_step-4 or row_step<width*point_step or len(data)!=height*row_step:raise ValueError('Invalid cloud layout')
    raw=bytearray(data)
    z=np.ndarray((height,width),dtype='>f4' if bigendian else '<f4',buffer=raw,offset=z_offset,strides=(row_step,point_step))
    z[:]+=dz
    return bytes(raw)

def generate(calibration,lio,scan,out):
    import subprocess
    for path,pin in ((lio,LIO_PIN),(scan,SCAN_PIN)):
        if subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()!=pin:raise ValueError('Upstream checkout differs: '+str(path))
    c=load_calibration(calibration);out.mkdir(parents=True,exist_ok=True)
    sensor=yaml.safe_load((lio/'yaml/sensors/livox.yaml').read_text())
    lidar_body=np.linalg.inv(c['T_base_lidar'])
    sensor['offset'].update(imu_t_lidar=c['T_imu_lidar'][:3,3].tolist(),imu_R_lidar=c['T_imu_lidar'][:3,:3].tolist(),lidar_t_body=lidar_body[:3,3].tolist(),lidar_R_body=lidar_body[:3,:3].tolist())
    sensor['topic_sub']['lidar_topic_name']='/rob2nav/livox_custom'
    runtime=yaml.safe_load((lio/'yaml/runtime/mapping.yaml').read_text())
    runtime['elevator']['enable']=False;runtime['wheel']['wheel_enable']=False
    runtime['pub']['high_frequency_odom']=True
    # Do not change near-field masking automatically: it must match the assembly.
    for name,data in [('sensors/rob2nav.yaml',sensor),('runtime/rob2nav.yaml',runtime),('root_rob2nav.yaml',dict(sensor_config='sensors/rob2nav.yaml',runtime_config='runtime/rob2nav.yaml',logging_config='logging/default.yaml'))]:
        (lio/'yaml'/name).write_text(yaml.safe_dump(data,sort_keys=False))
    # Pose adapter supplies the actual lidar pose. Disable upstream hardcoded
    # sensor extrinsics and do not launch simulator or synthetic joint states.
    launch='''<launch>
  <arg name="use_sim_time" default="false"/>
  <param name="use_sim_time" value="$(arg use_sim_time)"/>
  <param name="body_pose_topic" value="/rob2nav/body_pose"/>
  <include file="$(find scan_planner)/launch/advanced_param.xml">
    <arg name="is_real_world" value="true"/><arg name="navi_mode" value="1"/>
    <arg name="sensor_type" value="lidar"/>
    <arg name="body_pose_topic" value="/rob2nav/body_pose"/>
    <arg name="sensor_pose_topic" value="/rob2nav/lidar_pose"/>
    <arg name="cloud_topic" value="/rob2nav/cloud_world"/>
    <arg name="cloud_is_world" value="true"/>
    <arg name="depth_topic" value="/unused_depth"/>
    <arg name="cx" value="0"/><arg name="cy" value="0"/>
    <arg name="fx" value="1"/><arg name="fy" value="1"/>
    <arg name="max_vel" value="0.20"/><arg name="max_acc" value="0.20"/>
    <arg name="planning_horizon" value="3.0"/>
  </include>
  <param name="scan_planner_node/grid_map/need_extrinsic" value="false"/>
  <param name="scan_planner_node/grid_map/frame_id" value="nav_odom"/>
  <param name="closed_loop_controller/max_vx" value="0.20"/>
  <param name="closed_loop_controller/max_vy" value="0.10"/>
  <param name="closed_loop_controller/max_vyaw" value="0.30"/>
  <node pkg="scan_planner" type="closed_loop_controller" name="closed_loop_controller" output="screen">
    <remap from="cmd_vel" to="/rob2nav/cmd_vel_raw"/>
  </node>
</launch>
'''
    (out/'scan_go2edu.launch').write_text(launch)
    print('Generated:',out/'scan_go2edu.launch','; LIO root_rob2nav.yaml. No SDK or robot connection made.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--calibration',type=Path,required=True);p.add_argument('--lio',type=Path,required=True);p.add_argument('--scan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();generate(a.calibration,a.lio,a.scan,a.out)
