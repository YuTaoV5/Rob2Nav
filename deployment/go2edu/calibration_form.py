"""Interactive offline form; no robot, ROS or sensor connection."""
import argparse,ipaddress,json,math
from pathlib import Path
import numpy as np
from prepare import transform

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise SystemExit('Output exists; choose a new filename to preserve calibration history')
    def matrix(name):
        print(name+': T_A_B maps B points into A; xyz in metres, roll/pitch/yaw in degrees.')
        xyz=[float(x) for x in input('x y z: ').split()]
        r,p,y=[math.radians(float(x)) for x in input('roll pitch yaw (Rz*Ry*Rx): ').split()]
        rx=np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]])
        ry=np.array([[math.cos(p),0,math.sin(p)],[0,1,0],[-math.sin(p),0,math.cos(p)]])
        rz=np.array([[math.cos(y),-math.sin(y),0],[math.sin(y),math.cos(y),0],[0,0,1]])
        t=np.eye(4);t[:3,:3]=rz@ry@rx;t[:3,3]=xyz
        return transform(t).tolist()
    c=dict(reviewed=False,robot='Go2 EDU',T_base_lidar=matrix('T_base_lidar (measured mounting)'),T_imu_lidar=matrix('T_imu_lidar (sensor internal calibration, not mounting angle)'))
    c['initial_base_height_m']=float(input('Standing body origin height above level ground (metres): '))
    c['sdk_interface']=input('Go2 SDK network interface (NOT the lidar USB adapter): ').strip()
    for key in ('lidar_ip','host_lidar_ip'):c[key]=str(ipaddress.IPv4Address(input(key+': ').strip()))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(c,indent=2))
    print('Saved unreviewed calibration. Verify axes/directions/units/physical measurements before reviewed=true:',a.out)

if __name__=='__main__':main()
