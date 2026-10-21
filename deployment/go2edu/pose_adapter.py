"""Noetic: transform real LIO poses, never create synthetic robot odometry."""
import argparse,copy,math,threading
import numpy as np
from prepare import load_calibration,shift_cloud_z

def main():
    import rospy,tf.transformations as tfm,tf2_ros
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import TransformStamped
    from sensor_msgs.msg import PointCloud2,PointField
    p=argparse.ArgumentParser();p.add_argument('--calibration',required=True);a,_=p.parse_known_args()
    c=load_calibration(a.calibration);rospy.init_node('rob2nav_pose_adapter')
    pubs=[rospy.Publisher('/rob2nav/body_pose',Odometry,queue_size=5),rospy.Publisher('/rob2nav/lidar_pose',Odometry,queue_size=5)]
    cloud_pub=rospy.Publisher('/rob2nav/cloud_world',PointCloud2,queue_size=2)
    broad=tf2_ros.TransformBroadcaster();lock=threading.Lock();offset=[None];frame=[None];previous=[None]
    def callback(m,index):
        if not m.header.frame_id or m.header.frame_id=='nav_odom':return
        stamp=m.header.stamp.to_sec();age=rospy.Time.now().to_sec()-stamp
        if not 0<=age<.5:return
        pose=m.pose.pose;q=pose.orientation;v=np.array([q.x,q.y,q.z,q.w])
        if not np.isfinite(v).all() or abs(np.linalg.norm(v)-1)>.01:return
        matrix=tfm.quaternion_matrix(v);matrix[:3,3]=[pose.position.x,pose.position.y,pose.position.z]
        if not np.isfinite(matrix).all():return
        with lock:
            if index==0 and offset[0] is None:offset[0]=c['initial_base_height_m']-matrix[2,3];frame[0]=m.header.frame_id
            if offset[0] is None:return
            if m.header.frame_id!=frame[0]:return
            dz=offset[0]
        if index==1:matrix=matrix@c['T_imu_lidar']
        matrix[2,3]+=dz
        msg=copy.deepcopy(m);msg.header.frame_id='nav_odom';msg.child_frame_id='body' if index==0 else 'lidar'
        pose=msg.pose.pose;pose.position.x,pose.position.y,pose.position.z=matrix[:3,3]
        q=tfm.quaternion_from_matrix(matrix);pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=q
        # Sensor twist/covariance are not valid after a lever-arm transform.
        # SCAN consumes sensor pose; do not advertise copied IMU twist as lidar twist.
        if index==1:msg.twist=Odometry().twist;msg.pose.covariance=[0.0]*36
        else:
            # This pinned LIO leaves body twist at zero. SCAN expects world
            # velocity; estimate it from actual body poses (no command odometry).
            with lock:
                old=previous[0];previous[0]=(stamp,matrix[:3,3].copy())
            if old is None:return
            dt=stamp-old[0]
            if not .002<dt<.25:return
            velocity=(matrix[:3,3]-old[1])/dt
            if np.linalg.norm(velocity)>2.:return
            msg.twist.twist.linear.x,msg.twist.twist.linear.y,msg.twist.twist.linear.z=velocity
        pubs[index].publish(msg)
        if index==0:
            t=TransformStamped();t.header=msg.header;t.child_frame_id=m.header.frame_id
            t.transform.translation.z=dz;t.transform.rotation.w=1.;broad.sendTransform(t)
    def cloud(m):
        with lock:dz=offset[0];source=frame[0]
        if dz is None or m.header.frame_id!=source:return
        if not 0<=rospy.Time.now().to_sec()-m.header.stamp.to_sec()<.5:return
        field=next((f for f in m.fields if f.name=='z' and f.datatype==PointField.FLOAT32 and f.count==1),None)
        if field is None:return
        try:
            data=shift_cloud_z(m.data,m.width,m.height,m.point_step,m.row_step,field.offset,dz,m.is_bigendian)
        except ValueError:return
        msg=copy.deepcopy(m);msg.data=data;msg.header.frame_id='nav_odom';cloud_pub.publish(msg)
    rospy.Subscriber('/LIO/odom_vehicle',Odometry,lambda m:callback(m,0),queue_size=5,tcp_nodelay=True)
    rospy.Subscriber('/LIO/odom_imu',Odometry,lambda m:callback(m,1),queue_size=5,tcp_nodelay=True)
    rospy.Subscriber('/LIO/clouds_lidar',PointCloud2,cloud,queue_size=2,tcp_nodelay=True)
    rospy.spin()

if __name__=='__main__':main()
