"""Publish one real Noetic goal, relative to fresh measured body pose."""
import argparse,math
def main():
    import rospy,tf.transformations as tfm
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped
    p=argparse.ArgumentParser();mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--relative',nargs=2,type=float,metavar=('FORWARD_M','LEFT_M'));mode.add_argument('--absolute',nargs=2,type=float,metavar=('X_M','Y_M'));a=p.parse_args()
    values=a.relative if a.relative is not None else a.absolute
    if not all(math.isfinite(x) for x in values):p.error('Goal must be finite')
    rospy.init_node('rob2nav_set_goal',anonymous=True)
    odom=rospy.wait_for_message('/rob2nav/body_pose',Odometry,timeout=5)
    if odom.header.frame_id!='nav_odom' or not 0<=rospy.Time.now().to_sec()-odom.header.stamp.to_sec()<.25:raise RuntimeError('Fresh nav_odom body pose required')
    q=odom.pose.pose.orientation;yaw=tfm.euler_from_quaternion([q.x,q.y,q.z,q.w])[2];pos=odom.pose.pose.position
    if a.relative is not None:
        forward,left=a.relative;x=pos.x+math.cos(yaw)*forward-math.sin(yaw)*left;y=pos.y+math.sin(yaw)*forward+math.cos(yaw)*left
    else:x,y=a.absolute
    goal=PoseStamped();goal.header.frame_id='nav_odom';goal.header.stamp=rospy.Time.now();goal.pose.position.x=x;goal.pose.position.y=y;goal.pose.position.z=pos.z;goal.pose.orientation.w=1.
    pub=rospy.Publisher('/move_base_simple/goal',PoseStamped,queue_size=1)
    deadline=rospy.Time.now()+rospy.Duration(5)
    while pub.get_num_connections()<2 and rospy.Time.now()<deadline:rospy.sleep(.05)
    if pub.get_num_connections()<2:raise RuntimeError('Expected planner and velocity-gate subscribers; start both before setting a goal')
    goal.header.stamp=rospy.Time.now();pub.publish(goal);rospy.sleep(.3)
    print('Published nav_odom goal:',x,y,'relative mode' if a.relative is not None else 'absolute mode')
if __name__=='__main__':main()
