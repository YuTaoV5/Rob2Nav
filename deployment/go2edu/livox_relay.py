"""Explicit ROS1 message namespace adapter; requires identical wire schemas."""
def main():
    import io,rospy
    from livox_ros_driver2.msg import CustomMsg as DriverMsg
    from lio.msg import CustomMsg as LioMsg
    if DriverMsg._md5sum!=LioMsg._md5sum:raise RuntimeError('Livox/LIO wire schemas differ: do not relay; inspect pinned message definitions')
    rospy.init_node('rob2nav_livox_relay')
    pub=rospy.Publisher('/rob2nav/livox_custom',LioMsg,queue_size=5)
    def callback(m):
        buffer=io.BytesIO();m.serialize(buffer);output=LioMsg();output.deserialize(buffer.getvalue());pub.publish(output)
    rospy.Subscriber('/livox/lidar',DriverMsg,callback,queue_size=5,tcp_nodelay=True)
    rospy.spin()
if __name__=='__main__':main()
