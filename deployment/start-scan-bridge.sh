#!/usr/bin/env bash
set -e
source /opt/ros/jazzy/setup.bash
source /root/gpufree-data/scan-reproduction/upstream/install/setup.bash
export ROS_DOMAIN_ID=88 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
cd /root/gpufree-data/Rob2Nav
exec /usr/bin/python3 -m skills.navigation.scan_ros
