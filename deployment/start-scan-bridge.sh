#!/usr/bin/env bash
set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/docs_env.sh"
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "$SCAN_WS/install/setup.bash"
export ROS_DOMAIN_ID=88 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
exec "$ROS_PY" -m skills.navigation.scan_ros
