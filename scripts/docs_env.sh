#!/usr/bin/env bash
# Source this file from any directory; never starts a service or a training job.
export ROB2NAV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROB2NAV_WORKSPACE="${ROB2NAV_WORKSPACE:-$(dirname "$ROB2NAV_ROOT")}"
export ROB2NAV_UPSTREAMS="${ROB2NAV_UPSTREAMS:-$ROB2NAV_WORKSPACE/Rob2Nav-upstreams}"
export ROB2NAV_ENVS="${ROB2NAV_ENVS:-$ROB2NAV_WORKSPACE/envs}"
export ROB2NAV_DATA="${ROB2NAV_DATA:-$ROB2NAV_ROOT/outputs/reproduction-data}"
export CPU_PY="${CPU_PY:-$ROB2NAV_ENVS/go2-skills-cpu/bin/python}"
export WEB_PY="${WEB_PY:-$ROB2NAV_ENVS/rob2nav-studio/bin/python}"
export ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-$ROB2NAV_WORKSPACE/isaacsim}"
export ISAAC_PY="${ISAAC_PY:-$ISAAC_SIM_ROOT/python.sh}"
export ISAACLAB_ROOT="${ISAACLAB_ROOT:-$ROB2NAV_WORKSPACE/IsaacLab}"
export SCAN_WS="${SCAN_WS:-$ROB2NAV_WORKSPACE/scan-reproduction/upstream}"
export ROS_DISTRO="${ROS_DISTRO:-jazzy}"
export ROS_PY="${ROS_PY:-/usr/bin/python3}"
export PYTHONPATH="$ROB2NAV_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
cd "$ROB2NAV_ROOT" || return 1
