#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
source /workspace/project/ros_ws/install/setup.bash

exec "$@"
