FROM ros:humble-ros-base

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions \
    python3-pip \
    ros-humble-launch-ros \
    ros-humble-sensor-msgs \
    ros-humble-std-srvs \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir numpy google-genai \
    && python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /workspace/project
COPY package ./package
COPY data/language ./data/language
COPY ros_ws ./ros_ws
COPY docker ./docker

ENV LILAC_PROJECT_ROOT=/workspace/project
ENV PYTHONPATH=/workspace/project/package
ENV ROS_LOCALHOST_ONLY=1

RUN source /opt/ros/humble/setup.bash \
    && cd /workspace/project/ros_ws \
    && colcon build --symlink-install

RUN chmod +x /workspace/project/docker/entrypoint.sh
ENTRYPOINT ["/workspace/project/docker/entrypoint.sh"]
CMD ["ros2", "launch", "lilac_ros", "lilac_demo.launch.py"]
