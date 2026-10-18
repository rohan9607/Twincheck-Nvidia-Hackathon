# TwinCheck ROS 2 workspace image (multi-arch: works on laptops x86_64 and the DGX Spark arm64).
#   docker build -f docker/ros.Dockerfile -t twincheck-ros .
#   docker run --rm -it --net=host -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID -v $PWD:/twincheck twincheck-ros
# Detector inference on the Spark GPU runs in the NVIDIA PyTorch container instead (see docs/SPARK.md);
# this image is for the ROS nodes, MoveIt and the FANUC driver.
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-cv-bridge ros-jazzy-moveit ros-jazzy-moveit-py ros-jazzy-rmw-cyclonedds-cpp \
      python3-pip python3-opencv python3-colcon-common-extensions git tmux \
    && rm -rf /var/lib/apt/lists/*

COPY twincheck_core /tmp/twincheck_core
RUN pip3 install /tmp/twincheck_core "psycopg[binary]" ultralytics pyyaml

WORKDIR /twincheck
ENV TWINCHECK_CONFIG=/twincheck/config/twincheck.yaml RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
CMD ["bash", "-lc", "source /opt/ros/jazzy/setup.bash && cd ros2_ws && colcon build --symlink-install && source install/setup.bash && exec bash"]
