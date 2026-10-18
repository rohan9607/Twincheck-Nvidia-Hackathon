cmake_minimum_required(VERSION 3.8)
project(twincheck_msgs)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DetectedObject.msg"
  "msg/ObjectArray.msg"
  "msg/GraspState.msg"
  "msg/RobotEvent.msg"
  "msg/ExpectedState.msg"
  "msg/CheckResult.msg"
  "msg/ResyncReport.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
