cmake_minimum_required(VERSION 3.8)
project(lilac_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ActiveLanguage.msg"
  "msg/LatentZ.msg"
  "msg/LILACState.msg"
  "msg/EEDelta.msg"
  "msg/QPosCommand.msg"
  "msg/ContactInfo.msg"
  "msg/Rumble.msg"
  "srv/ApplyUtterance.srv"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
