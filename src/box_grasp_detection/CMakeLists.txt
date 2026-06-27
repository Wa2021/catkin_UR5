cmake_minimum_required(VERSION 3.0.2)
project(box_grasp_detection)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  pcl_ros
  pcl_conversions
  message_generation
)

find_package(PCL REQUIRED)

add_message_files(
  FILES
  BoxGrasp.msg
  BoxGraspArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    visualization_msgs
    pcl_ros
    pcl_conversions
    message_runtime
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# Box detection node
add_executable(box_detector_node 
  src/box_detector_node.cpp
  src/box_detector.cpp
)
add_dependencies(box_detector_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(box_detector_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

install(TARGETS box_detector_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(PROGRAMS
  scripts/single_frame_detector.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
