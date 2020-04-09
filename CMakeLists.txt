cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(camera_fusion)

set(OpenCV_DIR /usr/local/share/OpenCV)

find_package(OpenCV 3.4 REQUIRED)
set(OpenCV_INCLUDE_DIRS
  /usr/local/include
  /usr/local/include/opencv2
)

set(OpenCV_LIB_DIR
  /usr/local/lib
)
find_package(OpenCV 3.4 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Executable for create matrix exercise
add_executable (3D_object_tracking src/camFusion.cpp src/FinalProject_Camera.cpp src/lidarData.cpp src/matching2D.cpp src/objectDetection2D.cpp)
target_link_libraries (3D_object_tracking ${OpenCV_LIBRARIES})