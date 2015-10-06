cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(lidar_stereo)

find_package(OpenCV REQUIRED)
find_package(PCL 1.3 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (lidar_stereo lidar_stereo.cpp)
target_link_libraries (lidar_stereo ${OpenCV_LIBS} ${PCL_LIBRARIES} ${COVIS_LIBRARIES})


