cmake_minimum_required(VERSION 2.8)
project(radar_lidar_fusion_EKF)

# for advanced features of c++
add_definitions(-std=c++11)

# Set the directory of binaries
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) 

# Include the headers into the project.
include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB SOURCES "src/*.cpp")

add_executable(radar_lidar_fusion_EKF ${SOURCES})
