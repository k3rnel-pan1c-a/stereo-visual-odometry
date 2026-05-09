# cmake_minimum_required( VERSION 3.5)
# project( HelloSLAM )
# # add_executable(helloSLAM helloSLAM.cpp)
# # add_library(hello SHARED libHelloSLAM.cpp)
# # add_executable(useHello useHello.cpp)
# # target_link_libraries(useHello hello)
# # add_executable(eigenMatrix eigenMatrix.cpp)
# # add_executable(useGeometry useGeometry.cpp)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find_package(OpenCV REQUIRED)

# include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Pangolin_INCLUDE_DIRS})
# include_directories("/usr/include/eigen3")

# add_executable(stereoVision stereoVision.cpp)

cmake_minimum_required(VERSION 3.10)
project(stereo_vo_cpp)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
)

add_executable(stereo_vo
    src/main.cpp
    src/KittiDataset.cpp
    src/Calibration.cpp
    src/FeatureExtractor.cpp
    src/StereoMatcher.cpp
    src/MotionEstimator.cpp
    src/Trajectory.cpp
    src/Viewer.cpp
)

target_link_libraries(stereo_vo
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
)