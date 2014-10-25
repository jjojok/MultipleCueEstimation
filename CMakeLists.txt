project(MultipleCueEstimation)
cmake_minimum_required(VERSION 2.8)
set (OpenCV_DIR $ENV{OPENCV_DIR})
set (Eigen_DIR $ENV{EIGEN_DIR})
set (BIAS_DIR $ENV{BIAS_ROOT_DIR})

#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/LBD_Descriptor)
#include(${CMAKE_CURRENT_SOURCE_DIR}/LBD_Descriptor/CMakeLists.txt)

MESSAGE(OpenCV: ${OpenCV_DIR})
MESSAGE(Eigen: ${Eigen_DIR})
MESSAGE(BIAS: ${BIAS_DIR})

find_package( OpenCV REQUIRED )
add_executable( MultipleCueEstimation main.cpp MCE.cpp MCE.h)
target_link_libraries( MultipleCueEstimation ${OpenCV_LIBS} )

