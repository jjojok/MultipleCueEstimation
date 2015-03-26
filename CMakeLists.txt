PROJECT(MultipleCueEstimation)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET(OpenCV_DIR $ENV{OPENCV_DIR})
SET(Eigen_DIR $ENV{EIGEN_DIR})
#SET(BIAS_DIR $ENV{BIAS_ROOT_DIR})

MESSAGE(OpenCV: ${OpenCV_DIR})
MESSAGE(Eigen: ${Eigen_DIR})

FIND_PACKAGE(OpenCV REQUIRED)
#FIND_PACKAGE(BIAS REQUIRED)

SET(BUILD_SHARED_LIBS ON)
SET(CMAKE_BUILD_TYPE Debug)

#INCLUDE_DIRECTORIES(/LineMatchingSourceCode)
#INCLUDE_DIRECTORIES(${BIAS_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

#MESSAGE(libs: ${OpenCV_LIBS})
#MESSAGE(incl: ${OpenCV_INCLUDE_DIRS})

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BIAS_CXX_FLAGS}")
#SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  ${BIAS_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET")

#INCLUDE(${BIAS_USE_FILE})
#LINK_DIRECTORIES( ${BIAS_LINK_DIRECTORIES})

#MESSAGE(BIAS: ${BIAS_DIR})

#MESSAGE(STATUS ${BIAS_LIBRARIES})

#ADD_EXECUTABLE( MultipleCueEstimation main.cpp MCE.cpp MCE.h LineMatcher.cpp LineMatcher.h)
#TARGET_LINK_LIBRARIES( MultipleCueEstimation ${OpenCV_LIBS} ${BIAS_LIBRARIES} ${CMAKE_SOURCE_DIR}/LineMatchingSourceCode/build/libLineMatching.so)
ADD_EXECUTABLE( MultipleCueEstimation main.cpp MultipleCueEstimation.cpp MultipleCueEstimation.h FEstimatorPoints.cpp FEstimatorPoints.h FEstimatorHPoints.cpp FEstimatorHPoints.h FEstimatorHLines.cpp FEstimatorHLines.h FEstimatorHPlanes.cpp FEstimatorHPlanes.h Utility.cpp Utility.h FEstimationMethod.cpp FEstimationMethod.h Statics.h LevenbergMarquardtLines.h LevenbergMarquardtGeneral.h FeatureMatchers.cpp FeatureMatchers.h DataStructs.h)
TARGET_LINK_LIBRARIES( MultipleCueEstimation ${OpenCV_LIBS} )
#MESSAGE(MultipleCueEstimation: ${CMAKE_CURRENT_SOURCE_DIR})

