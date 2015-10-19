PROJECT(SevenpointLevenbergMarquardt)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

INCLUDE_DIRECTORIES(/usr/local/include/octave-4.0.0)
INCLUDE_DIRECTORIES(/usr/local/include/octave-4.0.0/octave)

SET(CMAKE_BUILD_TYPE Release)

LINK_DIRECTORIES(/usr/local/lib/octave/4.0.0)

#ADD_EXECUTABLE( SevenpointLevenbergMarquardt SevenpointLevenbergMarquardt.cpp SevenpointLevenbergMarquardt.h)
add_library(SevenpointLevenbergMarquardt SHARED SevenpointLevenbergMarquardt.cpp SevenpointLevenbergMarquardt.h)
TARGET_LINK_LIBRARIES( SevenpointLevenbergMarquardt liboctave.so liboctinterp.so)



