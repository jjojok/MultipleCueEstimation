MultipleQueEstimation
=====================

Used Libs:
	OpenCV 3.0.0 - latest

Cmake Open CV: 
	cmake -D CMAKE_BUILD_TYPE=RELEASE -D BUILD_EXAMPLES=ON -D WITH_EIGEN=ON -D EIGEN_INCLUDE_PATH=<path to eigen> -D BUILD_DOCS=ON -D OPENCV_EXTRA_MODULES_PATH=<path to opencv_contrib>/modules -D BUILD_opencv_legacy=OFF -D CMAKE_INSTALL_PREFIX=<path to opencv install> ..

Add global variables (/etc/profile):
	export OPENCV_DIR=<path to opencv build>
	export EIGEN_DIR=<path to eigen home>
