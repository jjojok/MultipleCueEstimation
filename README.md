MultipleQueEstimation
=====================

Used Libs:

For MultipleCueExtimation:
	OpenCV 2.49

For LineMatching:
	BIAS Nighly build 2014-10-25: http://www.mip.informatik.uni-kiel.de/tiki-index.php?page=BIAS
		Disable all dependencies except USE_OPENCV
		LAPACK 3.5.0: http://www.netlib.org/lapack/ (maybe also available via apt)
			gfortran: sudo apt-get install gfortran
		GLEW 1.11.0: http://glew.sourceforge.net/ (maybe also available via apt)
		UUID Lib: sudo apt-get install uuid-devel
	ARPACK++ & ARPACK: sudo apt-get install libarpack++2-dev
		Replace header files in /usr/include/arpack++ with files in MultipleCueEstimation/Dependencies/arpack++
		BLAS: sudo apt-get install libblas-dev
	SuperLU: sudo apt-ger install libsuperlu3-dev 
	

Cmake Open CV: cmake -D CMAKE_BUILD_TYPE=RELEASE -D BUILD_EXAMPLES=ON -D WITH_EIGEN=ON -D EIGEN_INCLUDE_PATH=/home/cvstudent/Eigen -D BUILD_DOCS=ON ..

Add global variables (/etc/profile):

export GLEW_HOME=<path to glew build>
export OPENCV_DIR=<path to opencv build>
export EIGEN_DIR=<path to eigen home>
export LAPACK_HOME=<path to lapack build>
export BIAS_ROOT_DIR=<path to BIAS build>
