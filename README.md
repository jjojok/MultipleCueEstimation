MultipleQueEstimation
=====================

Requirements:
	- OpenCV 3.0.0
	- OpenCV 3.0.0 contrib package
	- Eigen 3.2.5
	- Octave 4.0

	- The file "clm.m" needs to be in the same directory as the executable.

Build OpenCV: 
$ cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_EIGEN=ON -D EIGEN_INCLUDE_PATH=<eigen_dir> -D BUILD_DOCS=ON -D OPENCV_EXTRA_MODULES_PATH=<contrib_path> -D BUILD_opencv_legacy=OFF

Execution:
	
	- Without ground truth information:
	MultipleQueEstimation <image 1> <image 2> <select intermediate estimations>

	- With ground truth information from Strecha dataset:
	MultipleQueEstimation <image 1> <image 2> <select intermediate estimations> <

	The flag "select intermediate estimations":
		1: Point in general pos.
		2: Lines on homographies
		4: Points  on homograpies
	or combinations, e.g. 7 = Run all estimations

Output:

	- Debug information to the command line (true errors only if ground truth supplied): 
	
	first image, second image, featureCountCombined, trueFeatureCountCombined, Fgt: trueSampsonErr, Fgt: trueErrorStdDev, Fgt: trueRootSampsonErr, Fgt: trueRootErrorStdDev, refined F: inlierCountCombined, refined F: trueInlierCountCombined, refined F: sampsonErrCombined, refined F: trueSampsonErr, refined F: trueSampsonErrStdDev, refined F: trueRootSampsonErr, refined F: trueRootSampsonErrStdDev, F_Points: featureCountGood, F_Points: trueFeatureCountGood, F_Points: featureCountComplete, F_Points: trueFeatureCountComplete, F_Points: inlierCountOwnGood, F_Points: trueInlierCountOwnGood, F_Points: inlierCountOwnComplete, F_Points: trueInlierCountOwnComplete, F_Points: inlierCountCombined, F_Points: trueInlierCountCombined, F_Points: sampsonErrOwn, F_Points: sampsonErrComplete, F_Points: sampsonErrCombined, F_Points: sampsonErrStdDevCombined, F_Points: trueSampsonErr, F_Points: trueSampsonErrStdDev, F_Points: trueRootSampsonErr, F_Points: trueRootSampsonErrStdDev, F_Points: quality, H_Lines: featureCountGood, H_Lines: trueFeatureCountGood, H_Lines: featureCountComplete, H_Lines: trueFeatureCountComplete, H_Lines: inlierCountOwnGood, H_Lines: trueInlierCountOwnGood, H_Lines: inlierCountOwnComplete, H_Lines: trueInlierCountOwnComplete, H_Lines: inlierCountCombined, H_Lines: trueInlierCountCombined, H_Lines: sampsonErrOwn, H_Lines: sampsonErrComplete, H_Lines: sampsonErrCombined, H_Lines: sampsonErrStdDevCombined, H_Lines: trueSampsonErr, H_Lines: trueSampsonErrStdDev, H_Lines: trueRootSampsonErr, H_Lines: trueRootSampsonErrStdDev, H_Lines: quality, H_Points: featureCountGood, H_Points: trueFeatureCountGood, H_Points: featureCountComplete, H_Points: trueFeatureCountComplete, H_Points: inlierCountOwnGood, H_Points: trueInlierCountOwnGood, H_Points: inlierCountOwnComplete, H_Points: trueInlierCountOwnComplete, H_Points: inlierCountCombined, H_Points: trueInlierCountCombined, H_Points: sampsonErrOwn, H_Points: sampsonErrComplete, H_Points: sampsonErrCombined, H_Points: sampsonErrStdDevCombined, H_Points: trueSampsonErr, H_Points: trueSampsonErrStdDev, H_Points: trueRootSampsonErr, H_Points: trueRootSampsonErrStdDev, H_Points: quality, time (min)

	- Computed fundamental matrices are saved in the follwing files in the directory executed from (if computations were successful):
		F_points.csv: Fundamental matrix from points in general position
		F_H_points.csv: Fundamental matrix from points on homographies
			H1_points.csv: First extracted homographie
			H2_points.csv: Second extracted homographie
		F_H_lines.csv: Fundamental matrix from lines on homographies
			H1_lines.csv: First extracted homographie
			H2_lines.csv: Second extracted homographie
		F_result.csv: Final fundamental matrix result
		F_gt.csv: Ground truth fundamental matrix if given
