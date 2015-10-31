#ifndef STATICS_H
#define STATICS_H
#define _USE_MATH_DEFINES

#include <math.h>

//Show debug messages
#define LOG_DEBUG false
//Create debug images
#define CREATE_DEBUG_IMG true
//Show debug images
#define SHOW_DEBUG_IMG false
//Save debug images
#define SAVE_DEBUG_IMG true

//Program parameters for selecting computation mathods
#define F_Reined 0
#define F_FROM_POINTS 1
#define F_FROM_LINES_VIA_H 2
#define F_FROM_POINTS_VIA_H 4

//General:
//Threshold to determine "inlier", compared with square root of sampson distance, e.g. by RANSAC
#define INLIER_THRESHOLD 3.0
//Initial threshold to determine "inlier" by LM in refinement step
#define INLIER_LM_THRESHOLD 100.0
//Weights for fundamental matrix quality function
#define QI 4.0
#define QE 1.0
#define QS 0.01
//Hard limit on number of computations for ransac & lmeds
#define MAX_NUM_COMPUTATIONS 15000
//Max number of numerical refinement steps after a first homographie was found
#define NUMERICAL_OPTIMIZATION_MAX_ITERATIONS 100
//max Percentage of error changing for error to be considered as stable
#define MAX_ERROR_CHANGE 0.005
//max features changing for error to be considered as stable
#define MAX_FEATURE_CHANGE 3
//How close two eigenvalues have to be to be considered equal
#define MARGIN 0.4
//Number of attempts to compute a second Homography if it is equal to the forst one
#define MAX_H2_ESTIMATIONS 80
//Max trys to find non linear point in remaining matches
#define MAX_POINT_SEARCH 1000
//Minimal number of matches to stop numerical optimization
#define NUMERICAL_OPTIMIZATION_MIN_MATCHES 8
//Rans5ac confidence
#define RANSAC_CONFIDENCE 0.999
//LM iterations for homographies
#define MAX_LM_ITER 200

//Points:
//Sift features
#define SIFT_FEATURE_COUNT 800
//Min Sift distance to be a "good" match
#define SIFT_MIN_DIST 0.06
//Factor of min Sift distance from all generated points where correspondence is still acceptable
#define SIFT_MIN_DIST_FACTOR 3.0
//Number of point correspondencies per homography estimation
#define NUM_POINT_CORRESP 4//6
//distance from point to line for 3 points below they are colinear
#define MAX_COLINEAR_DIST 4.0

//Number of segements for image pyramid
#define OCTAVES 2
//Scaling factor per segement
#define SCALING 2
//minimal line lenght = sqrt(width²+height²)*MIN_LENGTH_FACTOR
#define MIN_LENGTH_FACTOR 0.004;
//Max angle between matched lines
#define MAX_LINE_ANGLE M_PI/4.0
//Percentage of lines thought to be outlaiers (outliers are also lines which lie on another plane in 3D)
#define HOMOGRAPHY_OUTLIERS 0.80
//Number of correspondencies per homography estimation
#define NUM_LINE_CORRESP 4//6
//Max hemming distance of binary matchig
#define MAX_HEMMING_DIST 12
//Min hemming distance of binary matchig
#define MIN_HEMMING_DIST 5
//Angle difference blow lines being parallel
#define MAX_ANGLE_DIFF M_PI/20.0

#endif // STATICS_H
