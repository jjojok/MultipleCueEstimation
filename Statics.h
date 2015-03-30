#ifndef STATICS_H
#define STATICS_H
#define _USE_MATH_DEFINES

#include <math.h>

//Show debug messages
#define LOG_DEBUG true
//Show debug images
#define VISUAL_DEBUG true

//Program parameters for selecting computation mathods
#define F_FROM_POINTS 1
#define F_FROM_LINES_VIA_H 2
#define F_FROM_POINTS_VIA_H 4
#define F_FROM_PLANES_VIA_H 8

//General:
//Hard limit on number of computations for ransac & lmeds
#define MAX_NUM_COMPUTATIONS 15000
//Max number of refinement steps after a first homographie was found
#define MAX_REFINEMENT_ITERATIONS 300
//Max squared distance of lines to be considered as a correct projection (pixel)
#define MAX_TRANSFER_DIST 1.25
//max Percentage of error changing for error to beconsidered as stable
#define MAX_ERROR_CHANGE 0.1
//error for iteration to stop even if error still changes
#define MIN_ERROR 1.0e-15
//How close two values have to be to be considered equal
#define MARGIN 0.1
//Number of attempts to compute a second Homography if it is equal to the forst one
#define MAX_H2_ESTIMATIONS 50
//Max trys to find non linear point in remaining matches
#define MAX_POINT_SEARCH 1000

//Points:
//Sift features
#define SIFT_FEATURE_COUNT 800
//Min Sift distance to be a "good" match
#define SIFT_MIN_DIST 0.06
//Factor of min Sift distance from all generated points where correspondence is still acceptable
#define SIFT_MIN_DIST_FACTOR 3.0
//Ransac thredshold
#define RANSAC_THREDHOLD 1.25
//Ransac confidence
#define RANSAC_CONFIDENCE 0.9999
////Refinement thredshold
//#define REFINEMENT_THREDHOLD 3.0
//Number of point correspondencies per homography estimation
#define NUM_POINT_CORRESP 4//6
//Max distance from point to line for 3 points to be colinear
#define MAX_COLINEAR_DIST 5.0

//Number of segements for image pyramid
#define OCTAVES 2
//Scaling factor per segement
#define SCALING 2
//minimal line lenght = sqrt(width²+height²)*MIN_LENGTH_FACTOR
#define MIN_LENGTH_FACTOR 0.006;
//Max angle between matched lines
#define MAX_LINE_ANGLE M_PI/4.0
//Ransac confidence
#define CONFIDENCE 0.999
//Percentage of lines thought to be outlaiers (outliers are also lines which lie on another plane in 3D)
#define HOMOGRAPHY_OUTLIERS 0.8
//Number of correspondencies per homography estimation
#define NUM_LINE_CORRESP 4//6
//Max hemming distance of binary matchig
#define MAX_HEMMING_DIST 8
//Min hemming distance of binary matchig
#define MIN_HEMMING_DIST 5
//Factor for selecting wrong matches in refinement step after first homography estimation
#define OUTLIER_THESHOLD_FACTOR 2.0
//Max Angle difference for lines being parallel
#define MAX_ANGLE_DIFF M_PI/50.0

//Error Estimation:
//Number of points for error measure between two fundamental matrices
#define NUM_SAMPLES_F_COMARATION 2000
//Max trys per draw to find a point who's epipolar line intersects the other image
#define MAX_SAMPLE_TRYS 1000

#endif // STATICS_H
