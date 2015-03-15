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
#define F_FROM_LINES 2
#define F_FROM_PLANES 4

//Points:
//Sift features
#define SIFT_FEATURE_COUNT 800
//Min Sift distance to be a "good" match
#define SIFT_MIN_DIST 0.06
//Factor of min Sift distance from all generated points where correspondence is still acceptable
#define SIFT_MIN_DIST_FACTOR 3.0

//Lines:
//Number of segements for image pyramid
#define OCTAVES 2
//Scaling factor per segement
#define SCALING 2
//minimal line lenght = sqrt(width²+height²)*MIN_LENGTH_FACTOR
#define MIN_LENGTH_FACTOR 0.015;
//Max angle between matched lines
#define MAX_LINE_ANGLE M_PI/4.0
//defines number of subsets which are randomly picked to compute a Homography each. Homographies = Number of matches*NUM_OF_PAIR_SUBSETS_FACTOR
#define NUM_LINE_PAIR_SUBSETS_FACTOR 25.0//0.25
//Number of subsets for Homography computations
#define NUM_LINE_PAIR_SUBSETS 500
//Number of correspondencies per homography estimation
#define NUM_CORRESP 6
//Max hemming distance of binary matchig
#define MAX_HEMMING_DIST 20
//Min hemming distance of binary matchig
#define MIN_HEMMING_DIST 5
//Factor for selecting wrong matches in refinement step after first homography estimation
#define OUTLIER_THESHOLD_FACTOR 2.0
//How close a H has to be at unity to be teated as unity
#define MARGIN 0.1
//Number of attempts to compute a second Homography if it is equal to the forst one
#define MAX_H2_ESTIMATIONS 20
//Max Angle difference for lines being parallel
#define MAX_ANGLE_DIFF M_PI/30.0
//Max squared distance of projected lines to be considered as a correct projection (pixel)
#define MAX_PROJ_DIST 1.0

//Error Estimation:
//Number of points for error measure between two fundamental matrices
#define NUM_SAMPLES_F_COMARATION 50000
//Max trys per draw to find a point who's epipolar line intersects the other image
#define MAX_SAMPLE_TRYS 1000

#endif // STATICS_H
