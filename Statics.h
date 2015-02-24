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
#define SIFT_MIN_DIST 0.05
//Factor of min Sift distance from all generated points where correspondence is still acceptable
#define SIFT_MIN_DIST_FACTOR 4.0

//Lines:
//Number of segements for image pyramid
#define OCTAVES 2
//Scaling factor per segement
#define SCALING 2
//minimal line lenght = width*height*MIN_LENGTH_FACTOR
#define MIN_LENGTH_FACTOR 0.00001;
//Max angle between matched lines
#define MAX_LINE_ANGLE M_PI/4.0
//defines number of subsets which are randomly picked to compute a Homography each. Homographies = Number of matches*NUM_OF_PAIR_SUBSETS_FACTOR
#define NUM_LINE_PAIR_SUBSETS_FACTOR 20
//Number of subsets for Homography computations
#define NUM_LINE_PAIR_SUBSETS 500
//Number of correspondencies per homography estimation
#define NUM_CORRESP 4
//Max hemming distance of binary matchig
#define MAX_HEMMING_DIST 20
//Factor for selecting wrong matches in refinement step after first homography estimation
#define OUTLIER_THESHOLD_FACTOR 2.0

#endif // STATICS_H
