#ifndef STATICS_H
#define STATICS_H

//Sift features
#define SIFT_FEATURE_COUNT 800
//Show debug messages
#define LOG_DEBUG true
//Show debug images
#define VISUAL_DEBUG true

//Program parameters for selecting computation mathods
#define F_FROM_POINTS 1
#define F_FROM_LINES 2
#define F_FROM_PLANES 4


//Lines:
//Number of segements for image pyramid
#define OCTAVES 1
//Scaling factor per segement
#define SCALING 1
//minimal line lenght = width*height*MIN_LENGTH_FACTOR
#define MIN_LENGTH_FACTOR 0.00001;
//defines number of subsets which are randomly picked to compute a Homography each. Homographies = Number of matches*NUM_OF_PAIR_SUBSETS_FACTOR
#define NUM_OF_PAIR_SUBSETS_FACTOR 4
//Number of subsets for Homography computations
#define NUM_LINE_PAIR_SUBSETS 50
//Number of correspondencies per homography estimation
#define NUM_CORRESP 4

#endif // STATICS_H
