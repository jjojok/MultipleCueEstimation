#ifndef MCE_H
#define MCE_H

//ESTIMATORS
#include "FEstimationMethod.h"
#include "FEstimatorPoints.h"
#include "FEstimatorLines.h"
#include "FEstimatorPlanes.h"

//SYSTEM
#include <string>
#include <cstdio>
#include <stdio.h>
#include <iostream>
#include <fstream>

//OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "opencv2/core/utility.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/line_descriptor.hpp>

#define SIFT_FEATURE_COUNT 800
#define LOG_DEBUG true
#define VISUAL_DEBUG true
#define F_FROM_POINTS 1
#define F_FROM_LINES 2
#define F_FROM_PLANES 4

//Line Matching:
//Number of segements for image pyramid
#define OCTAVES 2
//Scaling factor per segement
#define SCALING 1
//minimal line lenght = width*height*MIN_LENGTH_FACTOR
#define MIN_LENGTH_FACTOR 0.00001;
//defines number of subsets which are randomly picked to compute a Homography each. Homographies = Number of matches*NUM_OF_PAIR_SUBSETS_FACTOR
#define NUM_OF_PAIR_SUBSETS_FACTOR 4

using namespace cv;

class MultipleCueEstimation
{
public:
    MultipleCueEstimation();

    void run();
    int loadData();
    FEstimationMethod* calcFfromPoints();
    FEstimationMethod* calcFfromLines();
    FEstimationMethod* calcFfromPlanes();
    FEstimationMethod* calcFfromConics();
    FEstimationMethod* calcFfromCurves();
    Mat refineF();

    Mat getGroundTruth();

    int arguments;
    std::string path_img1, path_P1;
    std::string path_img2, path_P2;
    unsigned int computations;
    bool compareWithGroundTruth;

private:

    Mat image_1, image_2;
    Mat image_1_color, image_2_color;
    std::vector<Point2f> x1, x2;   //corresponding points in image 1 and 2
};

#endif // MCE_MAIN_H
