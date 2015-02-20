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
