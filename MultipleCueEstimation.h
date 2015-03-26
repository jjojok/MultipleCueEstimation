#ifndef MCE_H
#define MCE_H

//ESTIMATORS
#include "Utility.h"
#include "FEstimatorPoints.h"
#include "FEstimatorHLines.h"
#include "FEstimatorHPlanes.h"
#include "FEstimatorHPoints.h"

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
    MultipleCueEstimation(Mat *img1, Mat *img2, int comp);
    MultipleCueEstimation(Mat *img1, Mat *img2, int comp, Mat *F_groudtruth);

    Mat compute();
    double getMeanSquaredRSSTError();
    double getMeanSquaredCSTError();
    std::vector<FEstimationMethod> getEstimations();

private:

    int checkData();
    FEstimationMethod* calcFfromPoints();
    FEstimationMethod* calcFfromHPoints();
    FEstimationMethod* calcFfromHLines();
    FEstimationMethod* calcFfromHPlanes();
    FEstimationMethod* calcFfromConics();
    FEstimationMethod* calcFfromCurves();
    Mat refineF(std::vector<FEstimationMethod> estimations);

    int arguments;
    unsigned int computations;
    bool compareWithGroundTruth;
    double meanSquaredRSSTError;    //Mean squared random sample symmetic transfer error
    double meanSquaredCSTError;     //Mean squared symmetic transfer error of combined matches
    Mat F;                          //Final Fundamental Matrix

    Mat image_1, image_2;
    Mat image_1_color, image_2_color;
    Mat Fgt;    //Ground truth
    std::vector<Point2d> x1, x2;   //corresponding points in image 1 and 2 (for debug only)
    std::vector<FEstimationMethod> estimations;

};



#endif // MCE_MAIN_H
