#ifndef MCE_H
#define MCE_H

//ESTIMATORS
#include "Utility.h"
#include "FEstimatorPoints.h"
#include "FEstimatorHLines.h"
#include "FEstimatorHPoints.h"
#include "LevenbergMarquardtGeneral.h"
#include "clm/SevenpointLevenbergMarquardt.h"
//#include "LevenbergMarquardtStandardDeviation.h"

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

    int combinedFeatures;
    int combinedFeaturesCorrect;
    int refinedFInlierCombined;

    double refinedFSampsonDistCombined;
    double refinedFSampsonDistCorrect;

    double refinedFSampsonErrStdDevCombined;

    double groundTruthSampsonDistCombined;

    Mat compute();
    std::vector<FEstimationMethod> getEstimations();

private:

    int checkData();
    FEstimationMethod* calcFfromPoints();
    FEstimationMethod* calcFfromHPoints();
    FEstimationMethod* calcFfromHLines();
    FEstimationMethod* calcFfromHPlanes();
    FEstimationMethod* calcFfromConics();
    FEstimationMethod* calcFfromCurves();

    Mat refineF(std::vector<FEstimationMethod> &estimations);

    void combinePointCorrespondecies();
    void combineAllPointCorrespondecies();
    //double qualitiy(double sampsonErrCombined, int inlierCountCombined, int combinedMatches, double sampsonErrStdDevCombined);
    double qualitiy(double sampsonErrCombined, double smallestSampsonErr, int inlierCountCombined, int largestInlier, double sampsonErrStdDevCombined, double smallestSampsonErrStdDev);
    //    void levenbergMarquardt(Mat &Flm, std::vector<Mat> x1, std::vector<Mat> x2, std::vector<Mat> &goodCombindX1, std::vector<Mat> &goodCombindX2, double &errorThr, int minFeatureChange, double minErrorChange, double lmErrorThr, double errorDecay, int &inliers, int minStableSolutions, int maxIterations, double maxError, double &stdDeviation, double &error);

    bool SPLM(Mat &F, std::vector<Mat> x1, std::vector<Mat> x2);

    int arguments;
    unsigned int computations;
    bool compareWithGroundTruth;
    Mat F;                          //Final Fundamental Matrix

    Mat image_1, image_2;
    Mat image_1_color, image_2_color;
    Mat Fgt;    //Ground truth
    std::vector<Point2d> x1, x2;   //corresponding points in image 1 and 2 (for debug only)
    std::vector<Mat> x1Combined, x2Combined;   //combined corresponding points in image 1 and 2 from all computation
    std::vector<Mat> x1Complete, x2Complete;   //complete set of extracted correspondences
    std::vector<FEstimationMethod> estimations;

};



#endif // MCE_MAIN_H
