#ifndef FESTIMATION_METHOD_H
#define FESTIMATION_METHOD_H

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
#include <opencv2/line_descriptor.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

#include "Statics.h"
#include "Utility.h"

using namespace cv;

class FEstimationMethod
{
public:
    virtual bool compute();
    FEstimationMethod();
    ~FEstimationMethod();
    Mat getF();
    virtual int extractMatches();
    std::string name;
    bool isSuccessful();
    std::vector<Mat> getFeaturesImg1();
    std::vector<Mat> getFeaturesImg2();
    std::vector<Mat> getCompleteFeaturesImg1();
    std::vector<Mat> getCompleteFeaturesImg2();
    int getType();

    int featureCount;
    int featureCountComplete;
    int inlierCountOwn;
    int inlierCountCombined;

    int featureCountCorrect;
    int inlierCountOwnCorrect;
    int inlierCountCombinedCorrect;

    double sampsonErrOwn;
    double sampsonErrCombined;
    double sampsonErrCorrect;

    double sampsonErrStdDevCombined;

protected:
    virtual void init();
    virtual Mat normalize(Mat T, Mat p);
    virtual Mat normalize(Mat T, double x, double y, double z = 1);
    virtual Mat denormalize(Mat M, Mat T1, Mat T2);

	Mat F;
    Mat normT1, normT2;
    Mat image_1_color, image_2_color, image_1, image_2;
	double epipolarError;
    bool successful;
    int computationType;
    std::vector<Mat> featuresImg1;
    std::vector<Mat> featuresImg2;
    std::vector<Mat> compfeaturesImg1;
    std::vector<Mat> compfeaturesImg2;

private:

};

#endif // FESTIMATION_METHOD_H
