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
    virtual Mat compute();
    FEstimationMethod();
    ~FEstimationMethod();
    Mat getF();
    virtual int extractMatches();
    std::string name;

protected:
    virtual void init();
    virtual Mat normalize(Mat T, Mat p);
    virtual Mat normalize(Mat T, float x, float y, float z = 1);
    virtual Mat denormalize(Mat M, Mat T1, Mat T2);

	Mat F;
    Mat normT1, normT2;
    Mat image_1_color, image_2_color, image_1, image_2;
	double epipolarError;

private:

};

#endif // FESTIMATION_METHOD_H
