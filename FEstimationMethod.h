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
    //FEstimationMethod(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name);
	double getEpipolarError(std::vector<Point2f> points1, std::vector<Point2f> points2);
    virtual Mat compute() = 0;

    std::string name;

protected:
    virtual void init();
    virtual int extractMatches() = 0;
    virtual Point2f normalize();
    virtual Mat denormalize();

	Mat F;
    Mat normT1, normT2;
    Mat image_1_color, image_2_color, image_1, image_2;
	double epipolarError;
};

#endif // COMPUTATION_METHOD_H
