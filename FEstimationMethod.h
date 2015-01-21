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
#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/line_descriptor.hpp>

using namespace cv;

class FEstimationMethod
{
public:
    FEstimationMethod(Mat img1, Mat img2);
    virtual Mat compute();
	virtual int extractMatches();
	double getEpipolarError(std::vector<Point2f> points1, std::vector<Point2f> points2);

    std::string name;

protected:
	Mat F;
	Mat image1, image2;
	double epipolarError;
};

#endif // COMPUTATION_METHOD_H
