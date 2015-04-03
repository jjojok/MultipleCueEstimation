#ifndef FEATUREMATCHERS_H
#define FEATUREMATCHERS_H

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
int extractPointMatches(Mat image_1, Mat image_2, std::vector<pointCorrespStruct> &allMatchedPoints);
int extractLineMatches(Mat image_1, Mat image_2, std::vector<lineCorrespStruct> &allMatchedLines);
int filterLineExtractions(double minLenght, std::vector<cv::line_descriptor::KeyLine> &keylines);
bool filterLineMatch(cv::line_descriptor::KeyLine l1, cv::line_descriptor::KeyLine l2);

#endif
