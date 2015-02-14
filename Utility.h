#ifndef UTILITY_H
#define UTILITY_H

//SYSTEM
#include <string>
#include <cstdio>
#include <stdio.h>
#include <iostream>
#include <fstream>

//OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

void showImage(std::string name, Mat image, int type = WINDOW_NORMAL, int width = 800, int height = 0);
Scalar squaredError(Mat A, Mat B);

#endif // UTILITY_H
