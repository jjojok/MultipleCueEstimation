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
double squaredError(Mat A, Mat B);
Mat MatFromFile(std::string file, int cols);
Mat crossProductMatrix(Mat input);
void rectify(std::vector<Point2f> p1, std::vector<Point2f> p2, Mat F, Mat image1, Mat image2, std::string windowName);
void drawEpipolarLines(std::vector<Point2f> p1, std::vector<Point2f> p2, Mat F, Mat image1, Mat image2, std::string name);
std::string getType(Mat m);
int calcMatRank(Mat M);
int calcNumberOfSolutions(Mat linEq);
double epipolarSADError(Mat F, std::vector<Point2f> points1, std::vector<Point2f> points2);
Mat matVector(float x, float y, float z);
Mat matVector(Point2f p);
double epipolarLineDistanceError(Mat F1, Mat F2, Mat image, int numOfSamples);
double epipolarLineDistanceErrorSub(Mat F1, Mat F2, Mat image, int numOfSamples);

#endif // UTILITY_H
