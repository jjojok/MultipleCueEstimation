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

void visualizeHomography(Mat H21, Mat img1, Mat img2, std::string name);
double smallestRelAngle(double ang1, double ang2);
void showImage(std::string name, Mat image, int type = WINDOW_NORMAL, int width = 800, int height = 0);
double squaredError(Mat A, Mat B);
Mat MatFromFile(std::string file, int cols);
Mat crossProductMatrix(Mat input);
void rectify(std::vector<Point2d> p1, std::vector<Point2d> p2, Mat F, Mat image1, Mat image2, std::string windowName);
void drawEpipolarLines(std::vector<Point2d> p1, std::vector<Point2d> p2, Mat F, Mat image1, Mat image2, std::string name);
std::string getType(Mat m);
int calcMatRank(Mat M);
int calcNumberOfSolutions(Mat linEq);
double epipolarSADError(Mat F, std::vector<Point2d> points1, std::vector<Point2d> points2);
Mat matVector(double x, double y, double z);
Mat matVector(Point2d p);
double randomSampleSymmeticTransferError(Mat F1, Mat F2, Mat image, int numOfSamples);
double randomSampleSymmeticTransferErrorSub(Mat F1, Mat F2, Mat image, int numOfSamples);
bool ImgParamsFromFile(std::string file, Mat &K, Mat &R, Mat &t);
double fnorm(double x, double y);
void enforceRankTwoConstraint(Mat &F);
void decomPoseFundamentalMat(Mat F, Mat &P1, Mat &P2);
void decomPoseFundamentalMat(Mat F, Mat &K1, Mat &R12, Mat T12);

#endif // UTILITY_H
