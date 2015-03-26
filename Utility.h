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
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

#include "Utility.h"
#include "Statics.h"
#include "DataStructs.h"
#include "FEstimationMethod.h"

#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include <ctime>

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
double meanSquaredSymmeticTransferError(Mat F, std::vector<Point2d> points1, std::vector<Point2d> points2);
double symmeticTransferError(Mat F, Mat x1, Mat x2);
Mat matVector(double x, double y, double z);
Mat matVector(Point2d p);
double randomSampleSymmeticTransferError(Mat F1, Mat F2, Mat image1, Mat image2, int numOfSamples);
double randomSampleSymmeticTransferErrorSub(Mat F1, Mat F2, Mat image1, Mat image2, int numOfSamples);
bool ImgParamsFromFile(std::string file, Mat &K, Mat &R, Mat &t);
double fnorm(double x, double y);
void enforceRankTwoConstraint(Mat &F);
lineCorrespStruct getlineCorrespStruct(double start1x, double start1y, double end1x, double end1y, double start2x, double start2y, double end2x, double end2y, int id);
lineCorrespStruct getlineCorrespStruct(cv::line_descriptor::KeyLine l1, cv::line_descriptor::KeyLine l2, int id);
lineCorrespStruct getlineCorrespStruct(lineCorrespStruct lcCopy);
void visualizeMatches(Mat image_1_color, Mat image_2_color, std::vector<lineCorrespStruct> correspondencies, int lineWidth, bool drawConnections, std::string name);
void visualizeMatches(Mat image_1_color, Mat image_2_color, std::vector<Point2d> p1, std::vector<Point2d> p2, int lineWidth, bool drawConnections, std::string name);
bool isUnity(Mat m);
bool computeUniqeEigenvector(Mat H, Mat &e);
std::vector<double> computeCombinedErrorVect(std::vector<FEstimationMethod> estimations, Mat F);
std::vector<double> computeCombinedErrorVect(std::vector<Mat> x1_vect, std::vector<Mat> x2_vect, Mat F);
double computeCombinedMeanSquaredError(std::vector<FEstimationMethod> estimations, Mat F);
double computeCombinedMeanSquaredError(std::vector<Mat> x1, std::vector<Mat> x2, Mat impF);
void findGoodCombinedMatches(std::vector<FEstimationMethod> estimations, std::vector<Mat> &x1, std::vector<Mat> &x2, Mat F, double maxDist);
void computeEpipoles(Mat F, Mat &e1, Mat &e2);
Mat computeGeneralHomography(Mat F);
//void symmeticTransferLineError(Mat H_invT, Mat H_T, Mat line1Start, Mat line1End, Mat line2Start, Mat line2End, double *err1, double *err2);
double transferLineError(Mat H, Mat line1Start, Mat line1End, Mat line2Start, Mat line2End);
double squaredTransferLineError(Mat H, Mat line1Start, Mat line1End, Mat line2Start, Mat line2End);
double squaredTransferPointError(Mat H, Mat p1, Mat p2);

#endif // UTILITY_H
