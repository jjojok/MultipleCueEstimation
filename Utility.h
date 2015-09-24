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
void visualizeLineMatches(Mat image_1_color, Mat image_2_color, std::vector<lineCorrespStruct> correspondencies, int lineWidth, bool drawConnections, std::string name);
void visualizePointMatches(Mat image_1_color, Mat image_2_color, std::vector<Point2d> p1, std::vector<Point2d> p2, int lineWidth, bool drawConnections, std::string name);
bool isUnity(Mat m);
bool computeUniqeEigenvector(Mat H, Mat &e);
//std::vector<double> computeCombinedErrorVect(std::vector<FEstimationMethod> estimations, Mat F);
std::vector<double> computeCombinedErrorVect(std::vector<Mat> x1_vect, std::vector<Mat> x2_vect, Mat F);
std::vector<double> computeCombinedSquaredErrorVect(std::vector<Mat> x1, std::vector<Mat> x2, Mat F);
//double errorFunctionCombinedMeanSquared(std::vector<FEstimationMethod> estimations, Mat F);
void errorFunctionCombinedMeanSquared(std::vector<Mat> x1, std::vector<Mat> x2, Mat impF, double &error, int &inliers, double inlierThr, double &standardDeviation);
void errorFunctionCombinedMean(std::vector<Mat> x1, std::vector<Mat> x2, Mat impF, double &error, int &inliers, double inlierThr, double &standardDeviation);
void findGoodCombinedMatches(std::vector<Mat> x1Combined, std::vector<Mat> x2Combined, std::vector<Mat> &x1, std::vector<Mat> &x2, Mat F, double maxDist);
void findGoodCombinedMatches(std::vector<Point2d> x1Combined, std::vector<Point2d> x2Combined, std::vector<Point2d> &x1, std::vector<Point2d> &x2, Mat F, double maxDist, double minDist);
void computeEpipoles(Mat F, Mat &e1, Mat &e2);
Mat computeGeneralHomography(Mat F);
pointCorrespStruct getPointCorrespStruct(pointCorrespStruct pcCopy);
double computeRelativeOutliers(double generalOutliers, double uesdCorresp, double correspCount);
int computeNumberOfEstimations(double confidence, double outliers, int corrspNumber);
bool isUniqe(std::vector<int> subsetsIdx, int newIdx);
void visualizePointMatches(Mat image_1_color, Mat image_2_color, std::vector<Mat> x1, std::vector<Mat> x2, int lineWidth, bool drawConnections, std::string name);
void visualizePointMatches(Mat image_1_color, Mat image_2_color, std::vector<pointCorrespStruct> pointCorresp, int lineWidth, bool drawConnections, std::string name);
void homogMat(Mat &m);
Mat* normalize(std::vector<Mat> x1, std::vector<Mat> x2, std::vector<Mat> &x1norm, std::vector<Mat> &x2norm);
double normalizeThr(Mat T1, Mat T2, double thrdth);
bool isEqualPointCorresp(Mat x11, Mat x12, Mat x21, Mat x22);
void matToPoint(std::vector<Mat> xin, std::vector<Point2d> &xout);
bool compareCorrespErrors(correspSubsetError ls1, correspSubsetError ls2);
bool compareFundMatSets(fundamentalMatrix* f1, fundamentalMatrix* f2);
bool compareFundMatSetsSelectedInliers(fundamentalMatrix* f1, fundamentalMatrix* f2);
bool compareFundMatSetsError(fundamentalMatrix* f1, fundamentalMatrix* f2);
bool compareFundMatSetsInlinerError(fundamentalMatrix* f1, fundamentalMatrix* f2);
double meanSampsonFDistanceGoodMatches(Mat Fgt, Mat F, std::vector<Mat> x1, std::vector<Mat> x2);
int goodMatchesCount(Mat Fgt, std::vector<Mat> x1, std::vector<Mat> x2, double thr);

//Error functions:

double sampsonDistanceFundamentalMatSymmetric(Mat F, Mat x1, Mat x2);
double sampsonDistanceFundamentalMat(Mat F, Mat x1, Mat x2);
double sampsonDistanceFundamentalMat(Mat F, std::vector<Point2d> points1, std::vector<Point2d> points2);
double sampsonDistanceFundamentalMat(Mat F, std::vector<Mat> points1, std::vector<Mat> points2);
//error point homogrpahy
double sampsonDistanceHomographySymmetric(Mat H, Mat H_inv, Mat x1, Mat x2);
double sampsonDistanceHomography(Mat H, Mat x1, Mat x2);
//error line homography
double sampsonDistanceHomographySymmetric(Mat H, Mat H_inv, Mat line1Start, Mat line1End, Mat line2Start, Mat line2End);
double sampsonDistanceHomography(Mat H, Mat line1Start, Mat line1End, Mat line2Start, Mat line2End);




//double calc2DHomogSampsonErr(Mat x1, Mat x2, Mat H);
//double errorFunctionHLinesSqared(Mat H, Mat line1Start, Mat line1End, Mat line2Start, Mat line2End);
//double errorFunctionFPointsSquared(Mat F, Mat x1, Mat x2);
//double errorFunctionHPointsSqared(Mat H, Mat x1, Mat x2);
//double errorFunctionHLines(Mat H, Mat line1Start, Mat line1End, Mat line2Start, Mat line2End);
//double errorFunctionHLines(Mat H, Mat H_inv, Mat line1Start, Mat line1End, Mat line2Start, Mat line2End);
//double errorFunctionFPoints(Mat F, Mat x1, Mat x2);
//double errorFunctionHPoints(Mat H, Mat x1, Mat x2);
//double squaredTransferLineError(Mat H, Mat line1Start, Mat line1End, Mat line2Start, Mat line2End);
//double squaredSymmetricTransferPointError(Mat H, Mat H_inv, Mat x1, Mat x2);
//double sampsonHDistance(Mat H, Mat x1, Mat x2);
//double computeUnsquaredSampsonHDistance(Mat H, Mat H_inv, Mat x1, Mat x2);
//double transferPointError(Mat H, Mat p1, Mat p2);
//double symmetricTransferPointError(Mat H, Mat H_inv, Mat x1, Mat x2);
//double squaredTransferPointError(Mat H, Mat x1, Mat x2);
//double transferLineError(Mat H, Mat line1Start, Mat line1End, Mat line2Start, Mat line2End);
//double errorFunctionFPointsSquared(Mat F, Mat x1, Mat x2);
//double computeUnsquaredSampsonFDistance(Mat F, Mat x1, Mat x2);
//double sampsonFDistance(Mat F, Mat x1, Mat x2);
//double sampsonFDistance(Mat F, std::vector<Mat> points1, std::vector<Mat> points2);
//double sampsonFDistance(Mat F, std::vector<Point2d> points1, std::vector<Point2d> points2);
//double meanSquaredSymmeticTransferError(Mat F, std::vector<Point2d> points1, std::vector<Point2d> points2);
//double symmeticTransferError(Mat F, Mat x1, Mat x2);

void meanSampsonFDistanceGoodMatches(Mat Fgt, Mat F, std::vector<Mat> x1, std::vector<Mat> x2, double &error, int &used);

#endif // UTILITY_H
