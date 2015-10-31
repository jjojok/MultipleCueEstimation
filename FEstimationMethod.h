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

    void setGroundTruth(Mat Fgt);

    //Evaluation data:
    //Features which are thought to be "good" after extraction, used for ransac
    int featureCountGood;
    //Complete set of extracted features
    int featureCountComplete;
    //Solution inlier of "good" extracted features
    int inlierCountOwnGood;
    //Solution inlier of complete set of extracted features
    int inlierCountOwnComplete;
    //Solution inlier of combined feature sets from all estimations
    int inlierCountCombined;

    //Amount of features which are thought to be "good" after extraction which are described by the ground truth (error < INLIER_THRESHOLD)
    int trueFeatureCountGood;
    //Amount of features extracted features which are described by the ground truth (error < INLIER_THRESHOLD)
    int trueFeatureCountComplete;
    //Solution inlier of "good" extracted features which match the ground truth (error < INLIER_THRESHOLD)
    int trueInlierCountOwnGood;
    //Solution inlier of complete set of extracted features which are described by the ground truth (error < INLIER_THRESHOLD)
    int trueInlierCountOwnComplete;
    //Solution inlier of combined feature sets from all estimations which are described by the ground truth (error < INLIER_THRESHOLD)
    int trueInlierCountCombined;

    //Mean sampson distance from "good" features
    double sampsonErrOwn;
    //Mean sampson distance from complete set of extraced features
    double sampsonErrComplete;
    //Mean sampson distance from combined feature sets from all estimations
    double sampsonErrCombined;
    //Mean sampson distance from combined feature sets which are described by the ground truth (error < INLIER_THRESHOLD)
    double trueSampsonErr;
    //Mean root sampson distance from combined feature sets which are described by the ground truth (error < INLIER_THRESHOLD)
    double trueRootSampsonErr;

    //Standarddeviation of mean sampson distance from combined feature sets from all estimations
    double sampsonErrStdDevCombined;
    //Standarddeviation of mean sampson distance from combined feature sets which are described by the ground truth (error < INLIER_THRESHOLD)
    double trueSampsonErrStdDev;
    //Standarddeviation of mean root sampson distance from combined feature sets which are described by the ground truth (error < INLIER_THRESHOLD)
    double trueRootSampsonErrStdDev;

    //Quality measure for best estimation selection
    double quality;

protected:
    virtual void init();
    virtual Mat normalize(Mat T, Mat p);
    virtual Mat normalize(Mat T, double x, double y, double z = 1);
    virtual Mat denormalize(Mat M, Mat T1, Mat T2);


    double compareWithGroundTruth;
    Mat Fgt;
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
