#ifndef FESTIMATORHPOINTS_H
#define FESTIMATORHPOINTS_H

#include "Utility.h"
#include "FeatureMatchers.h"
#include "LevenbergMarquardtPoints.h"

class FEstimatorHPoints : public FEstimationMethod {
public:
    FEstimatorHPoints(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name);
    bool compute();
    int extractMatches();

private:

    bool estimateHomography(pointSubsetStruct &result, std::vector<pointCorrespStruct> pointCorresp, int method, int sets, double errorThr);
    pointSubsetStruct calcRANSAC(std::vector<pointSubsetStruct> &subsets, double threshold, std::vector<pointCorrespStruct> pointCorresp);
//    pointSubsetStruct calcLMedS(std::vector<pointSubsetStruct> &subsets, std::vector<pointCorrespStruct> pointCorresp);
//    double calcMedS(pointSubsetStruct &subset, std::vector<pointCorrespStruct> pointCorresp);
    bool findPointHomography(pointSubsetStruct &bestSubset, std::vector<pointCorrespStruct> goodMatches, std::vector<pointCorrespStruct> allMatches, std::vector<pointCorrespStruct> &ransacInlier, int method, double confidence, double outliers, double threshold);
    double errorFunctionHPointsSquared_(Mat H, pointCorrespStruct pointCorresp);
    int filterUsedPointMatches(std::vector<pointCorrespStruct> &pointCorresp, std::vector<pointCorrespStruct> usedPointCorresp);
    void computeHomography(pointSubsetStruct &subset);
    int filterBadPointMatches(pointSubsetStruct subset, std::vector<pointCorrespStruct> &pointCorresp, double threshold);
    bool isColinear(std::vector<pointCorrespStruct> fixedCorresp, pointCorrespStruct pcNew);
    double levenbergMarquardt(pointSubsetStruct &bestSubset);
    Mat* normalizePoints(std::vector<pointCorrespStruct> &correspondencies);
    bool isUniqe(std::vector<pointCorrespStruct> existingCorresp, pointCorrespStruct newCorresp);
//    double errorFunctionHPoints_(Mat H, pointCorrespStruct pointCorresp);
    double sampsonDistanceHomography_(Mat H, std::vector<pointCorrespStruct> pointCorresp);
    double sampsonDistanceHomography_(Mat H, Mat H_inv, pointCorrespStruct pointCorresp);

    std::vector<pointCorrespStruct> goodMatchedPoints;  //Vector of (good) corresponing points
    std::vector<pointCorrespStruct> allMatchedPoints;  //Vector of (all) corresponing points

    std::vector<pointCorrespStruct> goodMatchedPointsConst;  //Vector of (good) corresponing points, does not change
    std::vector<pointCorrespStruct> allMatchedPointsConst;  //Vector of (all) corresponing points, does not change

    std::vector<pointCorrespStruct> inlierUsed;

};

#endif // FESTIMATORHPOINTS_H
