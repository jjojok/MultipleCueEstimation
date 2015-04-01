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

    std::vector<pointCorrespStruct> pointCorrespondencies; //corresponding points in image 1 and 2
    bool estimateHomography(pointSubsetStruct &result, std::vector<pointCorrespStruct> pointCorresp, int method, int sets);
    pointSubsetStruct calcRANSAC(std::vector<pointSubsetStruct> &subsets, double threshold, std::vector<pointCorrespStruct> pointCorresp);
    pointSubsetStruct calcLMedS(std::vector<pointSubsetStruct> &subsets, std::vector<pointCorrespStruct> pointCorresp);
    double calcMedS(pointSubsetStruct &subset, std::vector<pointCorrespStruct> pointCorresp);
    bool findPointHomography(pointSubsetStruct &bestSubset, int method, double confidence, double outliers);
    double squaredPointError(Mat H, Mat H_inv, pointCorrespStruct pointCorresp);
    double meanSquaredPointError(Mat H, std::vector<pointCorrespStruct> pointCorresp);
    int filterUsedPointMatches(std::vector<pointCorrespStruct> &pointCorresp, std::vector<pointCorrespStruct> usedPointCorresp);
    void computeHomography(pointSubsetStruct &subset);
    int filterBadPointMatches(pointSubsetStruct subset, std::vector<pointCorrespStruct> &pointCorresp, double threshold);
    bool isColinear(std::vector<pointCorrespStruct> fixedCorresp, pointCorrespStruct pcNew);
    double levenbergMarquardt(pointSubsetStruct &bestSubset);
    Mat* normalizePoints(std::vector<pointCorrespStruct> &correspondencies);
    bool isUniqe(std::vector<pointCorrespStruct> existingCorresp, pointCorrespStruct newCorresp);
};

#endif // FESTIMATORHPOINTS_H
