#ifndef FESTIMATORHPOINTS_H
#define FESTIMATORHPOINTS_H

#include "Utility.h"
#include "FeatureMatchers.h"

class FEstimatorHPoints : public FEstimationMethod {
public:
    FEstimatorHPoints(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name);
    bool compute();
    int extractMatches();

private:

    std::vector<pointCorrespStruct> pointCorrespondencies; //corresponding points in image 1 and 2
    pointSubsetStruct estimateHomography(std::vector<pointCorrespStruct> pointCorresp, int method, int sets);
    pointSubsetStruct calcRANSAC(std::vector<pointSubsetStruct> &subsets, double threshold, std::vector<pointCorrespStruct> pointCorresp);
    pointSubsetStruct calcLMedS(std::vector<pointSubsetStruct> &subsets, std::vector<pointCorrespStruct> pointCorresp);
    double calcMedS(pointSubsetStruct &subset, std::vector<pointCorrespStruct> pointCorresp);
    bool findPointHomography(std::vector<pointCorrespStruct> &pointCorresp, int method, double confidence, double outliers, pointSubsetStruct &result);
    double sampsonDistance(Mat H, pointCorrespStruct pointCorresp);
    double sampsonDistance(Mat H, Mat H_inv, pointCorrespStruct pointCorresp);
    double sampsonDistance(Mat H, std::vector<pointCorrespStruct> pointCorresp);
    void filterUsedPointMatches(std::vector<pointCorrespStruct> &pointCorresp, std::vector<pointCorrespStruct> usedPointCorresp);
    void findHomography(pointSubsetStruct &subset);
};

#endif // FESTIMATORHPOINTS_H
