#ifndef FESTIMATORHLINES_H
#define FESTIMATORHLINES_H

#include "Utility.h"
#include "DataStructs.h"

class FEstimatorHLines : public FEstimationMethod {
public:
    FEstimatorHLines(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name);
    ~FEstimatorHLines();
    bool compute();
    int extractMatches();

private:

    double levenbergMarquardt(lineSubsetStruct &bestSubset);
    bool computeHomography(lineSubsetStruct &subset);
    void visualizeProjectedLines(lineSubsetStruct subset, int lineWidth, bool drawConnections, std::string name);
    bool isParallel(std::vector<lineCorrespStruct> fixedCorresp, lineCorrespStruct lcNew);
    lineSubsetStruct calcRANSAC(std::vector<lineSubsetStruct> &subsets, double threshold, std::vector<lineCorrespStruct> lineCorrespondencies);
    void fillHLinEq(Mat &linEq, std::vector<lineCorrespStruct> correspondencies);
    void fillHLinEqBase(Mat &linEq, double x, double y, double A, double B, double C, int row);
    Mat* normalizeLines(std::vector<lineCorrespStruct> &correspondencies, std::vector<lineCorrespStruct> &goodCorrespondencies);
    Mat* normalizeLines(std::vector<lineCorrespStruct> &correspondencies);
    int filterUsedLineMatches(std::vector<lineCorrespStruct> &matches, std::vector<lineCorrespStruct> usedMatches);
    bool findLineHomography(lineSubsetStruct &bestSubset, std::vector<lineCorrespStruct> goodMatches, std::vector<lineCorrespStruct> allMatches, std::vector<lineCorrespStruct> &ransacInlier, int method, double confidence, double outliers, double threshold);
    bool estimateHomography(lineSubsetStruct &result, std::vector<lineCorrespStruct> lineCorrespondencies, int method, int sets, double ransacThr);
    void computePointCorrespondencies(Mat H, std::vector<lineCorrespStruct> lineMatches, std::vector<Mat> &pointMatchesX1, std::vector<Mat> &pointMatchesX2);
    void addAllPointCorrespondencies(Mat H, std::vector<lineCorrespStruct> goodLineMatches);
    int filterBadLineMatches(lineSubsetStruct subset, std::vector<lineCorrespStruct> &lineCorresp, double threshold);
    bool isUniqe(std::vector<lineCorrespStruct> existingCorresp, lineCorrespStruct newCorresp);

    std::vector<lineCorrespStruct> goodMatchedLines;  //Vector of (good) corresponing line segments (start & endpoints)
    std::vector<lineCorrespStruct> allMatchedLines;  //Vector of (all) corresponing line segments (start & endpoints)


    std::vector<lineCorrespStruct> goodMatchedLinesConst;  //Vector of (good) corresponing line segments (start & endpoints), does not change
    std::vector<lineCorrespStruct> allMatchedLinesConst;  //Vector of (all) corresponing line segments (start & endpoints), does not change
};

#endif // FESTIMATORHLINES_H
