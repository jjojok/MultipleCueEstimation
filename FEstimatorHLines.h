#ifndef FESTIMATORHLINES_H
#define FESTIMATORHLINES_H

#include "Utility.h"
#include "DataStructs.h"

bool compareLineCorrespErrors(lineCorrespSubsetError ls1, lineCorrespSubsetError ls2);

class FEstimatorHLines : public FEstimationMethod {
public:
    FEstimatorHLines(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name);
    ~FEstimatorHLines();
    bool compute();
    int extractMatches();

private:

    double levenbergMarquardt(lineSubsetStruct &bestSubset);
    bool computeHomography(lineSubsetStruct &subset);
    double squaredSymmeticTransferError(Mat H_invT, Mat H_T, lineCorrespStruct lc);
    double squaredSymmeticTransferError(Mat H, lineCorrespStruct lc);
    void visualizeProjectedLines(lineSubsetStruct subset, int lineWidth, bool drawConnections, std::string name);
    bool isParallel(std::vector<lineCorrespStruct> fixedCorresp, lineCorrespStruct lcNew);
    lineSubsetStruct calcRANSAC(std::vector<lineSubsetStruct> &subsets, double threshold, std::vector<lineCorrespStruct> lineCorrespondencies);
    //int filterLineMatches(std::vector<DMatch> &matches);
    void fillHLinEq(Mat &linEq, std::vector<lineCorrespStruct> correspondencies);
    void fillHLinEqBase(Mat &linEq, double x, double y, double A, double B, double C, int row);
    lineSubsetStruct calcLMedS(std::vector<lineSubsetStruct> &subsets, std::vector<lineCorrespStruct> lineCorrespondencies);
    double calcMedS(lineSubsetStruct &subset, std::vector<lineCorrespStruct> lineCorrespondencies);
    Mat* normalizeLines(std::vector<lineCorrespStruct> &correspondencies);
    int filterUsedLineMatches(std::vector<lineCorrespStruct> &matches, std::vector<lineCorrespStruct> usedMatches);
    bool findLineHomography(std::vector<lineCorrespStruct> &lineCorrespondencies, int method, double confidence, double outliers, lineSubsetStruct &result);
    bool estimateHomography(lineSubsetStruct &result, std::vector<lineCorrespStruct> lineCorrespondencies, int method, int sets);
    void addPointCorrespondencies(Mat H, std::vector<lineCorrespStruct> goodLineMatches);
    double squaredSymmeticTransferLineError(Mat H, lineCorrespStruct lc);
    double squaredSymmeticTransferLineError(Mat H_invT, Mat H_T, lineCorrespStruct lc);
    int filterBadLineMatches(lineSubsetStruct subset, std::vector<lineCorrespStruct> &lineCorresp, double threshold);
    std::vector<lineCorrespStruct> matchedLines;  //Vector of corresponing line segments (start & endpoints)
};

#endif // FESTIMATORHLINES_H
