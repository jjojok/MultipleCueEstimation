#ifndef FESTIMATORHLINES_H
#define FESTIMATORHLINES_H

#include "FEstimationMethod.h"
#include <eigen3/Eigen/Eigenvalues>
#include <opencv2/core/eigen.hpp>
#include "Utility.h"

struct lineCorrespStruct {
    //cv::Sline_descriptor::KeyLine line1, line2;
    double line1Angle, line2Angle, line1Length, line2Length;
    Mat line1StartNormalized, line1EndNormalized, line2StartNormalized, line2EndNormalized;
    Mat line1Start, line1End, line2Start, line2End;
    int id;
};

struct lineCorrespSubsetError {
    int lineCorrespIdx;
    double lineCorrespError;
};

struct lineSubsetStruct {
    std::vector<lineCorrespStruct> lineCorrespondencies;
    //std::vector<int> lineCorrespondenceIdx;
    double qualityMeasure;
    double meanSquaredSymmeticTransferError;
    Mat Hs, Hs_normalized;
    //std::vector<lineCorrespSubsetError> correspErrors;
};

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
    bool hasGeneralPosition(std::vector<int> subsetsIdx, int newIdx, std::vector<lineCorrespStruct> lineCorrespondencies);
    lineSubsetStruct calcRANSAC(std::vector<lineSubsetStruct> &subsets, double threshold, std::vector<lineCorrespStruct> lineCorrespondencies);
    int filterLineExtractions(double minLenght, std::vector<cv::line_descriptor::KeyLine> &keylines);
    //int filterLineMatches(std::vector<DMatch> &matches);
    void fillHLinEq(Mat &linEq, std::vector<lineCorrespStruct> correspondencies);
    void fillHLinEqBase(Mat &linEq, double x, double y, double A, double B, double C, int row);
    lineSubsetStruct calcLMedS(std::vector<lineSubsetStruct> &subsets, std::vector<lineCorrespStruct> lineCorrespondencies);
    double calcMedS(lineSubsetStruct &subset, std::vector<lineCorrespStruct> lineCorrespondencies);
    Mat* normalizeLines(std::vector<lineCorrespStruct> &correspondencies);
    double filterUsedLineMatches(std::vector<lineCorrespStruct> &matches, std::vector<lineCorrespStruct> usedMatches);
    bool findHomography(std::vector<lineCorrespStruct> &lineCorrespondencies, int method, double confidence, double outliers, lineSubsetStruct &result);
    lineSubsetStruct estimateHomography(std::vector<lineCorrespStruct> lineCorrespondencies, int method, int sets);
    lineCorrespStruct getlineCorrespStruct(double start1x, double start1y, double end1x, double end1y, double start2x, double start2y, double end2x, double end2y, int id);
    lineCorrespStruct getlineCorrespStruct(cv::line_descriptor::KeyLine l1, cv::line_descriptor::KeyLine l2, int id);
    lineCorrespStruct getlineCorrespStruct(lineCorrespStruct lcCopy);
    void visualizeMatches(std::vector<lineCorrespStruct> correspondencies, int lineWidth, bool drawConnections, std::string name);
    bool filterLineMatch(cv::line_descriptor::KeyLine l1, cv::line_descriptor::KeyLine l2);
    bool isUnity(Mat m);
    bool isUniqe(std::vector<int> subsetsIdx, int newIdx);

    std::vector<lineCorrespStruct> matchedLines;  //Vector of corresponing line segments (start & endpoints)
};

#endif // FESTIMATORHLINES_H
