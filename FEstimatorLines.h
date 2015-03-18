#ifndef FESTIMATORLINES_H
#define FESTIMATORLINES_H

#include "FEstimationMethod.h"
#include <eigen3/Eigen/Eigenvalues>
#include <opencv2/core/eigen.hpp>
#include "Utility.h"

struct lineCorrespStruct {
    //cv::line_descriptor::KeyLine line1, line2;
    float line1Angle, line2Angle, line1Length, line2Length;
    Mat line1StartNormalized, line1EndNormalized, line2StartNormalized, line2EndNormalized;
    Mat line1Start, line1End, line2Start, line2End;
    int id;
};

struct lineCorrespSubsetError {
    int lineCorrespIdx;
    float lineCorrespError;
};

struct lineSubsetStruct {
    std::vector<lineCorrespStruct> lineCorrespondencies;
    //std::vector<int> lineCorrespondenceIdx;
    float qualityMeasure;
    double meanSquaredSymmeticTransferError;
    Mat Hs, Hs_normalized;
    //std::vector<lineCorrespSubsetError> correspErrors;
};

bool compareLineCorrespErrors(lineCorrespSubsetError ls1, lineCorrespSubsetError ls2);

class FEstimatorLines : public FEstimationMethod {
public:
    FEstimatorLines(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name);
    ~FEstimatorLines();
    bool compute();
    int extractMatches();
    bool computeHomography(lineSubsetStruct &subset);
    double squaredSymmeticTransferError(Mat H_invT, Mat H_T, lineCorrespStruct lc);
    double squaredSymmeticTransferError(Mat H, lineCorrespStruct lc);

private:

    void visualizeProjectedLines(lineSubsetStruct subset, int lineWidth, bool drawConnections, std::string name);
    bool hasGeneralPosition(std::vector<int> subsetsIdx, int newIdx, std::vector<lineCorrespStruct> lineCorrespondencies);
    lineSubsetStruct calcRANSAC(std::vector<lineSubsetStruct> &subsets, double threshold, std::vector<lineCorrespStruct> lineCorrespondencies);
    int filterLineExtractions(float minLenght, std::vector<cv::line_descriptor::KeyLine> &keylines);
    //int filterLineMatches(std::vector<DMatch> &matches);
    void fillHLinEq(Mat &linEq, std::vector<lineCorrespStruct> correspondencies);
    void fillHLinEqBase(Mat &linEq, float x, float y, float A, float B, float C, int row);
    lineSubsetStruct calcLMedS(std::vector<lineSubsetStruct> &subsets, std::vector<lineCorrespStruct> lineCorrespondencies);
    float calcMedS(lineSubsetStruct &subset, std::vector<lineCorrespStruct> lineCorrespondencies);
    Mat* normalizeLines(std::vector<lineCorrespStruct> &correspondencies);
    void filterUsedLineMatches(std::vector<lineCorrespStruct> &matches, std::vector<lineCorrespStruct> usedMatches);
    bool findHomography(std::vector<lineCorrespStruct> &lineCorrespondencies, int method, lineSubsetStruct &result);
    lineSubsetStruct estimateHomography(std::vector<lineCorrespStruct> lineCorrespondencies, int method);
    lineCorrespStruct getlineCorrespStruct(float start1x, float start1y, float end1x, float end1y, float start2x, float start2y, float end2x, float end2y, int id);
    lineCorrespStruct getlineCorrespStruct(cv::line_descriptor::KeyLine l1, cv::line_descriptor::KeyLine l2, int id);
    lineCorrespStruct getlineCorrespStruct(lineCorrespStruct lcCopy);
    void visualizeMatches(std::vector<lineCorrespStruct> correspondencies, int lineWidth, bool drawConnections, std::string name);
    bool filterLineMatch(cv::line_descriptor::KeyLine l1, cv::line_descriptor::KeyLine l2);
    bool isUnity(Mat m);
    bool isUniqe(std::vector<int> subsetsIdx, int newIdx);
    std::vector<lineCorrespStruct> matchedLines;  //Vector of consecutive normalized line startpoints and endpoints
};

#endif // FESTIMATORLINES_H
