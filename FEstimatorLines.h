#ifndef FESTIMATORLINES_H
#define FESTIMATORLINES_H

#include "FEstimationMethod.h"

struct lineCorrespStruct {
    cv::line_descriptor::KeyLine line1, line2;
    Mat line1StartNormalized, line1EndNormalized, line2StartNormalized, line2EndNormalized;
    Mat line1Start, line1End, line2Start, line2End;
};

struct lineSubsetStruct {
    std::vector<lineCorrespStruct> lineCorrespondencies;
    std::vector<int> lineCorrespondenceIdx;
    Mat Hs;
};

class FEstimatorLines : public FEstimationMethod {
public:
    FEstimatorLines(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name);
    ~FEstimatorLines();
    Mat compute();
    int extractMatches();

private:
    int filterLineExtractions(std::vector<cv::line_descriptor::KeyLine> &keylines);
    void filterLineMatches(cv::Mat descr1, cv::Mat descr2, std::vector<DMatch> matches);
    void fillHLinEq(Mat* linEq, std::vector<lineCorrespStruct> correspondencies);
    void fillHLinEqBase(Mat* linEq, float x, float y, float A, float B, float C, int row);
    Mat calcLMedS(std::vector<lineSubsetStruct> subsets);
    float calcMedS(Mat Hs);
    Mat* normalizeLines(std::vector<lineCorrespStruct> &correspondencies);
    double squaredDistance(Mat H, lineCorrespStruct lc);
    lineCorrespStruct getlineCorrespStruct(float start1x, float start1y, float start2x, float start2y, float end1x, float end1y, float end2x, float end2y);

    std::vector<lineCorrespStruct> lineCorrespondencies;  //Vector of consecutive normalized line startpoints and endpoints
};

#endif // FESTIMATORLINES_H
