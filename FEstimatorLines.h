#ifndef FESTIMATORLINES_H
#define FESTIMATORLINES_H

#include "FEstimationMethod.h"

struct lineCorrespStruct {
    cv::line_descriptor::KeyLine line1, line2;
    Point2f line1StartNormalized, line1EndNormalized, line2StartNormalized, line2EndNormalized;
};

struct lineSubsetStruct {
    std::vector<lineCorrespStruct> lineCorrespondencies;
    Mat Hs;
};

class FEstimatorLines : public FEstimationMethod {
public:
    FEstimatorLines(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name);
    ~FEstimatorLines();
    Mat compute();
    int extractMatches();

private:
    int filterLineExtractions(std::vector<cv::line_descriptor::KeyLine>* keylines);
    void filterLineMatches(cv::Mat descr1, cv::Mat descr2, std::vector<DMatch> matches);
    void fillHLinEq(Mat* linEq, lineCorrespStruct lc, int numPair);
    void fillHLinEqBase(Mat* linEq, float x, float y, float A, float B, float C, int row);
    Mat calcLMedS(std::vector<lineSubsetStruct> subsets);
    float calcMedS(Mat Hs);

    std::vector<lineCorrespStruct> lineCorrespondencies;  //Vector of consecutive normalized line startpoints and endpoints
};

#endif // FESTIMATORLINES_H
