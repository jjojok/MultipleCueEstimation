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

class FEstimatorLines : FEstimationMethod {
public:
    FEstimatorLines(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name);
    Mat compute();

private:
    int extractMatches();
    void fillHLinEq(Mat* linEq, lineCorrespStruct lc, int numPair);
    void fillHLinEqBase(Mat* linEq, float x, float y, float A, float B, float C, int row);
    Mat calcLMedS(std::vector<lineSubsetStruct> subsets);
    float calcMedS(Mat Hs);

    std::vector<lineCorrespStruct> lineCorrespondencies;  //Vector of consecutive normalized line startpoints and endpoints
};

#endif // FESTIMATORLINES_H
