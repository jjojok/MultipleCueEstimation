#ifndef FESTIMATORHPLANES_H
#define FESTIMATORHPLANES_H

#include "Utility.h"
#include "DataStructs.h"

class FEstimatorHPlanes : public FEstimationMethod {
public:
    FEstimatorHPlanes(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name);
    bool compute();
    int extractMatches();

private:
    void findSegments(Mat image, Mat image_color, std::string image_name, std::vector<segmentStruct> &segmentList, Mat &segments);
};

#endif // FESTIMATORHPLANES_H
