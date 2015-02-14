#ifndef FESTIMATORPLANES_H
#define FESTIMATORPLANES_H

#include "FEstimationMethod.h"

struct segmentStruct {
    int id;
    int area;
    Point startpoint;
    int contours_idx;
};

class FEstimatorPlanes : FEstimationMethod {
public:
    FEstimatorPlanes(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name);
    Mat compute();

private:
    int extractMatches();
    void findSegments(Mat image, Mat image_color, std::string image_name, std::vector<segmentStruct> &segmentList, Mat &segments);
};

#endif // FESTIMATORPLANES_H
