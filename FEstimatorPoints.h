#ifndef FESTIMATORPOINTS_H
#define FESTIMATORPOINTS_H

#include "FEstimationMethod.h"

class FEstimatorPoints : public FEstimationMethod {
public:
    FEstimatorPoints(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name);
    Mat compute();
    std::vector<Point2f> getX1();
    std::vector<Point2f> getX2();
    int extractMatches();

private:

    std::vector<Point2f> x1, x2;    //corresponding points in image 1 and 2
};

#endif // FESTIMATOR_H
