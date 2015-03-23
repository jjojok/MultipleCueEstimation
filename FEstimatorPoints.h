#ifndef FESTIMATORPOINTS_H
#define FESTIMATORPOINTS_H

#include "Utility.h"
#include "FeatureMatchers.h"

class FEstimatorPoints : public FEstimationMethod {
public:
    FEstimatorPoints(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name);
    bool compute();
    int extractMatches();

private:

    std::vector<Point2d> x1_used, x2_used;
    std::vector<Point2d> x1, x2;    //corresponding points in image 1 and 2
};

#endif // FESTIMATOR_H
