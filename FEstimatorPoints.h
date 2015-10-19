#ifndef FESTIMATORPOINTS_H
#define FESTIMATORPOINTS_H

#include "Utility.h"
#include "FeatureMatcher.h"

class FEstimatorPoints : public FEstimationMethod {
public:
    FEstimatorPoints(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name);
    bool compute();
    int extractMatches();

    std::vector<Point2d> getUsedX1();
    std::vector<Point2d> getUsedX2();

private:

    std::vector<Point2d> x1_used, x2_used;
    std::vector<pointCorrespStruct> allPointCorrespondencies;
    std::vector<Point2d> x1, x2;    //corresponding points in image 1 and 2
};

#endif // FESTIMATOR_H
