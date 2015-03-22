#ifndef FESTIMATORHPOINTS_H
#define FESTIMATORHPOINTS_H

#include "FEstimationMethod.h"

class FEstimatorHPoints : public FEstimationMethod {
public:
    FEstimatorHPoints(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name);
    bool compute();
    int extractMatches();

private:

    void visualizeMatches(std::vector<Point2d> p1, std::vector<Point2d> p2, int lineWidth, bool drawConnections, std::string name);

    std::vector<Point2d> x1_used, x2_used;
    std::vector<Point2d> x1, x2;    //corresponding points in image 1 and 2
};

#endif // FESTIMATORHPOINTS_H
