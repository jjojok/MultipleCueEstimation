#ifndef FESTIMATORHPOINTS_H
#define FESTIMATORHPOINTS_H

#include "Utility.h"
#include "FeatureMatchers.h"

class FEstimatorHPoints : public FEstimationMethod {
public:
    FEstimatorHPoints(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name);
    bool compute();
    int extractMatches();

private:

    std::vector<Point2d> x1_used, x2_used;
    std::vector<Point2d> x1, x2;    //corresponding points in image 1 and 2
    lineSubsetStruct estimateHomography(std::vector<Point2d> x1, std::vector<Point2d> x2, int method, int sets);
    pointSubsetStruct calcRANSAC(std::vector<pointSubsetStruct> &subsets, double threshold, std::vector<Point2d> x1, std::vector<Point2d> x2);
    pointSubsetStruct calcLMedS(std::vector<lineSubsetStruct> &subsets, std::vector<Point2d> x1, std::vector<Point2d> x2);
    double calcMedS(pointSubsetStruct &subset, std::vector<Point2d> x1, std::vector<Point2d> x2);
    double squaredSymmeticTransferPointError(Mat H, Point2d x1, Point2d x2);
};

#endif // FESTIMATORHPOINTS_H
