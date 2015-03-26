#ifndef DATASTRUCTURES_H
#define DATASTRUCTURES_H

#include <opencv2/opencv.hpp>

using namespace cv;

struct lineCorrespStruct {
    double line1Angle, line2Angle, line1Length, line2Length;
    Mat line1StartNormalized, line1EndNormalized, line2StartNormalized, line2EndNormalized;
    Mat line1Start, line1End, line2Start, line2End;
    int id;
};

struct lineCorrespSubsetError {
    int lineCorrespIdx;
    double lineCorrespError;
};

struct lineSubsetStruct {
    std::vector<lineCorrespStruct> lineCorrespondencies;
    double qualityMeasure;
    double meanSquaredSymmeticTransferError;
    Mat Hs, Hs_normalized;
};

struct pointSubsetStruct {
    std::vector<Point2d> x1;
    std::vector<Point2d> x2;
    double qualityMeasure;
    double meanSquaredSymmeticTransferError;
    Mat Hs, Hs_normalized;
};

struct segmentStruct {
    int id;
    int area;
    Point startpoint;
    int contours_idx;
};

#endif
