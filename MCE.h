#ifndef MCE_H
#define MCE_H

#include <opencv2/opencv.hpp>
#include <string>
#include "LineMatchingSourceCode/LineMatcher.hh"
#include <cstdio>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <fstream>

#define SIFT_FEATURE_COUNT 800

using namespace cv;

class MCE
{
public:
    MCE(int argc, char** argv);

    void run();
    int loadData();
    void extractSIFT();
    void extractLines();
    Mat calcFfromPoints();
    Mat MatFromFile(std::string file, int cols);
    void PointsToFile(std::vector<Point2f>* points, std::string file);

private:

    int arguments;
    std::string path_img1, path_P1;
    std::string path_img2, path_P2;
    Mat image_1, image_2;
    vector<CvPoint>* lineCorrespondencies;
    std::vector<Point2f> x1, x2;   //corresponding points in image 1 and 2
};

#endif // MCE_MAIN_H
