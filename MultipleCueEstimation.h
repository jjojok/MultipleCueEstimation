#ifndef MCE_H
#define MCE_H

//ESTIMATORS
#include "FEstimatorPoints.h"

//SYSTEM
#include <string>
#include <cstdio>
#include <stdio.h>
#include <iostream>
#include <fstream>

//OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "opencv2/core/utility.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/line_descriptor.hpp>

#define SIFT_FEATURE_COUNT 800
#define LOG_DEBUG true
#define VISUAL_DEBUG true
#define F_FROM_POINTS 1
#define F_FROM_LINES 2
#define F_FROM_PLANES 4

//Line Matching:
//Number of segements for image pyramid
#define OCTAVES 2
//Scaling factor per segement
#define SCALING 1
//minimal line lenght = width*height*MIN_LENGTH_FACTOR
#define MIN_LENGTH_FACTOR 0.00001;
//defines number of subsets which are randomly picked to compute a Homography each. Homographies = Number of matches*NUM_OF_PAIR_SUBSETS_FACTOR
#define NUM_OF_PAIR_SUBSETS_FACTOR 4

using namespace cv;

struct segmentStruct {
    int id;
    int area;
    Point startpoint;
    int contours_idx;
};

struct matrixStruct {
    std::string source;
    Mat F;
    float error;
};

struct lineCorrespStruct {
    cv::line_descriptor::KeyLine line1, line2;
    Point2f line1StartNormalized, line1EndNormalized, line2StartNormalized, line2EndNormalized;
};

struct lineSubsetStruct {
    std::vector<lineCorrespStruct> lineCorrespondencies;
    Mat Hs;
};

class MultipleCueEstimation
{
public:
    MultipleCueEstimation();

    void run();
    int loadData();
    void extractPoints();
    void extractLines();
    void extractPlanes();
    Mat calcFfromPoints();
    Mat calcFfromLines();
    Mat calcFfromPlanes();
    Mat calcFfromConics();
    Mat calcFfromCurves();
    Mat refineF();

    //Utility finctions:

    void findSegments(Mat image, Mat image_color, std::string image_name, std::vector<segmentStruct> &segmentList, Mat &segments);
    Mat MatFromFile(std::string file, int cols);
    void PointsToFile(std::vector<Point2f>* points, std::string file);
    Mat crossProductMatrix(Mat input);
    void rectify(std::vector<Point2f> p1, std::vector<Point2f> p2, Mat F, Mat image, int imgNum, std::string windowName);
    Mat getGroundTruth();
    void drawEpipolarLines(std::vector<Point2f> p1, std::vector<Point2f> p2, Mat F, Mat image1, Mat image2);
    std::string getType(Mat m);
    Scalar averageSquaredError(Mat A, Mat B);
    Scalar squaredError(Mat A, Mat B);
    Point2f normalize(float x, float y, int img_width, int img_height);
    void fillHLinEq(Mat* linEq, lineCorrespStruct lc, int numPair);
    void fillHLinEqBase(Mat* linEq, float x, float y, float A, float B, float C, int row);
    Mat calcLMedS(std::vector<lineSubsetStruct> subsets);
    float calcMedS(Mat Hs);
    int calcMatRank(Mat M);
    int calcNumberOfSolutions(Mat linEq);
    double epipolarSADError(std::vector<Point2f> p1, std::vector<Point2f> p2, Mat F);
    void showImage(std::string name, Mat image, int type = WINDOW_NORMAL, int width = 800, int height = 0);
    int filterLineExtractions(std::vector<cv::line_descriptor::KeyLine>* keylines);
    void filterLineMatches(cv::Mat descr1, cv::Mat descr2, std::vector<DMatch> matches);

    int arguments;
    std::string path_img1, path_P1;
    std::string path_img2, path_P2;
    unsigned int computations;
    bool compareWithGroundTruth;

private:

    Mat image_1, image_2;
    Mat image_1_color, image_2_color;
    std::vector<lineCorrespStruct> lineCorrespondencies;  //Vector of consecutive normalized line startpoints and endpoints
    std::vector<Point2f> x1, x2;   //corresponding points in image 1 and 2
};

#endif // MCE_MAIN_H
