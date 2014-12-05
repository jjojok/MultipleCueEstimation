#ifndef MCE_H
#define MCE_H

//SYSTEM
#include <string>
#include <cstdio>
#include <stdio.h>
#include <iostream>
#include <fstream>

//OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/core.hpp>

//BIAS
#include <bias_config.h>
#include <FeatureDetector/ConstantRegionDetector.hh>
#include <FeatureDetector/LinearRegionDetector.hh>
#include <FeatureDetector/BlobDetectorDOM.hh>
#include <Base/ImageUtils/ImageDraw.hh>
#include <Base/Image/ImageConvert.hh>
#include <Base/Image/ImageIO.hh>
#include <Base/Image/WrapBias2Ipl.hh>

//OTHER
#include "LineMatchingSourceCode/LineMatcher.hh"

#define SIFT_FEATURE_COUNT 800
#define LOG_DEBUG true
#define VISUAL_DEBUG false
#define F_FROM_POINTS 1
#define F_FROM_LINES 2
#define F_FROM_PLANES 4

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
    Point2f line1Start, line1End, line2Start, line2End;
};

struct lineSubsetStruct {
    std::vector<lineCorrespStruct> lineCorrespondencies;
    Mat Hs;
};

class MCE
{
public:
    MCE();

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

    void findSegments(Mat image, Mat image_color, std::string image_name, Vector<segmentStruct> &segmentList, Mat &segments);
    Mat MatFromFile(std::string file, int cols);
    void PointsToFile(std::vector<Point2f>* points, std::string file);
    Mat crossProductMatrix(Mat input);
    void rectify(std::vector<Point2f> p1, std::vector<Point2f> p2, Mat F, Mat image, int imgNum, std::string windowName);
    Mat getGroundTruth();
    void drawEpipolarLines(std::vector<Point2f> p1, std::vector<Point2f> p2, Mat F, Mat image1, Mat image2);
    std::string getType(Mat m);
    Scalar averageSquaredError(Mat A, Mat B);
    Point2f normalize(Point2f p, int img_width, int img_height);
    void fillHLinEq(Mat* linEq, lineCorrespStruct lc, int numPair);
    void fillHLinEqBase(Mat* linEq, Point2f point, float A, float B, float C, int row);

    int arguments;
    std::string path_img1, path_P1;
    std::string path_img2, path_P2;
    unsigned int computations;
    bool compareWithGroundTruth;

private:

    Mat image_1, image_2;
    Mat image_1_color, image_2_color;
    std::vector<lineCorrespStruct> lineCorrespondencies;  //Vector of consecutive line startpoints and endpoints
    std::vector<Point2f> x1, x2;   //corresponding points in image 1 and 2
};

#endif // MCE_MAIN_H
