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
#include <opencv2/opencv.hpp>

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
#define LOG_DEBUG false
#define VISUAL_DEBUG true

using namespace cv;

class MCE
{
public:
    MCE(int argc, char** argv);

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

    Mat MatFromFile(std::string file, int cols);
    void PointsToFile(std::vector<Point2d>* points, std::string file);
    Mat crossProductMatrix(Mat input);
    void rectify(std::vector<Point2f> p1, std::vector<Point2f> p2, Mat F, Mat image, int imgNum, std::string windowName);
    Mat getGroundTruth();
    void drawEpipolarLines(std::vector<Point2f> p1, std::vector<Point2f> p2, Mat F, Mat image1, Mat image2);
    std::string getType(Mat m);
    Scalar averageSquaredError(Mat A, Mat B);

private:

    int arguments;
    std::string path_img1, path_P1;
    std::string path_img2, path_P2;
    Mat image_1, image_2;
    Mat image_1_color, image_2_color;
    vector<CvPoint>* lineCorrespondencies;
    std::vector<Point2f> x1, x2;   //corresponding points in image 1 and 2
};

#endif // MCE_MAIN_H
