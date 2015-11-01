#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <cstdio>
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace cv;

Mat matVector(Point2d p) {
    Mat vect = Mat::zeros(3,1,CV_64FC1);
    vect.at<double>(0,0) = p.x;
    vect.at<double>(1,0) = p.y;
    vect.at<double>(2,0) = 1;
    return vect;
}

double sampsonDistanceFundamentalMat(Mat F, Mat x1, Mat x2) {      //See: Hartley Ziss, p287
    //Mat x1 = matVector(p1);
    //Mat x2 = matVector(p2);
    double n = Mat(x2.t()*F*x1).at<double>(0,0);
    Mat b1 = F*x1;
    Mat b2 = F.t()*x2;
    return std::pow(n, 2)/(std::pow(b1.at<double>(0,0), 2) + std::pow(b1.at<double>(1,0), 2) + std::pow(b2.at<double>(0,0), 2) + std::pow(b2.at<double>(1,0), 2));
}

void findGoodCombinedMatches(std::vector<Point2d> x1Combined, std::vector<Point2d> x2Combined, std::vector<Point2d> &x1, std::vector<Point2d> &x2, Mat F, double maxDist, double minDist) {
    x1.clear();
    x2.clear();
    for(int i = 0; i < x1Combined.size(); i++) {
        double err = sqrt(sampsonDistanceFundamentalMat(F, matVector(x1Combined.at(i)), matVector(x2Combined.at(i))));
        if(minDist < err && err < maxDist) {
            x1.push_back(x1Combined.at(i));
            x2.push_back(x2Combined.at(i));
        }
    }
}

int extractPointMatches(Mat image_1, Mat image_2, std::vector<Point2d> &x1, std::vector<Point2d> &x2) {
    std::vector<cv::KeyPoint> keypoints_1;
    std::vector<cv::KeyPoint> keypoints_2;

    //SurfFeatureDetector detector(SIFT_FEATURE_COUNT);
    Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(800);

    detector->detect(image_1, keypoints_1);
    detector->detect(image_2, keypoints_2);

    //-- Step 2: Calculate descriptors (feature vectors)
    Ptr<xfeatures2d::SurfDescriptorExtractor> extractor = xfeatures2d::SurfDescriptorExtractor::create();

    Mat descriptors_1, descriptors_2;

    extractor->compute( image_1, keypoints_1, descriptors_1 );
    extractor->compute( image_2, keypoints_2, descriptors_2 );

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    for( int i = 0; i < descriptors_1.rows; i++ )
    {
        x1.push_back(keypoints_1[matches[i].queryIdx].pt);
        x2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    return x1.size();
}


bool MatFromFile(std::string file, Mat &M) {
    M = Mat::zeros(3,3,CV_64FC1);
    std::ifstream inputStream;
//    FileStorage fs(file,FileStorage::READ);
//    fs >> M;
    try {
        inputStream.open(file.c_str());
    } catch(...) {
        std::cout << "fail" << std::endl;
        return false; }
    int n = 0;
    std::string row;
    if (inputStream.good() && inputStream.is_open()) {
        while ( inputStream.good() )
        {
            std::getline(inputStream, row);
            int end = row.find_first_of(',');
            std::string x1 = row.substr(1,end);
            M.at<double>(n, 0) = std::atof(x1.c_str());
            int end2 = row.find_first_of(',', end+1);
            x1 = row.substr(end+1,end2-end);
            M.at<double>(n, 1) = std::atof(x1.c_str());
            int end3;
            if(n==2) end3 = row.find_first_of(']', end2);
            else end3 = row.find_first_of(';', end2);
            x1 = row.substr(end2+1,end3-end2-1);
            M.at<double>(n, 2) = std::atof(x1.c_str());
            n++;
        }
//        while(inputStream >> x) {
//            M.push_back(x);
//        }
        //std::cout << M << std::endl;
        return true;
    } else {
        //std::cerr << "Unable to open file: " << file <<std::endl;
        return false;
    }

}

bool ImgParamsFromFile(std::string file, Mat &K, Mat &R, Mat &t) {
    int values = 1;
    std::ifstream inputStream;
    double x;
    inputStream.open(file.c_str());
    if (inputStream.is_open()) {
        while(inputStream >> x) {
            if(values >= 1 && values <= 9) {    //K
                K.push_back(x);
            } else if(values >= 13 && values <= 21) { //R
                R.push_back(x);
            } else if(values >= 22 && values <= 24) { //t
                t.push_back(x);
            }
            values++;
        }
        inputStream.close();
        K = K.reshape(1, 3);
        R = R.reshape(1, 3);
        t = t.reshape(1, 3);
        return true;
    } else {
        std::cerr << "Unable to open file: " << file <<std::endl;
        return false;
    }
}

void calcRotations(Mat R, double &rotX, double &rotY, double &rotZ) {
    double r11 = R.at<double>(0,0);
    double r21 = R.at<double>(1,0);
    double r32 = R.at<double>(2,1);
    double r33 = R.at<double>(2,2);
    double r31 = R.at<double>(2,0);

    rotX = atan2(r32,r33);
    rotY = atan2(-r31,sqrt(r32*r32 + r33*r33));
    rotZ = atan2(r21,r11);
}

void CamDist(std::vector<Point2d> x1True, std::vector<Point2d> x2True, Mat K, Mat Fgt, Mat Ftest, Point2d pp, double f) {
    Mat Etest1 = K.t()*Ftest*K;
    Mat Egt1 = K.t()*Fgt*K;
    Mat Etest2 = K.t()*Ftest.t()*K;
    Mat Egt2 = K.t()*Fgt.t()*K;

    Mat Rtest1, ttest1, Rgt1, tgt1, Rtest2, ttest2, Rgt2, tgt2;

    recoverPose(Etest1, x1True, x2True, Rtest1, ttest1, f, pp);
    recoverPose(Egt1, x1True, x2True, Rgt1, tgt1, f, pp);

    recoverPose(Etest2, x2True, x1True, Rtest2, ttest2, f, pp);
    recoverPose(Egt2, x2True, x1True, Rgt2, tgt2, f, pp);

    Mat Rgt12 = Rgt2.t()*Rgt1;                  //Rotation from camera 1 to camera 2
    Mat Tgt12 = Rgt2.t()*Mat(tgt2 - tgt1);      //Translation from camera 1 to camera 2

    Mat Rtest12 = Rtest2.t()*Rtest1;                  //Rotation from camera 1 to camera 2
    Mat Ttest12 = Rtest2.t()*Mat(ttest2 - ttest1);      //Translation from camera 1 to camera 2

    double rotgtX, rotgtY, rotgtZ, rottestX, rottestY, rottestZ;

    calcRotations(Rgt12, rotgtX, rotgtY, rotgtZ);
    calcRotations(Rtest12, rottestX, rottestY, rottestZ);

    //std::cout << "," << norm(Tgt12,Ttest12,NORM_L2) << "," << sqrt(pow(rotgtX - rottestX,2) + pow(rotgtY - rottestY,2) + pow(rotgtZ - rottestZ,2)) << "," << rotgtX << "," << rotgtY<< "," <<  rotgtZ<< "," <<  rottestX<< "," <<  rottestY<< "," <<  rottestZ << ",";
    std::cout << norm(Tgt12,Ttest12,NORM_L2) << "," << sqrt(pow(rotgtX - rottestX,2) + pow(rotgtY - rottestY,2) + pow(rotgtZ - rottestZ,2)) << ",";

    //std::cout << "Rtest = " << std::endl << Rtest << std::endl << "ttest = " << std::endl << ttest << std::endl;
    //std::cout << "Rgt = " << std::endl << Rgt << std::endl << "tgt = " << std::endl << tgt << std::endl;

    //return norm(Ttest12,Tgt12,NORM_L2);
}

int main(int argc, char** argv )
{
    Mat Fgt, Ffinal, Fpoints, Fhpoints, Fhlines, K, R, t;
    MatFromFile(argv[1], Fgt);
    ImgParamsFromFile(argv[2], K, R, t);

    Mat image_1 = imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE).clone();
    Mat image_2 = imread(argv[4], CV_LOAD_IMAGE_GRAYSCALE).clone();

    MatFromFile(argv[5], Ffinal);
    bool hasPoints = MatFromFile(argv[6], Fpoints);
    bool hasHlines = MatFromFile(argv[7], Fhlines);
    bool hasHpoints = MatFromFile(argv[8], Fhpoints);

    if(!Fgt.data || !Ffinal.data || !K.data || !image_1.data || ! image_2.data ) {
        std::cerr << "Error input data!" << std::endl;
        return -1;
    }

    double f = (K.at<double>(0,0) + K.at<double>(1,1))/2.0;
    Point2d pp;
    pp.x = K.at<double>(0, 2);
    pp.y = K.at<double>(1, 2);

    //std::cout << "K = " << std::endl << K << std::endl;

    //std::cout << "Principle point: (" << pp.x << "," << pp.y << "), focal lenght: " << f << std::endl;

    std::vector<Point2d> x1;
    std::vector<Point2d> x2;

    std::vector<Point2d> x1True;
    std::vector<Point2d> x2True;

    int matches = extractPointMatches(image_1, image_2, x1, x2);
    findGoodCombinedMatches(x1, x2, x1True, x2True, Fgt, 1.0, 0.0);

    //std::cout << "Point matches: " << matches << ", true matches: " << x1True.size() << std::endl;

    std::cout << argv[2] << "," << argv[9] << ",";
    CamDist(x1True, x2True, K, Fgt, Ffinal, pp, f);
    if(hasPoints) CamDist(x1True, x2True, K, Fgt, Fpoints, pp, f);
    else std::cout << ",,";
    if(hasHlines) CamDist(x1True, x2True, K, Fgt, Fhlines, pp, f);
    else std::cout << ",,";
    if(hasHpoints) CamDist(x1True, x2True, K, Fgt, Fhpoints, pp, f);
    else std::cout << ",,";
    std::cout << "," << x1True.size() << std::endl;

//    std::cout << argv[2] << " " << argv[9] << "," << CamDist(x1True, x2True, K, Fgt, Ffinal, pp, f) << "," << CamDist(x2True, x1True, K, Fgt.t(), Ffinal.t(), pp, f);
//                         if(hasPoints) std::cout << "," << CamDist(x1True, x2True, K, Fgt, Fpoints, pp, f) << "," << CamDist(x2True, x1True, K, Fgt.t(), Fpoints.t(), pp, f);
//                         else std::cout << ",,";
//                         if(hasHlines) std::cout << "," << CamDist(x1True, x2True, K, Fgt, Fhlines, pp, f) << "," << CamDist(x2True, x1True, K, Fgt.t(), Fhlines.t(), pp, f);
//                         else std::cout << ",,";
//                         if(hasHpoints) std::cout << "," << CamDist(x1True, x2True, K, Fgt, Fhpoints, pp, f) << "," << CamDist(x2True, x1True, K, Fgt.t(), Fhpoints.t(), pp, f);
//                         else std::cout << ",,";
//                         std::cout << "," << x1True.size() << std::endl;
}
