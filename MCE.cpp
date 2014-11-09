#include "MCE.h"
#define DEBUG false

using namespace cv;

MCE::MCE(int argc, char** argv)
{
    this->arguments = argc;
    if (arguments == 5) {
        this->path_P1 = argv[3];
        this->path_P2 = argv[4];
    }
    this->path_img1 = argv[1];
    this->path_img2 = argv[2];
    this->lineCorrespondencies = new std::vector<CvPoint>();
}

void MCE::run() {
    Mat Fpt, Fgt;       //Fundamental matric from point correspondencies
    if (loadData()) {
        extractSIFT();
//        extractLines();

        Fpt = calcFfromPoints();




        if (arguments == 5) {   //Compare to ground truth

            Fgt = getGroundTruth();
            std::cout << "Fgt = " << std::endl << Fgt << std::endl;

        }

        std::cout << "Fpt = " << std::endl << Fpt << std::endl;
        std::cout << "Average Squared Error (Fpt) = " << averageSquaredError(Fgt,Fpt) << std::endl;

        rectify(x1, x2, Fgt, image_1, 1, "Image 1 rect Fgt");
        rectify(x1, x2, Fpt, image_1, 1, "Image 1 rect Fpt");

        drawEpipolarLines(x1, x2, Fgt, image_1.clone(), image_2.clone());

        waitKey(0);
        //Mat h = findHomography(x1, x2, noArray(), CV_RANSAC, 3);
        //compare with "real" Fundamental Matrix or calc lush point cloud?:
        //void triangulatePoints(InputArray projMatr1, InputArray projMatr2, InputArray projPoints1, InputArray projPoints2, OutputArray points4D)¶
    }

}

int MCE::loadData() {
    if (arguments != 3 && arguments != 5) {
        std::cout << "Usage: MultipleCueEstimation <path to first image> <path to second image> <optional: path to first camera matrix> <optional: path to second camera matrix>" << std::endl;
        return 0;
    }

    image_1 = imread(path_img1, CV_LOAD_IMAGE_GRAYSCALE);
    image_2 = imread(path_img2, CV_LOAD_IMAGE_GRAYSCALE);

    if ( !image_1.data || !image_2.data )
    {
        printf("No image data \n");
        return 0;
    }
    namedWindow("Image 1 original", CV_WINDOW_NORMAL);
    imshow("Image 1 original", image_1);

    namedWindow("Image 2 original", CV_WINDOW_NORMAL);
    imshow("Image 2 original", image_2);

    return 1;
}

void MCE::extractLines() {
    LineMatcher lm;

    Mat image_1_down;
    Mat image_2_down;

    std::string path_img1_down = path_img1 + "_down";
    std::string path_img2_down = path_img2 + "_down";

    //TODO: Memory leak, works only with small images (e.g. 800x600), replace with opencv if ver 3 is out
    resize(imread(path_img1, CV_LOAD_IMAGE_COLOR), image_1_down, Size(0,0), 0.25, 0.25, INTER_NEAREST);
    resize(imread(path_img1, CV_LOAD_IMAGE_COLOR), image_2_down, Size(0,0), 0.25, 0.25, INTER_NEAREST);

    imwrite(path_img1_down, image_1_down);
    imwrite(path_img2_down, image_2_down);


    std::cout << "Extracting line correspondencies..." << std::endl;
    int corresp = lm.match(path_img1_down.c_str(), path_img1_down.c_str(), lineCorrespondencies);       //TODO image scales to 25% -> multiply line coords by 4
    std::cout << "Found " << corresp << " line correspondencies " << std::endl;

    namedWindow("Image 1 lines", CV_WINDOW_AUTOSIZE);
    imshow("Image 1 lines", imread("LinesInImage1.png", CV_LOAD_IMAGE_COLOR));

    namedWindow("Image 2 lines", CV_WINDOW_AUTOSIZE);
    imshow("Image 2 lines", imread("LinesInImage2.png", CV_LOAD_IMAGE_COLOR));

    namedWindow("Line matches", CV_WINDOW_AUTOSIZE);
    imshow("Line matches", imread("LBDSG.png", CV_LOAD_IMAGE_COLOR));
}

void MCE::extractSIFT() {

    // ++++ Source: http://docs.opencv.org/doc/tutorials/features2d/feature_flann_matcher/feature_flann_matcher.html

    std::cout << "EXTRACTING SURF FEATURES:" << std::endl;

    std::vector<cv::KeyPoint> keypoints_1;
    std::vector<cv::KeyPoint> keypoints_2;

    SurfFeatureDetector detector(SIFT_FEATURE_COUNT);
    detector.detect(image_1, keypoints_1);
    detector.detect(image_2, keypoints_2);

    std::cout << "-- Keypoints 1 : " << keypoints_1.size() << std::endl;
    std::cout << "-- Keypoints 2 : " << keypoints_2.size() << std::endl;

    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;

    Mat descriptors_1, descriptors_2;

    extractor.compute( image_1, keypoints_1, descriptors_1 );
    extractor.compute( image_2, keypoints_2, descriptors_2 );

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_1.rows; i++ )
    { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
    }

    std::cout << "-- Max dist : " << max_dist << std::endl;
    std::cout << "-- Min dist : " << min_dist << std::endl;

    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    std::vector< DMatch > good_matches;

    std::cout << "-- Overall matches : " << descriptors_1.rows << std::endl;

    for( int i = 0; i < descriptors_1.rows; i++ )
    {
        if( matches[i].distance <= max(2*min_dist, 0.02) )
        {
            good_matches.push_back( matches[i]);
        }
    }

    std::cout << "-- Number of matches : " << good_matches.size() << std::endl;

    //-- Draw only "good" matches
    Mat img_matches;
    drawMatches( image_1, keypoints_1, image_2, keypoints_2,
               good_matches, img_matches );

    //-- Show detected matches
    namedWindow("SURF results", CV_WINDOW_NORMAL);
    imshow( "SURF results", img_matches );

    // ++++

    std::cout << "Create point pair of matches" << std::endl;
    for( int i = 0; i < good_matches.size(); i++ )
    {
        x1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
        x2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
    }

}

Mat MCE::calcFfromPoints() {
    std::cout << "Calc Fpt..." << std::endl;
    return findFundamentalMat(x1, x2, FM_RANSAC, 3.0, 0.99, noArray());
}

Mat MCE::MatFromFile(std::string file, int rows) {

    Mat matrix;
    std::ifstream inputStream;
    double x;
    inputStream.open(file.c_str());
    if (inputStream.is_open()) {
        while(inputStream >> x) {
            matrix.push_back(x);
        }
        matrix = matrix.reshape(1, rows);
        inputStream.close();
    } else {
        std::cerr << "Unable to open file: " << file;
    }
    return matrix;
}

void MCE::PointsToFile(std::vector<Point2d>* points, std::string file) {

    Point2d point;
    std::ofstream outputStream;
    outputStream.open(file.c_str());
    for (int i = 0; points->size(); i++) {
            point = points->at(i);
            outputStream << point.x;
            outputStream << ' ';
            outputStream << point.y;
            outputStream << '\n';
    }
    outputStream.flush();
    outputStream.close();
}

Mat MCE::crossProductMatrix(Mat input) {    //3 Vector to cross procut matrix
    Mat crossMat = Mat::zeros(3,3, input.type());
    crossMat.at<double>(0,1) = -input.at<double>(2);
    crossMat.at<double>(0,2) = input.at<double>(1);
    crossMat.at<double>(1,0) = input.at<double>(2);
    crossMat.at<double>(1,2) = -input.at<double>(0);
    crossMat.at<double>(2,0) = -input.at<double>(1);
    crossMat.at<double>(2,1) = input.at<double>(0);
    return crossMat;
}

void MCE::rectify(std::vector<Point2f> p1, std::vector<Point2f> p2, Mat F, Mat image, int imgNum, std::string windowName) {
    Mat H1, H2, H, rectified;
    if(stereoRectifyUncalibrated(p1, p2, F, Size(image.cols,image.rows), H1, H2, 0 )) {
        if (DEBUG) {
            std::cout << "H1 = " << std::endl << H1 << std::endl;
            std::cout << "H2 = " << std::endl << H2 << std::endl;
        }

        if (imgNum == 1) H = H1;
        else H = H2;

        warpPerspective(image, rectified, H, Size(image.cols,image.rows));

        namedWindow(windowName, CV_WINDOW_NORMAL);
        imshow(windowName, rectified);
    }
}

Mat MCE::getGroundTruth() {
    Mat P1w = MatFromFile(path_P1, 3); //P1 in world coords
    Mat P2w = MatFromFile(path_P2, 3); //P2 in world coords
    Mat T1w, T2w, R1w, R2w;   //World rotation, translation
    Mat K1, K2, K; //calibration matrices
    Mat Rrel, Trel; //Relative rotation, translation

    if (DEBUG) {
        std::cout << "P1w = " << std::endl << P1w << std::endl;
        //std::cout << "P2w = " << std::endl << P2w << std::endl;
    }

    decomposeProjectionMatrix(P1w, K1, R1w, T1w, noArray(), noArray(), noArray(), noArray() );
    //decomposeProjectionMatrix(P2w, K2, R2w, T2w, noArray(), noArray(), noArray(), noArray() );

    //K = (K1 + K2)/2;    //Images with same K

    T1w = T1w/T1w.at<double>(3);      //convert to homogenius coords
    //T1w.resize(3);

    //T2w = T2w/T2w.at<double>(3);      //convert to homogenius coords
    //T2w.resize(3);

//    R2w = R2w.t();      //switch rotation: world to 2. cam frame (Rc2w) to 2. cam to world frame (Rwc2)
//    R1w = R1w.t();      //switch rotation: world to 1. cam frame (Rc1w) to 1. cam to world frame (Rwc1)



    if (DEBUG) {
        std::cout << "T1w = " << std::endl << T1w << std::endl;
        //std::cout << "T2w = " << std::endl << T2w << std::endl;

        std::cout << "R1w = " << std::endl << R1w << std::endl;
        //std::cout << "R2w = " << std::endl << R2w << std::endl;
    }

    //Rrel = R1w*R2w.t(); //Relative rotation between cam1 and cam2; Rc1c2 = Rwc1^T * Rwc2

    if (DEBUG) {
        //std::cout << "K = " << std::endl << K << std::endl;
        //std::cout << "K2 = " << std::endl << K2 << std::endl;
    }

    //Trel = T2w - T1w;    //Realtive translation between cam1 and cam2

    if (DEBUG) {
        //std::cout << "Rrel = " << std::endl << Rrel << std::endl;
        std::cout << "Trel = " << std::endl << Trel << std::endl;
    }

    Mat F = crossProductMatrix(P2w*T1w)*P2w*P1w.inv(DECOMP_SVD);

//    Mat C = (Mat_<double>(4,1) << 0, 0, 0, 1.0);
//    Mat e = P2w*C;
//    std::cout << "e = " << std::endl << e << std::endl;
//    return crossMatrix(e)*P2w*P1w.inv(DECOMP_SVD);

//    F = K.t().inv()*crossProductMatrix(Trel)*Rrel*K.inv();
//    //return K.t().inv()*crossProductMatrix(Trel)*Rrel*K.inv();
//    //return K2.t().inv()*crossProductMatrix(Trel)*Rrel*K1.inv(); //(See Hartley, Zisserman: p. 244)
//    F = K2.t().inv()*Rrel*K1.t()*crossProductMatrix(K1*Rrel.t()*Trel);
    return F / F.at<double>(2,2);       //Set global arbitrary scale factor 1 -> easier to compare
}

void MCE::drawEpipolarLines(std::vector<Point2f> p1, std::vector<Point2f> p2, Mat F, Mat image1, Mat image2) {

    //#################################################################################
    //From: http://opencv-cookbook.googlecode.com/svn/trunk/Chapter%2009/estimateF.cpp
    //#################################################################################

    std::vector<cv::Vec3f> lines1, lines2;
    cv::computeCorrespondEpilines(p1, 1, F, lines1);
    for (vector<cv::Vec3f>::const_iterator it= lines1.begin();
         it!=lines1.end(); ++it) {

             cv::line(image2,cv::Point(0,-(*it)[2]/(*it)[1]),
                             cv::Point(image2.cols,-((*it)[2]+(*it)[0]*image2.cols)/(*it)[1]),
                             cv::Scalar(255,255,255));
    }

    cv::computeCorrespondEpilines(p2,2,F,lines2);
    for (vector<cv::Vec3f>::const_iterator it= lines2.begin();
         it!=lines2.end(); ++it) {

             cv::line(image1,cv::Point(0,-(*it)[2]/(*it)[1]),
                             cv::Point(image1.cols,-((*it)[2]+(*it)[0]*image1.cols)/(*it)[1]),
                             cv::Scalar(255,255,255));
    }

    // Draw the inlier points
//    std::vector<cv::Point2d> points1In, points2In;
//    std::vector<cv::Point2d>::const_iterator itPts= p1.begin();
//    std::vector<uchar>::const_iterator itIn= inliers.begin();
//    while (itPts!=points1.end()) {

//        // draw a circle at each inlier location
//        if (*itIn) {
//            cv::circle(image1,*itPts,3,cv::Scalar(255,255,255),2);
//            points1In.push_back(*itPts);
//        }
//        ++itPts;
//        ++itIn;
//    }

//    itPts= p2.begin();
//    itIn= inliers.begin();
//    while (itPts!=points2.end()) {

//        // draw a circle at each inlier location
//        if (*itIn) {
//            cv::circle(image2,*itPts,3,cv::Scalar(255,255,255),2);
//            points2In.push_back(*itPts);
//        }
//        ++itPts;
//        ++itIn;
//    }

    // Display the images with points
    cv::namedWindow("Right Image Epilines", CV_WINDOW_NORMAL);
    cv::imshow("Right Image Epilines",image1);
    cv::namedWindow("Left Image Epilines", CV_WINDOW_NORMAL);
    cv::imshow("Left Image Epilines",image2);

    //#############################################################################
}

std::string MCE::getType(Mat m) {
    std::string type = "Type: ";
    switch(m.type() & TYPE_MASK) {
        case CV_8U: "CV_8U"; break;
        case CV_8S: type+="CV_8U";  break;
        case CV_16U: type+="CV_16U"; break;
        case CV_16S: type+="CV_16S"; break;
        case CV_32S: type+="CV_32S"; break;
        case CV_32F: type+="CV_32F"; break;
        case CV_64F: type+="CV_64F"; break;
        default: type+="unknown"; break;
    }
//    type+=", depth: ";
//    type+=(DEPTH_MASK & m.type());
    return type;
}

Scalar MCE::averageSquaredError(Mat A, Mat B) {
    return cv::sum((A-B).mul(A-B))/((double)(A.cols*A.rows));
}
