#include "MCE.h"

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
    Mat Fpt;       //Fundamental matric from point correspondencies
    if (loadData()) {
        extractSIFT();
        extractLines();

        Fpt = calcFfromPoints();

        std::cout << "Fpt = " << std::endl << Fpt << std::endl;

        Mat H1, H2;
        Mat rectified;// = Mat::zeros(Size(image_1.cols,image_1.rows), image_1.type());


        if(stereoRectifyUncalibrated(x1, x2, Fpt, Size(image_1.cols,image_1.rows), H1, H2, 5 )) {
            std::cout << "H1 = " << std::endl << H1 << std::endl;
            std::cout << "H2 = " << std::endl << H2 << std::endl;

            warpPerspective(image_1, rectified, H1, Size(image_1.cols,image_1.rows));

            namedWindow("Image 1 rectified", CV_WINDOW_AUTOSIZE );
            imshow("Image 1 rectified", rectified);

        }

        if (arguments == 5) {   //Compare to ground truth
            //Mat K1 = MatFromFile(path_P1);
            //Mat K2 = MatFromFile(path_P2);


        }

        waitKey(0);
        //Mat h = findHomography(x1, x2, noArray(), CV_RANSAC, 3);
        //compare with "real" Fundamental Matrix or calc lush point cloud?:
        //void triangulatePoints(InputArray projMatr1, InputArray projMatr2, InputArray projPoints1, InputArray projPoints2, OutputArray points4D)¶
    }

}

int MCE::loadData() {
    if (arguments != 3 || arguments != 5) {
        std::cout << "Usage: multiplecueestimation <path to first image> <path to second image> <optional: path to first camera matrix> <optional: path to second camera matrix>" << std::endl;
        return 0;
    }

    image_1 = imread(path_img1, CV_LOAD_IMAGE_GRAYSCALE);
    image_2 = imread(path_img2, CV_LOAD_IMAGE_GRAYSCALE);

    if ( !image_1.data || !image_2.data )
    {
        printf("No image data \n");
        return 0;
    }
    namedWindow("First original image", CV_WINDOW_AUTOSIZE );
    imshow("First original image", image_1);

    namedWindow("Second original image", CV_WINDOW_AUTOSIZE);
    imshow("Second original image", image_2);

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
    int corresp = lm.match(path_img1_down.c_str(), path_img1_down.c_str(), lineCorrespondencies);
    std::cout << "Found " << corresp << " line correspondencies " << std::endl;

    namedWindow("Image 1 lines", CV_WINDOW_AUTOSIZE );
    imshow("Image 1 lines", imread("LinesInImage1.png", CV_LOAD_IMAGE_COLOR));

    namedWindow("Image 2 lines", CV_WINDOW_AUTOSIZE );
    imshow("Image 2 lines", imread("LinesInImage2.png", CV_LOAD_IMAGE_COLOR));

    namedWindow("Line matches", CV_WINDOW_AUTOSIZE );
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
    namedWindow("SURF results", CV_WINDOW_AUTOSIZE);
    imshow( "SURF results", img_matches );

    // ++++

    std::cout << "Create point pair of matches" << std::endl;
    for( int i = 0; i < good_matches.size(); i++ )
    {
//        std::cout << "good_matches[" << i << "].queryIdx=" << good_matches[i].queryIdx << std::endl;
//        std::cout << "good_matches[" << i << "].trainIdx=" << good_matches[i].trainIdx << std::endl;
        x1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
        x2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
    }

}

Mat MCE::calcFfromPoints() {
    std::cout << "Calc Fpt..." << std::endl;
    return findFundamentalMat(x1, x2, FM_RANSAC, 2., 0.999, noArray());
}

Mat MCE::MatFromFile(std::string file, int cols) {

    Mat matrix;
    std::ifstream inputStream;
    float x;
    inputStream.open(file.c_str());
    if (inputStream.is_open()) {
        //while (!inputStream.eof()) {
        while(inputStream >> x) {
            matrix.push_back(x);
        }
        matrix = matrix.reshape(1, cols);
        inputStream.close();
    } else {
        std::cerr << "Unable to open file: " << file;
    }

    return matrix;
}

std::vector<Mat> decomposeFtoK(Mat F) {
//    SVD svd();
//    Mat U, S, V;
//    svd.compute(F, U, S, V);
//    Mat e = U.col(3);
//    P = [-vgg_contreps(e)*F e];
}

void MCE::PointsToFile(std::vector<Point2f>* points, std::string file) {

    Point2f point;
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
