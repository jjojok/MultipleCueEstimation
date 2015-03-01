#include "FEstimatorPoints.h"

FEstimatorPoints::FEstimatorPoints(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name) {
    image_1 = img1.clone();
    image_2 = img2.clone();
    image_1_color = img1_c.clone();
    image_2_color = img2_c.clone();
    this->name = name;
    std::cout << "Estimating: " << name << std::endl;

    normT1 = Mat::eye(3,3,CV_32FC1);
    normT2 = Mat::eye(3,3,CV_32FC1);
}

int FEstimatorPoints::extractMatches() {
    std::vector<cv::KeyPoint> keypoints_1;
    std::vector<cv::KeyPoint> keypoints_2;

    //SurfFeatureDetector detector(SIFT_FEATURE_COUNT);
    Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(SIFT_FEATURE_COUNT);

    detector->detect(image_1, keypoints_1);
    detector->detect(image_2, keypoints_2);

    std::cout << "-- First image: " << keypoints_1.size() << std::endl;
    std::cout << "-- Second image: " << keypoints_2.size() << std::endl;

    //-- Step 2: Calculate descriptors (feature vectors)
    Ptr<xfeatures2d::SurfDescriptorExtractor> extractor = xfeatures2d::SurfDescriptorExtractor::create();

    Mat descriptors_1, descriptors_2;

    extractor->compute( image_1, keypoints_1, descriptors_1 );
    extractor->compute( image_2, keypoints_2, descriptors_2 );

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    if (LOG_DEBUG) {
        std::cout << "-- Max dist : " << max_dist << std::endl;
        std::cout << "-- Min dist : " << min_dist << std::endl;
    }
    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    std::vector< DMatch > good_matches;

    if(LOG_DEBUG) std::cout << "-- Overall matches : " << descriptors_1.rows << std::endl;

    for( int i = 0; i < descriptors_1.rows; i++ )
    {
        if( matches[i].distance <= max(SIFT_MIN_DIST_FACTOR*min_dist, SIFT_MIN_DIST) )
        {
            good_matches.push_back( matches[i]);
        }
    }

    std::cout << "-- Number of matches: " << good_matches.size() << std::endl;

    // ++++

    for( int i = 0; i < good_matches.size(); i++ )
    {
        x1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
        x2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
    }
}

Mat FEstimatorPoints::compute() {
    std::vector<bool> mask;
    int used = 0;
    extractMatches();
    F = findFundamentalMat(x1, x2, FM_RANSAC, 3.0, 0.99, mask);
    F.convertTo(F, CV_32FC1);
    for(int i = 0; i < x1.size(); i++) {
        if(mask.at(i)) {
            x1_used.push_back(x1.at(i));
            x2_used.push_back(x2.at(i));
            used++;
        }
    }
    std::cout << "-- Used matches (RANSAC): " << x1_used.size() << std::endl;

    visualizeMatches(getX1(), getX2(), 3, true, "Used matches");

    std::cout << std::endl;

    return F;
}

std::vector<Point2f> FEstimatorPoints::getX1() {
    return x1_used;
}

std::vector<Point2f> FEstimatorPoints::getX2() {
    return x2_used;
}

void FEstimatorPoints::visualizeMatches(std::vector<Point2f> p1, std::vector<Point2f> p2, int lineWidth, bool drawConnections, std::string name) {
    Mat img;
    hconcat(image_1_color.clone(), image_2_color.clone(), img);
    for(int i = 0; i < p1.size(); i++) {
        Scalar color = Scalar(rand()%255, rand()%255, rand()%255);
        cv::circle(img, p1.at(i), 2, color, lineWidth);
        cv::circle(img, cvPoint2D32f(p2.at(i).x + image_1_color.cols, p2.at(i).y), 2, color, lineWidth);
        //cv::line(img, p1.at(i), p1.at(i), color, lineWidth);
        //cv::line(img, cvPoint2D32f(p2.at(i).x + image_1_color.cols, p2.at(i).y), cvPoint2D32f(p2.at(i).x + image_1_color.cols, p2.at(i).y), color, lineWidth);
        if(drawConnections) {
            cv::line(img, p1.at(i), cvPoint2D32f(p2.at(i).x + image_1_color.cols, p2.at(i).y), color, lineWidth);
        }
    }
    showImage(name, img, WINDOW_NORMAL, 1600);
}
