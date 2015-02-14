#include "FEstimatorPoints.h"

FEstimatorPoints::FEstimatorPoints(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name) {
    epipolarError = -1;
    image_1 = img1.clone();
    image_2 = img2.clone();
    image_1_color = img1_c.clone();
    image_2_color = img2_c.clone();
    this->name = name;
    std::cout << "Estimating: " << name << std::endl;
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
    { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
    }

    if (LOG_DEBUG) {
        std::cout << "-- Max dist : " << max_dist << std::endl;
        std::cout << "-- Min dist : " << min_dist << std::endl;
    }
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

    std::cout << "-- Number of matches: " << good_matches.size() << std::endl;

    //-- Draw only "good" matches
    Mat img_matches;
    drawMatches( image_1_color, keypoints_1, image_2_color, keypoints_2,
               good_matches, img_matches );

    //-- Show detected matches
    if(VISUAL_DEBUG) {
        showImage( "SURF results", img_matches );
    }

    // ++++

    for( int i = 0; i < good_matches.size(); i++ )
    {
        x1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
        x2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
    }

    std::cout << std::endl;
}

Mat FEstimatorPoints::compute() {
    extractMatches();
    F = findFundamentalMat(x1, x2, FM_RANSAC, 3.0, 0.99, noArray());
    return F;
}
