#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "MCE.h"

using namespace cv;

MCE::MCE(int argc, char** argv)
{
    this->arguments = argc;
    this->paths = argv;
}

void MCE::run() {
    loadData();
    extractSIFT();

}

void MCE::loadData() {
    if ( arguments != 3 )
    {
        printf("usage: MultipleQueEstimation <Path_to_first_image> <Path_to_second_image>\n");
        return;
    }

    image_1 = imread( paths[1], CV_LOAD_IMAGE_GRAYSCALE );
    image_2 = imread( paths[2], CV_LOAD_IMAGE_GRAYSCALE );

    if ( !image_1.data || !image_2.data )
    {
        printf("No image data \n");
        return;
    }
    namedWindow("Image 1", CV_WINDOW_AUTOSIZE );
    imshow("Image 1", image_1);

    namedWindow("Image 2", CV_WINDOW_AUTOSIZE);
    imshow("Image 2", image_2);

    waitKey(0);

    return;
}

void MCE::extractSIFT() {

    //Source: http://docs.opencv.org/doc/tutorials/features2d/feature_flann_matcher/feature_flann_matcher.html

    SurfFeatureDetector detector(SIFT_FEATURE_COUNT);
    detector.detect(image_1, keypoints_1);
    detector.detect(image_2, keypoints_2);

    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;

    Mat descriptors_1, descriptors_2;

    extractor.compute( image_1, keypoints_1, descriptors_1 );
    extractor.compute( image_2, keypoints_2, descriptors_2 );

    printf("-- Keypoints 1 : %i \n", keypoints_1.size() );
    printf("-- Keypoints 2 : %i \n", keypoints_2.size() );

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

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    std::vector< DMatch > good_matches;

    printf("-- Overall matches : %i \n", descriptors_1.rows );

    for( int i = 0; i < descriptors_1.rows; i++ )
    {
        if( matches[i].distance <= max(2*min_dist, 0.02) )
        {
            good_matches.push_back( matches[i]);
        }
    }

    printf("-- Number of matches : %i \n", good_matches.size() );

    //-- Draw only "good" matches
    Mat img_matches;
    drawMatches( image_1, keypoints_1, image_2, keypoints_2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    //-- Show detected matches
    namedWindow("SURF results", CV_WINDOW_AUTOSIZE);
    imshow( "SURF results", img_matches );
}

void MCE::calcFwithPoints() {

}
