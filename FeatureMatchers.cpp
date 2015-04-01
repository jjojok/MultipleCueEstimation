#include "FeatureMatchers.h"

int extractPointMatches(Mat image_1, Mat image_2, std::vector<Point2d> &x1, std::vector<Point2d> &x2) {
    std::vector<cv::KeyPoint> keypoints_1;
    std::vector<cv::KeyPoint> keypoints_2;

    //SurfFeatureDetector detector(SIFT_FEATURE_COUNT);
    Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(SIFT_FEATURE_COUNT);

    detector->detect(image_1, keypoints_1);
    detector->detect(image_2, keypoints_2);

    if(LOG_DEBUG) std::cout << "-- First image: " << keypoints_1.size() << std::endl;
    if(LOG_DEBUG) std::cout << "-- Second image: " << keypoints_2.size() << std::endl;

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

    min_dist = max(min_dist*SIFT_MIN_DIST_FACTOR, SIFT_MIN_DIST);

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
        if( matches[i].distance <= min_dist )
        {
            good_matches.push_back( matches[i]);
        }
    }

    if(LOG_DEBUG) std::cout << "-- Number of good matches: " << good_matches.size() << std::endl;

    // ++++

    for( int i = 0; i < good_matches.size(); i++ )
    {
        x1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
        x2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
    }

    return x1.size();
}

int extractLineMatches(Mat image_1, Mat image_2, std::vector<lineCorrespStruct> &allMatchedLines) {
    /********************************************************************
     * From: http://docs.opencv.org/trunk/modules/line_descriptor/doc/tutorial.html
     * ******************************************************************/

    /* create binary masks */
    cv::Mat mask1 = Mat::ones( image_1.size(), CV_8UC1 );
    cv::Mat mask2 = Mat::ones( image_2.size(), CV_8UC1 );

    /* create a pointer to a BinaryDescriptor object with default parameters */
    Ptr<cv::line_descriptor::BinaryDescriptor> bd = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();

    bd->setNumOfOctaves(OCTAVES);
    bd->setReductionRatio(SCALING);
    bd->setWidthOfBand(9);

    Ptr<cv::line_descriptor::LSDDetector> lsd = cv::line_descriptor::LSDDetector::createLSDDetector();

    /* lines */
    std::vector<cv::line_descriptor::KeyLine> keylines1, keylines2;

    /* extract lines */
    lsd->detect( image_1, keylines1, OCTAVES, SCALING, mask1 );
    lsd->detect( image_2, keylines2, OCTAVES, SCALING, mask2 );

    //Filter detected lines:
    double minLenght = sqrt(image_1.cols*image_1.cols + image_1.rows*image_1.rows)*MIN_LENGTH_FACTOR;
    if(LOG_DEBUG) std::cout << "-- Min line segment length: " << minLenght << std::endl;
    int filtered1 = filterLineExtractions(minLenght, keylines1);
    int filtered2 = filterLineExtractions(minLenght, keylines2);

    if(LOG_DEBUG) {
        std::cout << "-- First image: " << keylines1.size() << " filtered: " << filtered1 << std::endl;
        std::cout << "-- Second image: " << keylines2.size() << " filtered: " << filtered2 << std::endl;
    }

    std::vector<DMatch> matches;

    /* compute descriptors */
    cv::Mat descr1, descr2;
    bd->compute( image_1, keylines1, descr1 );
    bd->compute( image_2, keylines2, descr2 );

    /* create a BinaryDescriptorMatcher object */
    Ptr<cv::line_descriptor::BinaryDescriptorMatcher> bdm = cv::line_descriptor::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

    /* require match */
    bdm->match( descr1, descr2, matches );

    /************************************************************************/

    cv::line_descriptor::KeyLine l1, l2;
    int filteredMatches = 0;
    int id = 0;
    //Reduce max hemming distance if number of matches are high
    int maxHemmingDist = MAX_HEMMING_DIST;//= MIN_HEMMING_DIST + std::min((int)((MAX_HEMMING_DIST - MIN_HEMMING_DIST)*1400.0/matches.size()), (MAX_HEMMING_DIST - MIN_HEMMING_DIST));
    if(LOG_DEBUG) std::cout << "-- Min match hemming dist: " << maxHemmingDist << std::endl;

    for (std::vector<DMatch>::const_iterator it= matches.begin(); it!=matches.end(); ++it) {

        l1 = keylines1[it->queryIdx];
        l2 = keylines2[it->trainIdx];

        id++;
        lineCorrespStruct lc = getlineCorrespStruct(l1,l2, id);
        lc.isGoodMatch = false;

        if (it->distance > maxHemmingDist || filterLineMatch(l1,l2)) {  //Bad match
            filteredMatches++;
        } else {    //Good match, add to correspondence list
            lc.isGoodMatch = true;
            //goodMatchedLines.push_back(lc);
        }

        allMatchedLines.push_back(lc);
    }

    if(LOG_DEBUG) {
        std::cout << "-- Number of matches : " << allMatchedLines.size() << " good matches: " << allMatchedLines.size() - filteredMatches << std::endl;
    }

    if(LOG_DEBUG) std::cout << std::endl;
    return allMatchedLines.size();
}

bool filterLineMatch(cv::line_descriptor::KeyLine l1, cv::line_descriptor::KeyLine l2) {
    if(smallestRelAngle(l1.angle, l2.angle) > MAX_LINE_ANGLE) return true;
    return false;
}

int filterLineExtractions(double minLenght, std::vector<cv::line_descriptor::KeyLine> &keylines) {
    int filtered = 0;
    std::vector<cv::line_descriptor::KeyLine>::iterator it= keylines.begin();
    while (it!=keylines.end()) {
        if ((it->octave > 0 && it->octave*SCALING*it->lineLength < minLenght) || (it->octave == 0 && it->lineLength < minLenght)) {
            keylines.erase(it);
            filtered++;
        } else it++;
    }
    return filtered;
}

int matchTangents(Mat image_1, Mat image_2, std::vector<Point2d> &x1, std::vector<Point2d> &x2) {
////    cvtColor(img_flood,img_flood,CV_RGB2GRAY);
////    threshold(img_flood, img_binary, 254, 255, cv::THRESH_BINARY);
////    erode(img_binary, img_binary, kernel_opening,center_opening);
////    dilate(img_binary, img_binary, kernel_opening,center_opening);
////    dilate(img_binary, img_binary, kernel_closing,center_closing);
////    erode(img_binary, img_binary, kernel_closing,center_closing);

////    if(VISUAL_DEBUG) {
////        showImage(image_name+" binary", img_binary);
////    }

//    Mat contours1, contours2;

//    findContours(image_1, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

//    Mat img_contours = Mat::zeros(image.rows, image.cols, image.type());

//    for( int i = 0; i< segmentList.size(); i++ )
//    {
//        drawContours(img_contours, contours, segmentList[i].contours_idx, Scalar(255,255,255), 2);
//    }

//    showImage(image_name+" plane contours", img_contours);
//    //    if(VISUAL_DEBUG) {
//    //        showImage(image_name+" binary", img_binary);
//    //    }

//    for (int i = 0; i < contours.size(); i++) {
//        for (int j = 0; j < segmentList_temp.size(); j++) {
//            if(pointPolygonTest(contours[i], Point2d(segmentList_temp[j].startpoint.x, segmentList_temp[j].startpoint.y),false) >= 0) {
//                segmentStruct segment = segmentList_temp[j];
//                segment.contours_idx = i;
//                for (int k = 0; k < segmentList.size(); k++) {      //Keep only one segment per connected component
//                    if (segmentList[k].contours_idx == i) segment.contours_idx = -1;
//                }

//                if(segment.contours_idx >= 0) {
//                    if (segmentList.size() == 0) segment.id = 1;
//                    else segment.id = segmentList.back().id + 1;
//                    segmentList.push_back(segment);
//                }
//            }
//        }
//    }
}
