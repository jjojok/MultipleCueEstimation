#include "MCE.h"

using namespace cv;

MCE::MCE()
{
    compareWithGroundTruth = false;
    computations = 0;
}

void MCE::run() {
    Mat Fgt;       //Ground truth Fundamental matrix
    std::vector<matrixStruct> fundamentalMatrices;
    if (loadData()) {
        if(computations & F_FROM_POINTS) {
            extractPoints();
            matrixStruct ms;
            ms.source = "points";
            ms.F = calcFfromPoints();
            fundamentalMatrices.push_back(ms);
        }
        if(computations & F_FROM_LINES) {
            extractLines();
            matrixStruct ms;
            ms.source = "lines";
            ms.F = calcFfromLines();
            Mat transformed;
            warpPerspective(image_2, transformed, ms.F, Size(image_2.cols,image_2.rows));
            namedWindow("H21", CV_WINDOW_NORMAL);
            imshow("H21", transformed);
            //fundamentalMatrices.push_back(ms);
        }
        if(computations & F_FROM_PLANES) {
            extractPlanes();
        }

        if (compareWithGroundTruth) {   //Compare to ground truth
            Fgt = getGroundTruth();
            std::cout << "Fgt = " << std::endl << Fgt << std::endl;
        }

        for (std::vector<matrixStruct>::iterator it = fundamentalMatrices.begin() ; it != fundamentalMatrices.end(); ++it) {
            it->error = averageSquaredError(Fgt,it->F)[0];
            std::cout << "F from " << it->source << " = " << std::endl << it->F << std::endl << "Error: " << it->error << std::endl;
        }

        //rectify(x1, x2, Fgt, image_1, 1, "Image 1 rect Fgt");
        //rectify(x1, x2, Fpt, image_1, 1, "Image 1 rect Fpt");

        //drawEpipolarLines(x1, x2, Fgt, image_1.clone(), image_2.clone());

        Mat linEq = (Mat_<double>(3,4) << 2, -2, -1, 1, -3, 6, -2, -1, -4, 5, -3, 2);
        //Mat linEq = (Mat_<double>(3,3) << 2, -1, -1, -3, 5, -2, -4, 7, -3);
        Mat testvect = Mat::zeros(3,1, CV_32FC1);
        std::cout << "Solutions = " << calcNumberOfSolutions(linEq) << std::endl;
        Mat a = linEq.colRange(0, linEq.cols-1);
        Mat b = -linEq.col(linEq.cols-1);
        solve(a, b, testvect);

        //SVD::solveZ(linEq, testvect);
        //testvect = testvect.reshape(1,3);

        std::cout << "linEq test " << " = " << std::endl << testvect << std::endl;


        waitKey(0);
        //Mat h = findHomography(x1, x2, noArray(), CV_RANSAC, 3);
        //compare with "real" Fundamental Matrix or calc lush point cloud?:
        //void triangulatePoints(InputArray projMatr1, InputArray projMatr2, InputArray projPoints1, InputArray projPoints2, OutputArray points4D)¶
    }

}

int MCE::loadData() {
    if (arguments != 4 && !compareWithGroundTruth) {
        std::cout << "Usage: MultipleCueEstimation <path to first image> <path to second image> <optional: path to first camera matrix> <optional: path to second camera matrix>" << std::endl;
        return 0;
    }

    image_1 = imread(path_img1, CV_LOAD_IMAGE_GRAYSCALE);
    image_2 = imread(path_img2, CV_LOAD_IMAGE_GRAYSCALE);

    image_1_color = imread(path_img1, CV_LOAD_IMAGE_COLOR);
    image_2_color = imread(path_img2, CV_LOAD_IMAGE_COLOR);

    if ( !image_1.data || !image_2.data || !image_1_color.data || !image_2_color.data )
    {
        printf("No image data \n");
        return 0;
    }

    if(VISUAL_DEBUG) {
        namedWindow("Image 1 original", CV_WINDOW_NORMAL);
        imshow("Image 1 original", image_1);

        namedWindow("Image 2 original", CV_WINDOW_NORMAL);
        imshow("Image 2 original", image_2);

    }

    return 1;
}

void MCE::extractLines() {
    LineMatcher* lm = new LineMatcher();

    Mat image_1_down;
    Mat image_2_down;

    std::string path_img1_down = path_img1 + "_down";
    std::string path_img2_down = path_img2 + "_down";

    //TODO: works only with small images (e.g. 800x600), replace with opencv if ver 3 is out
    resize(image_1_color, image_1_down, Size(0,0), 0.25, 0.25, INTER_NEAREST);
    resize(image_2_color, image_2_down, Size(0,0), 0.25, 0.25, INTER_NEAREST);

    imwrite(path_img1_down, image_1_down);
    imwrite(path_img2_down, image_2_down);


    std::vector<Point2f>* lineCorr = new std::vector<Point2f>();

    std::cout << "EXTRACTING LINES:" << std::endl;
    int corresp = lm->match(path_img1_down.c_str(), path_img2_down.c_str(), lineCorr);       //TODO image scaled to 25% -> multiply line coords by 4; TODO: Check if image dimensions % 4 = 0, cut img if not
    std::cout << "-- Number of matches: " << corresp << std::endl;

    int img_width = image_1.cols;
    int img_height = image_1.rows;

    for(int i = 0; i < lineCorr->size(); i+=4) {
        lineCorrespStruct lc;
        lc.line1Start = normalize(lineCorr->at(i), img_width/4, img_height/4);
        lc.line1End = normalize(lineCorr->at(i+1), img_width/4, img_height/4);
        lc.line2Start = normalize(lineCorr->at(i+2), img_width/4, img_height/4);
        lc.line2End = normalize(lineCorr->at(i+3), img_width/4, img_height/4);

        lineCorrespondencies.push_back(lc);
    }

    if(VISUAL_DEBUG) {

        namedWindow("Image 1 lines", CV_WINDOW_AUTOSIZE);
        imshow("Image 1 lines", imread("LinesInImage1.png", CV_LOAD_IMAGE_COLOR));

        namedWindow("Image 2 lines", CV_WINDOW_AUTOSIZE);
        imshow("Image 2 lines", imread("LinesInImage2.png", CV_LOAD_IMAGE_COLOR));

        namedWindow("Line matches", CV_WINDOW_AUTOSIZE);
        imshow("Line matches", imread("LBDSG.png", CV_LOAD_IMAGE_COLOR));

    }

    std::cout << std::endl;
}

void MCE::extractPlanes() {
    Vector<segmentStruct> segmentsList_1, segmentsList_2;
    Mat segments_1, segments_2;

    std::cout << "EXTRACTING PLANES:" << std::endl;

    findSegments(image_1, image_1_color, "Image 1", segmentsList_1, segments_1);
    findSegments(image_2, image_2_color, "Image 2", segmentsList_2, segments_2);

    std::cout << "-- First image: " << segmentsList_1.size() << std::endl;
    std::cout << "-- Second image: " << segmentsList_2.size() << std::endl;


    //std::cout << "-- Number of matches: " << good_matches.size() << std::endl;

    waitKey(0);

}

void MCE::findSegments(Mat image, Mat image_color, std::string image_name, Vector<segmentStruct> &segmentList, Mat &segments) {
    std::vector<cv::KeyPoint> keypoints;
    Vector<segmentStruct> segmentList_temp;

    std::vector< std::vector<Point> > contours;
    RNG rng;

    cv::SimpleBlobDetector::Params params;
    params.filterByArea = true;
    params.filterByCircularity= false;
    params.filterByColor = false;
    params.filterByConvexity = false;
    params.filterByInertia = false;
    params.minArea = image_1.cols*image_1.rows*0.0005;
    params.maxArea = image_1.cols*image_1.rows*0.2;
    params.minConvexity = 0.5;
    params.maxConvexity = 1;
    params.minDistBetweenBlobs = 20;

    SimpleBlobDetector blobs = SimpleBlobDetector(params);

    blobs.detect(image_color, keypoints);

    if(VISUAL_DEBUG) {

        Mat img_blobs = image_color.clone();

        for( int i = 0; i< keypoints.size(); i++ )
        {
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            circle(img_blobs, keypoints[i].pt, 15, color, 8);
        }

        namedWindow(image_name+" blobs", CV_WINDOW_NORMAL);
        imshow(image_name+" blobs", img_blobs);
    }
    Mat img_binary;
    Mat img_flood = image_color.clone();
    Scalar upper_range = Scalar(15,15,15);
    Scalar lower_range = Scalar(15,15,15);
    Mat kernel_opening = Mat::ones(7,7,CV_8UC1);
    Mat kernel_closing = Mat::ones(5,5,CV_8UC1);
    Point center_opening = Point(kernel_opening.cols/2,kernel_opening.rows/2+1);
    Point center_closing = Point(kernel_closing.cols/2,kernel_closing.rows/2+1);
    int minArea = image_1.cols*image_1.rows*0.0005;

    for(int i = 0; i < keypoints.size(); i++) {
        segmentStruct segment;
        segment.area = floodFill(img_flood, keypoints[i].pt, Scalar(255,255,255), (Rect*)0, lower_range, upper_range, CV_FLOODFILL_FIXED_RANGE);
        if (segment.area > minArea) {
            segment.startpoint = keypoints[i].pt;
            segment.contours_idx = -1;
            segmentList_temp.push_back(segment);
        }
    }

    if(VISUAL_DEBUG) {
        namedWindow(image_name+" flooded", CV_WINDOW_NORMAL);
        imshow(image_name+" flooded", img_flood);
    }

    cvtColor(img_flood,img_flood,CV_RGB2GRAY);
    threshold(img_flood, img_binary, 254, 255, cv::THRESH_BINARY);
    erode(img_binary, img_binary, kernel_opening,center_opening);
    dilate(img_binary, img_binary, kernel_opening,center_opening);
    dilate(img_binary, img_binary, kernel_closing,center_closing);
    erode(img_binary, img_binary, kernel_closing,center_closing);

    if(VISUAL_DEBUG) {
        namedWindow(image_name+" binary", CV_WINDOW_NORMAL);
        imshow(image_name+" binary", img_binary);
    }

    findContours(img_binary, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

    for (int i = 0; i < contours.size(); i++) {
        for (int j = 0; j < segmentList_temp.size(); j++) {
            if(pointPolygonTest(contours[i], Point2f(segmentList_temp[j].startpoint.x, segmentList_temp[j].startpoint.y),false) >= 0) {
                segmentStruct segment = segmentList_temp[j];
                segment.contours_idx = i;
                for (int k = 0; k < segmentList.size(); k++) {      //Keep only one segment per connected component
                    if (segmentList[k].contours_idx == i) segment.contours_idx = -1;
                }

                if(segment.contours_idx >= 0) {
                    if (segmentList.size() == 0) segment.id = 1;
                    else segment.id = segmentList.back().id + 1;
                    segmentList.push_back(segment);
                }
            }
        }
    }

    if(VISUAL_DEBUG) {

        Mat img_contours = Mat::zeros(image.rows, image.cols, image.type());

        for( int i = 0; i< segmentList.size(); i++ )
        {
            drawContours(img_contours, contours, segmentList[i].contours_idx, Scalar(255,255,255), 2);
        }

        namedWindow(image_name+" plane contours", CV_WINDOW_NORMAL);
        imshow(image_name+" plane contours", img_contours);

        CvFont font;
        cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0);

        Mat img_segments = Mat(image.rows, image.cols, image_color.type());

        cvtColor(img_contours, img_segments, CV_GRAY2BGR);

        for (int i = 0; i< segmentList.size(); i++ ) {
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            floodFill(img_segments, segmentList[i].startpoint, color, 0, Scalar(0,0,0), Scalar(0,0,0), CV_FLOODFILL_FIXED_RANGE);
        }

        char buff[10];
        for (int i = 0; i< segmentList.size(); i++ ) {
            circle(img_segments, segmentList[i].startpoint, 20, Scalar(255,255,255), 10);
            std::sprintf(buff," #%i", segmentList[i].id);
            putText(img_segments, buff, segmentList[i].startpoint, CV_FONT_HERSHEY_SIMPLEX, 3, Scalar(255, 255, 255), 10);
        }

        namedWindow(image_name+" plane segments", CV_WINDOW_NORMAL);
        imshow(image_name+" plane segments", img_segments);
    }

    segments = Mat::zeros(image.rows, image.cols, image.type());

    for (int i = 0; i< segmentList.size(); i++ ) {
        drawContours(segments, contours, segmentList[i].contours_idx, Scalar(255,255,255), 2);
        floodFill(segments, segmentList[i].startpoint, segmentList[i].id, 0, Scalar(0,0,0), Scalar(0,0,0), CV_FLOODFILL_FIXED_RANGE);
    }

    std::cout << std::endl;
}

void MCE::extractPoints() {

    // ++++ Source: http://docs.opencv.org/doc/tutorials/features2d/feature_flann_matcher/feature_flann_matcher.html

    std::cout << "EXTRACTING POINTS:" << std::endl;

    std::vector<cv::KeyPoint> keypoints_1;
    std::vector<cv::KeyPoint> keypoints_2;

    SurfFeatureDetector detector(SIFT_FEATURE_COUNT);
    detector.detect(image_1, keypoints_1);
    detector.detect(image_2, keypoints_2);

    std::cout << "-- First image: " << keypoints_1.size() << std::endl;
    std::cout << "-- Second image: " << keypoints_2.size() << std::endl;

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
    { float dist = matches[i].distance;
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
    drawMatches( image_1, keypoints_1, image_2, keypoints_2,
               good_matches, img_matches );

    //-- Show detected matches
    if(VISUAL_DEBUG) {
        namedWindow("SURF results", CV_WINDOW_NORMAL);
        imshow( "SURF results", img_matches );
    }

    // ++++

    for( int i = 0; i < good_matches.size(); i++ )
    {
        x1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
        x2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
    }

    std::cout << std::endl;

}

Mat MCE::calcFfromPoints() {
    std::cout << "Calc Fpt..." << std::endl;
    return findFundamentalMat(x1, x2, FM_RANSAC, 3.0, 0.99, noArray());
}

Mat MCE::calcFfromLines() {     // From Paper: "Robust line matching in image pairs of scenes with dominant planes", Sagúes C., Guerrero J.J.
    int numOfPairs = lineCorrespondencies.size();
    int numOfPairSubsets = numOfPairs/5;
    std::vector<lineSubsetStruct> subsets;

    //Compute H_21 from 4 line correspondencies
    for(int i = 0; i < numOfPairSubsets; i++) {
        std::vector<int> subsetsIdx;
        Mat linEq = Mat::ones(8,9,CV_32FC1);
        lineSubsetStruct subset;
        for(int j = 0; j < 4; j++) {
            int subsetIdx = 0;
            do {        //Generate 4 uniqe random indices for line pair selection
                subsetIdx = std::rand() % numOfPairs;
            } while(std::find(subsetsIdx.begin(), subsetsIdx.end(), subsetIdx) != subsetsIdx.end());

            subsetsIdx.push_back(subsetIdx);
            lineCorrespStruct lc = lineCorrespondencies.at(subsetIdx);
            subset.lineCorrespondencies.push_back(lc);
            fillHLinEq(&linEq, lc, j);
        }
        Mat A = linEq.colRange(0, linEq.cols-1);    //TODO: ist das korrekt? :)
        Mat x = -linEq.col(linEq.cols-1);
        solve(A, x, subset.Hs, DECOMP_SVD);
        //SVD::solveZ(linEq, subset.Hs);
        subset.Hs.resize(9);
        subset.Hs = subset.Hs.reshape(1,3);
        subset.Hs.at<float>(2,2) = 1.0;
        subsets.push_back(subset);
    }


    return calcLMedS(subsets);
}

Mat MCE::calcFfromPlanes() {    // From: 1. two Homographies (one in each image), 2. Planes as additinal point information (point-plane dualism)
    //findHomography with different planes
}

Mat MCE::calcFfromConics() {    // Maybe: something with vanishing points v1*w*v2=0 (Hartley, Zissarmen p. 235ff)

}

Mat MCE::calcFfromCurves() {    // First derivative of corresponding curves are gradients to the epipolar lines

}

Mat MCE::refineF() {    //Reduce error of F AFTER computing it seperatly form different sources

}


Mat MCE::MatFromFile(std::string file, int rows) {

    Mat matrix;
    std::ifstream inputStream;
    float x;
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

Mat MCE::crossProductMatrix(Mat input) {    //3 Vector to cross procut matrix
    Mat crossMat = Mat::zeros(3,3, input.type());
    crossMat.at<float>(0,1) = -input.at<float>(2);
    crossMat.at<float>(0,2) = input.at<float>(1);
    crossMat.at<float>(1,0) = input.at<float>(2);
    crossMat.at<float>(1,2) = -input.at<float>(0);
    crossMat.at<float>(2,0) = -input.at<float>(1);
    crossMat.at<float>(2,1) = input.at<float>(0);
    return crossMat;
}

void MCE::rectify(std::vector<Point2f> p1, std::vector<Point2f> p2, Mat F, Mat image, int imgNum, std::string windowName) {
    if(VISUAL_DEBUG) {
        Mat H1, H2, H, rectified;
        if(stereoRectifyUncalibrated(p1, p2, F, Size(image.cols,image.rows), H1, H2, 0 )) {
            if (LOG_DEBUG) {
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
}

Mat MCE::getGroundTruth() {
    Mat P1w = MatFromFile(path_P1, 3); //P1 in world coords
    Mat P2w = MatFromFile(path_P2, 3); //P2 in world coords
    Mat T1w, T2w, R1w, R2w;   //World rotation, translation
    Mat K1, K2, K; //calibration matrices
    Mat Rrel, Trel; //Relative rotation, translation

    if (LOG_DEBUG) {
        std::cout << "P1w = " << std::endl << P1w << std::endl;
        //std::cout << "P2w = " << std::endl << P2w << std::endl;
    }

    decomposeProjectionMatrix(P1w, K1, R1w, T1w, noArray(), noArray(), noArray(), noArray() );
    //decomposeProjectionMatrix(P2w, K2, R2w, T2w, noArray(), noArray(), noArray(), noArray() );

    //K = (K1 + K2)/2;    //Images with same K

    T1w = T1w/T1w.at<float>(3);      //convert to homogenius coords
    //T1w.resize(3);

    //T2w = T2w/T2w.at<float>(3);      //convert to homogenius coords
    //T2w.resize(3);

//    R2w = R2w.t();      //switch rotation: world to 2. cam frame (Rc2w) to 2. cam to world frame (Rwc2)
//    R1w = R1w.t();      //switch rotation: world to 1. cam frame (Rc1w) to 1. cam to world frame (Rwc1)



    if (LOG_DEBUG) {
        std::cout << "T1w = " << std::endl << T1w << std::endl;
        //std::cout << "T2w = " << std::endl << T2w << std::endl;

        std::cout << "R1w = " << std::endl << R1w << std::endl;
        //std::cout << "R2w = " << std::endl << R2w << std::endl;
    }

    //Rrel = R1w*R2w.t(); //Relative rotation between cam1 and cam2; Rc1c2 = Rwc1^T * Rwc2

    if (LOG_DEBUG) {
        //std::cout << "K = " << std::endl << K << std::endl;
        //std::cout << "K2 = " << std::endl << K2 << std::endl;
    }

    //Trel = T2w - T1w;    //Realtive translation between cam1 and cam2

    if (LOG_DEBUG) {
        //std::cout << "Rrel = " << std::endl << Rrel << std::endl;
        std::cout << "Trel = " << std::endl << Trel << std::endl;
    }

    Mat F = crossProductMatrix(P2w*T1w)*P2w*P1w.inv(DECOMP_SVD);

//    Mat C = (Mat_<float>(4,1) << 0, 0, 0, 1.0);
//    Mat e = P2w*C;
//    std::cout << "e = " << std::endl << e << std::endl;
//    return crossMatrix(e)*P2w*P1w.inv(DECOMP_SVD);

//    F = K.t().inv()*crossProductMatrix(Trel)*Rrel*K.inv();
//    //return K.t().inv()*crossProductMatrix(Trel)*Rrel*K.inv();
//    //return K2.t().inv()*crossProductMatrix(Trel)*Rrel*K1.inv(); //(See Hartley, Zisserman: p. 244)
//    F = K2.t().inv()*Rrel*K1.t()*crossProductMatrix(K1*Rrel.t()*Trel);
    return F / F.at<float>(2,2);       //Set global arbitrary scale factor 1 -> easier to compare
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
//    std::vector<cv::Point2f> points1In, points2In;
//    std::vector<cv::Point2f>::const_iterator itPts= p1.begin();
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
    return squaredError(A,B)/((double)(A.cols*A.rows));
}

Scalar MCE::squaredError(Mat A, Mat B) {
    return cv::sum((A-B).mul(A-B));
}

Point2f MCE::normalize(Point2f p, int img_width, int img_height) {
    p.x = (p.x - img_width/2) / img_width;
    p.y = (p.y - img_height/2) / img_height;
    return p;
}

void MCE::fillHLinEq(Mat* linEq, lineCorrespStruct lc, int numPair) {
    float A = lc.line2Start.y - lc.line2End.y;
    float B = lc.line2End.x - lc.line2Start.x;
    float C = lc.line2Start.x*lc.line2End.y - lc.line2End.x*lc.line2Start.y;
    int row = 2*numPair;
    fillHLinEqBase(linEq, lc.line1Start, A, B, C, row);
    fillHLinEqBase(linEq, lc.line1End, A, B, C, row + 1);
}

void MCE::fillHLinEqBase(Mat* linEq, Point2f point, float A, float B, float C, int row) {
    linEq->at<float>(row, 0) = A*point.x;
    linEq->at<float>(row, 1) = A*point.y;
    linEq->at<float>(row, 2) = A;
    linEq->at<float>(row, 3) = B*point.x;
    linEq->at<float>(row, 4) = B*point.y;
    linEq->at<float>(row, 5) = B;
    linEq->at<float>(row, 6) = C*point.x;
    linEq->at<float>(row, 7) = C*point.y;
    linEq->at<float>(row, 8) = C;
}

Mat MCE::calcLMedS(std::vector<lineSubsetStruct> subsets) {
    std::vector<lineSubsetStruct>::iterator it = subsets.begin();
    float lMedS = calcMedS(it->Hs);
    Mat lMedSH = it->Hs;
    it++;
    do {
        lineSubsetStruct ls = *it;
        float meds = calcMedS(ls.Hs);
        if(meds < lMedS) {
            lMedS = meds;
            lMedSH = ls.Hs;
        }
        it++;
    } while(it != subsets.end());
    return lMedSH;
}


float MCE::calcMedS(Mat Hs) {
    std::vector<float> dist;
    for(std::vector<lineCorrespStruct>::iterator it = lineCorrespondencies.begin() ; it != lineCorrespondencies.end(); ++it) {
        lineCorrespStruct lc = *it;
        Mat p1s = Mat(lc.line1Start);
        vconcat(p1s, Mat::ones(1,1,p1s.type()),p1s);
        Mat p1e = Mat(lc.line1End);
        vconcat(p1e, Mat::ones(1,1,p1e.type()),p1e);

        Mat p2s = Mat(lc.line2Start);
        vconcat(p2s, Mat::ones(1,1,p2s.type()),p2s);
        Mat p2e = Mat(lc.line2End);
        vconcat(p2e, Mat::ones(1,1,p2e.type()),p2e);

        Mat n1 = p1s.cross(p1e);
        Mat n2real = p2s.cross(p2e);
        Mat n2comp = Hs.inv(DECOMP_SVD).t()*n1;
        dist.push_back(sqrt(squaredError(n2real,n2comp)[0]));
    }

    std::sort(dist.begin(), dist.end());

    return dist.at(dist.size()/2);
}

int MCE::calcMatRank(Mat M) {
    Mat U,V,W;
    int rank = 0, diag;
    SVD::compute(M,U,V,W);
    if (W.cols < W.rows) diag = W.cols;
    else diag = W.rows;
    for(int i = 0; i < diag; i++) {
        if(fabs(W.at<float>(i,i)) > 10^(-10)) {
            rank++;
        }
    }
    return rank;
}

//returns: 0 = no solution, 1 = one solution, -1 = inf solutions
int MCE::calcNumberOfSolutions(Mat linEq) {
    Mat coefficients = linEq.colRange(0, linEq.cols-1);
    int coeffRank = calcMatRank(coefficients);
    int augmentedRank = calcMatRank(linEq);
    if (augmentedRank > coeffRank) return 0;
    if (augmentedRank == coeffRank) return 1;
    return -1;
}
