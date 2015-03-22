#include "FEstimatorHPlanes.h"

FEstimatorHPlanes::FEstimatorHPlanes(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name) {
    image_1 = img1.clone();
    image_2 = img2.clone();
    image_1_color = img1_c.clone();
    image_2_color = img2_c.clone();
    this->name = name;
    std::cout << "Estimating: " << name << std::endl;
    successful = false;
    computationType = F_FROM_PLANES_VIA_H;
}

bool FEstimatorHPlanes::compute() {
    return false;
}

int FEstimatorHPlanes::extractMatches() {
    std::vector<segmentStruct> segmentsList_1, segmentsList_2;
    Mat segments_1, segments_2;

    std::cout << "EXTRACTING PLANES:" << std::endl;

    findSegments(image_1, image_1_color, "Image 1", segmentsList_1, segments_1);
    findSegments(image_2, image_2_color, "Image 2", segmentsList_2, segments_2);

    std::cout << "-- First image: " << segmentsList_1.size() << std::endl;
    std::cout << "-- Second image: " << segmentsList_2.size() << std::endl;

    //TODO: MATCH!

    //std::cout << "-- Number of matches: " << good_matches.size() << std::endl;

    return 0;

}

void FEstimatorHPlanes::findSegments(Mat image, Mat image_color, std::string image_name, std::vector<segmentStruct> &segmentList, Mat &segments) {
    std::vector<cv::KeyPoint> keypoints;
    std::vector<segmentStruct> segmentList_temp;

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

    SimpleBlobDetector blobs = SimpleBlobDetector();

    blobs.detect(image_color, keypoints);

    if(VISUAL_DEBUG) {

        Mat img_blobs = image_color.clone();

        for( int i = 0; i< keypoints.size(); i++ )
        {
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            circle(img_blobs, keypoints[i].pt, 15, color, 8);
        }

        showImage(image_name+" blobs", img_blobs);
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
        showImage(image_name+" flooded", img_flood);
    }

    cvtColor(img_flood,img_flood,CV_RGB2GRAY);
    threshold(img_flood, img_binary, 254, 255, cv::THRESH_BINARY);
    erode(img_binary, img_binary, kernel_opening,center_opening);
    dilate(img_binary, img_binary, kernel_opening,center_opening);
    dilate(img_binary, img_binary, kernel_closing,center_closing);
    erode(img_binary, img_binary, kernel_closing,center_closing);

    if(VISUAL_DEBUG) {
        showImage(image_name+" binary", img_binary);
    }

    findContours(img_binary, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

    for (int i = 0; i < contours.size(); i++) {
        for (int j = 0; j < segmentList_temp.size(); j++) {
            if(pointPolygonTest(contours[i], Point2d(segmentList_temp[j].startpoint.x, segmentList_temp[j].startpoint.y),false) >= 0) {
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

        showImage(image_name+" plane contours", img_contours);

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

        showImage(image_name+" plane segments", img_segments);
    }

    segments = Mat::zeros(image.rows, image.cols, image.type());

    for (int i = 0; i< segmentList.size(); i++ ) {
        drawContours(segments, contours, segmentList[i].contours_idx, Scalar(255,255,255), 2);
        floodFill(segments, segmentList[i].startpoint, segmentList[i].id, 0, Scalar(0,0,0), Scalar(0,0,0), CV_FLOODFILL_FIXED_RANGE);
    }

    std::cout << std::endl;
}
