#include "FEstimatorPoints.h"

FEstimatorPoints::FEstimatorPoints(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name) {
    image_1 = img1.clone();
    image_2 = img2.clone();
    image_1_color = img1_c.clone();
    image_2_color = img2_c.clone();
    this->name = name;
    if(LOG_DEBUG) std::cout << "Estimating: " << name << std::endl;
    successful = false;

    computationType = F_FROM_POINTS;

    normT1 = Mat::eye(3,3,CV_64FC1);
    normT2 = Mat::eye(3,3,CV_64FC1);
}

int FEstimatorPoints::extractMatches() {
    std::vector<pointCorrespStruct> allPointCorrespondencies;
    extractPointMatches(image_1, image_2, allPointCorrespondencies);

    for(std::vector<pointCorrespStruct>::iterator iter = allPointCorrespondencies.begin(); iter != allPointCorrespondencies.end(); ++iter) {
        if(iter->isGoodMatch) {
            x1.push_back(iter->x1);
            x2.push_back(iter->x2);
        }
    }

    if(LOG_DEBUG) std::cout << "-- Number of good matches: " << x1.size() << std::endl;

    visualizePointMatches(image_1_color, image_2_color, x1, x2, 3, true, name+": good point matches");
}

bool FEstimatorPoints::compute() {
    //std::vector<int> mask;
    Mat mask;
    extractMatches();
    F = findFundamentalMat(x1, x2, CV_FM_RANSAC, RANSAC_THREDHOLD, RANSAC_CONFIDENCE, mask);
    //F.convertTo(F, CV_64FC1);
    for(int i = 0; i < x1.size(); i++) {
        if(mask.at<int>(i,0)) {
            x1_used.push_back(x1.at(i));
            x2_used.push_back(x2.at(i));
            featuresImg1.push_back(matVector(x1.at(i)));
            featuresImg2.push_back(matVector(x2.at(i)));
        }
    }
    if(LOG_DEBUG) std::cout << "-- Used matches (RANSAC): " << x1_used.size() << std::endl;

    if(featuresImg1.size() >= 7) successful = true;

    error = meanSquaredSymmeticTransferError(F, x1_used, x2_used);

    if(VISUAL_DEBUG) visualizePointMatches(image_1_color, image_2_color, x1_used, x2_used, 3, true, name+": Used point matches");

    if(LOG_DEBUG) std::cout << std::endl << std::endl;

    return successful;
}

std::vector<Point2d> FEstimatorPoints::getUsedX1() {
    return x1_used;
}

std::vector<Point2d> FEstimatorPoints::getUsedX2() {
    return x2_used;
}
