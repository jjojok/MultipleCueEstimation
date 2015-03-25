#include "FEstimatorHPoints.h"

FEstimatorHPoints::FEstimatorHPoints(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name) {
    image_1 = img1.clone();
    image_2 = img2.clone();
    image_1_color = img1_c.clone();
    image_2_color = img2_c.clone();
    this->name = name;
    if(LOG_DEBUG) std::cout << "Estimating: " << name << std::endl;
    successful = false;

    computationType = F_FROM_POINTS_VIA_H;

    normT1 = Mat::eye(3,3,CV_64FC1);
    normT2 = Mat::eye(3,3,CV_64FC1);
}

int FEstimatorHPoints::extractMatches() {
    extractPointMatches(image_1, image_2, x1, x2);
}

bool FEstimatorHPoints::compute() {

    extractMatches();

    if(LOG_DEBUG) std::cout << "-- First Estimation..."<< std::endl;

    std::vector<int> mask;
    Mat H1 = findHomography(x1, x2, LMEDS, RANSAC_THREDHOLD, mask, 2000, RANSAC_CONFIDENCE);

    if(!H1.data) {
        if(LOG_DEBUG) std::cout << "-- Estimation failed!  (Unable to compute H1)" << std::endl;
        return false;
    }

    if(VISUAL_DEBUG) {
        visualizeMatches(image_1_color, image_2_color, x1_used, x2_used, 3, true, "Point Homography 1 matches");
        visualizeHomography(H1, image_1_color, image_2_color, "Point homography 1");
    }

    for(int i = mask.size()-1; i >= 0 ; i--) {
        if(mask.at(i)) {
            x1_used.push_back(x1.at(i));
            x2_used.push_back(x2.at(i));
            featuresImg1.push_back(matVector(x1.at(i)));
            featuresImg2.push_back(matVector(x2.at(i)));
            x1.erase(x1.begin()+i);
            x2.erase(x2.begin()+i);
        }
    }

    if(LOG_DEBUG) std::cout << "-- Used matches: " << x1_used.size() << std::endl;

    int estCnt = 0;
    bool homographies_equal = true;
    Mat H, H2;
    Mat e2;

    std::vector<Point2d> goodX1, goodX2;

    while(homographies_equal) {

        if(LOG_DEBUG) std::cout << "-- First Estimation..."<< std::endl;

        estCnt++;

        goodX1.clear();
        goodX2.clear();
        mask.clear();
        for(int i = 0; i < x1.size(); i++) {
            goodX1.push_back(x1.at(i));
            goodX2.push_back(x2.at(i));
        }

        if(LOG_DEBUG) std::cout << "-- Refined number of matches: " << goodX1.size() << std::endl;

        H2 = findHomography(goodX1, goodX2, CV_RANSAC, RANSAC_THREDHOLD, mask, 2000, RANSAC_CONFIDENCE);

        if(!H2.data) {
            if(LOG_DEBUG) std::cout << "-- Estimation failed! (Unable to compute H2)" << std::endl;
            return false;
        }

        H = H1*H2.inv(DECOMP_SVD); // H = (H1*H2⁻1)

        homographies_equal = (computeUniqeEigenvector(H, e2) && isUnity(H));
        if(homographies_equal) {
            if(LOG_DEBUG) std::cout << "-- Homographies equal, repeating estimation..." << std::endl << "-- H = " << std::endl << H << std::endl;
            for(int i = mask.size()-1; i >= 0 ; i--) {
                if(mask.at(i)) {
                    x1.erase(x1.begin()+i);
                    x2.erase(x2.begin()+i);
                }
            }
        }

        if(MAX_H2_ESTIMATIONS < estCnt) {   //Not able to find a second homographie
            if(LOG_DEBUG) std::cout << "-- Estimation failed!" << std::endl;
            return false;
        }
    }

    std::vector<Point2d> x1_temp;
    std::vector<Point2d> x2_temp;

    for(int i = 0; i < mask.size(); i++) {
        if(mask.at(i)) {
            x1_used.push_back(x1.at(i));
            x2_used.push_back(x2.at(i));
            x1_temp.push_back(x1.at(i));
            x2_temp.push_back(x2.at(i));
            featuresImg1.push_back(matVector(x1.at(i)));
            featuresImg2.push_back(matVector(x2.at(i)));
        }
    }

    if(VISUAL_DEBUG) {
        visualizeMatches(image_1_color, image_2_color, x1_temp, x2_temp, 3, true, "Point Homography 2 matches");
        visualizeHomography(H2, image_1_color, image_2_color, "Point homography 2");
    }

    F = crossProductMatrix(e2)*H1;
    enforceRankTwoConstraint(F);

    if(LOG_DEBUG) std::cout << "-- Used matches: " << x1_used.size() << std::endl;

    if(x1_used.size() >= 16) successful = true;

    meanSymmetricTranferError = meanSquaredSymmeticTransferError(F, x1_used, x2_used);

    if(VISUAL_DEBUG) visualizeMatches(image_1_color, image_2_color, x1_used, x2_used, 3, true, "Point Homography used point matches");

    if(LOG_DEBUG) std::cout << std::endl << std::endl;

    return successful;
}
