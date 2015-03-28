#include "Utility.h"

FEstimationMethod::FEstimationMethod() {

}

FEstimationMethod::~FEstimationMethod() {

}

int FEstimationMethod::extractMatches() {

}

bool FEstimationMethod::compute() {

}

Mat FEstimationMethod::getF() {
    return F;
}

void FEstimationMethod::init() {

}

Mat FEstimationMethod::normalize(Mat T, Mat p) {
    return T*p;
}

Mat FEstimationMethod::normalize(Mat T, double x, double y, double z) {
    Mat *p = new Mat(3,1,CV_64FC1);

    p->at<double>(0,0) = x;
    p->at<double>(1,0) = y;
    p->at<double>(2,0) = 1;

    return normalize(T, *p);
}

Mat FEstimationMethod::denormalize(Mat M, Mat T1, Mat T2) {
    Mat T = T2.inv(DECOMP_SVD)*M*T1;
    T = T / T.at<double>(2,2);
    return T;
}

bool FEstimationMethod::isSuccessful() {
    return successful;
}


std::vector<Mat> FEstimationMethod::getFeaturesImg1() {
    return featuresImg1;
}

std::vector<Mat> FEstimationMethod::getFeaturesImg2() {
    return featuresImg2;
}

int FEstimationMethod::getType() {
    return computationType;
}

double FEstimationMethod::getError() {
    return error;
}

double FEstimationMethod::getMeanSquaredCSTError() {
    return meanSquaredCSTError;
}

double FEstimationMethod::getMeanSquaredRSSTError() {
    return meanSquaredRSSTError;
}
