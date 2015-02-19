#include "FEstimationMethod.h"

FEstimationMethod::FEstimationMethod() {

}

FEstimationMethod::~FEstimationMethod() {

}

int FEstimationMethod::extractMatches() {

}

Mat FEstimationMethod::compute() {

}

Mat FEstimationMethod::getF() {
    return F;
}

void FEstimationMethod::init() {

}

Mat FEstimationMethod::normalize(Mat T, Mat p) {
    return T*p;
}

Mat FEstimationMethod::normalize(Mat T, float x, float y, float z) {
    Mat *p = new Mat(3,1,CV_32FC1);

    p->at<float>(0,0) = x;
    p->at<float>(1,0) = y;
    p->at<float>(2,0) = 1;

    return normalize(T, *p);
}

Mat FEstimationMethod::denormalize(Mat M) {
    return normT2.inv(DECOMP_SVD)*M*normT1;
}
