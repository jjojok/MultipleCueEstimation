#include "FEstimationMethod.h"

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

double FEstimationMethod::computeMeanError(std::vector<FEstimationMethod> estimations) {
    return computeMeanError(estimations, this->F);
}

double FEstimationMethod::computeMeanError(std::vector<FEstimationMethod> estimations, Mat impF) {
    std::vector<double> errorVect = computeErrorVect(estimations, impF);
    error = 0;
    for(std::vector<double>::const_iterator errorIter = errorVect.begin(); errorIter != errorVect.end(); ++errorIter) {
        error += *errorIter;
    }
    return error/errorVect.size();
}

double FEstimationMethod::getError() {
    return error;
}

std::vector<double> FEstimationMethod::computeErrorVect(std::vector<FEstimationMethod> estimations) {
    return computeErrorVect(estimations, this->F);
}

std::vector<double> FEstimationMethod::computeErrorVect(std::vector<FEstimationMethod> estimations, Mat inpF) {

    std::vector<double> *errorVect = new std::vector<double>();
    Mat F_T = inpF.t();
    Mat F_invT = inpF.inv(DECOMP_SVD).t();

    for(std::vector<FEstimationMethod>::iterator estimationIter = estimations.begin(); estimationIter != estimations.end(); ++estimationIter) {

//        if(estimationIter->getType() == F_FROM_LINES_VIA_H) {

//            for(unsigned int i = 0; i < estimationIter->featuresImg1.size()/2; i++)
//            {
//                Mat line1Start = estimationIter->featuresImg1.at(2*i);
//                Mat line1End = estimationIter->featuresImg1.at(2*i+1);

//                Mat line2Start = estimationIter->featuresImg2.at(2*i);
//                Mat line2End = estimationIter->featuresImg2.at(2*i+1);

//                Mat A = inpF*crossProductMatrix(line2Start)*line2End;
//                Mat start1 = line1Start.t()*A;
//                Mat end1 = line1End.t()*A;
//                Mat B = inpF.t()*crossProductMatrix(line1Start)*line1End;
//                Mat start2 = line2Start.t()*B;
//                Mat end2 = line2End.t()*B;

//                errorVect->push_back(Mat(start1+end1).at<double>(0,0));
//                errorVect->push_back(Mat(start2+end2).at<double>(0,0));
//            }
//        }
//        if(estimationIter->getType() == F_FROM_POINTS) {
            for(unsigned int i = 0; i < estimationIter->featuresImg1.size(); i++)   //Distance form features to correspondig epipolarline in other image
            {
                Mat x1 = estimationIter->featuresImg1.at(i);
                Mat x2 = estimationIter->featuresImg2.at(i);

                errorVect->push_back(Mat(x2.t()*inpF*x1 + x1.t()*inpF.t()*x2).at<double>(0,0));
            }
        //}
    }
    return *errorVect;
}
