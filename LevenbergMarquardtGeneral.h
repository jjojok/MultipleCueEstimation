#ifndef LMA_LINES_H
#define LMA_LINES_H

/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++
* Code partially from: https://github.com/daviddoria/Examples/blob/master/c%2B%2B/Eigen/LevenbergMarquardt/CurveFitting.cpp
* +++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

#include <iostream>

#include "DataStructs.h"
#include "Utility.h"

struct GeneralFunctor : Functor<double>
{
    int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
    {

        Mat F = Mat::ones(3,3,CV_64FC1);

        F.at<double>(0,0) = x(0);
        F.at<double>(0,1) = x(1);
        F.at<double>(0,2) = x(2);

        F.at<double>(1,0) = x(3);
        F.at<double>(1,1) = x(4);
        F.at<double>(1,2) = x(5);

        F.at<double>(2,0) = x(6);
        F.at<double>(2,1) = x(7);
        F.at<double>(2,2) = x(8);

        homogMat(F);

        //std::vector<double> errorVect = computeCombinedErrorVect(*estimations, F);
        std::vector<double> errorVect = computeCombinedErrorVect(x1, x2, F);

        for(int i = 0; i < errorVect.size(); i++) {
            fvec(i) = errorVect.at(i);
        }

//        for(std::vector<FEstimationMethod>::const_iterator estimationIter = methods->begin(); estimationIter != methods->end(); ++estimationIter) {

//            if(estimationIter->getType() == F_FROM_LINES_VIA_H) {

//                for(unsigned int i = 0; i < estimationIter->featuresImg1.size()/2; i++)
//                {
//                    Mat line1Start = estimationIter->featuresImg1.at(2*i);
//                    Mat line1End = estimationIter->featuresImg1.at(2*i+1);

//                    Mat line2Start = estimationIter->featuresImg2.at(2*i);
//                    Mat line2End = estimationIter->featuresImg2.at(2*i+1);

//                    Mat A = F*crossProductMatrix(line2Start)*line2End;
//                    Mat start1 = line1Start.t()*A;
//                    Mat end1 = line1End.t()*A;
//                    Mat B = F_T*crossProductMatrix(line1Start)*line1End;
//                    Mat start2 = line2Start.t()*B;
//                    Mat end2 = line2End.t()*B;

//                    fvec(fvecPos++) = Mat(start1+end1).at<double>(0,0);
//                    fvec(fvecPos++) = Mat(start2+end2).at<double>(0,0);
//                }
//            }
//            if(estimationIter->getType() == F_FROM_POINTS) {
//                for(unsigned int i = 0; i < estimationIter->featuresImg1.size(); i++)   //Distance form features to correspondig epipolarline in other image
//                {
//                    Mat x1 = estimationIter->featuresImg1.at(i);
//                    Mat x2 = estimationIter->featuresImg2.at(i);

//                    fvec(fvecPos++) = Mat(x2.t()*F*x1).at<double>(0,0);
//                    fvec(fvecPos++) = Mat(x1.t()*F.t()*x2).at<double>(0,0);
//                }
//            }
//        }

        return 0;
    }

//std::vector<FEstimationMethod> *estimations;
std::vector<Mat> x1;
std::vector<Mat> x2;
int numValues;

int inputs() const { return 9; } // There are 9 parameters of the model
int values() const {
//    int numValues = 0;
//    for(std::vector<FEstimationMethod>::iterator estimationIter = estimations->begin(); estimationIter != estimations->end(); ++estimationIter) {
//        numValues += estimationIter->getFeaturesImg1().size();
//    }
    return numValues;
} // The number of observations
};

#endif // LMA_LINES_H
