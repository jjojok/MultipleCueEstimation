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

        std::vector<double> errorVect = computeCombinedErrorVect(x1, x2, F);

        for(int i = 0; i < errorVect.size(); i++) {
            if(fabs(errorVect.at(i)) <= inlierThr) errorVect.at(i) = 0;        //TODO! remove?
            else {
                if(errorVect.at(i) > 0) fvec(i) = errorVect.at(i) - inlierThr;
                else fvec(i) = errorVect.at(i) + inlierThr;
            }
        }

        return 0;
    }

std::vector<Mat> x1;
std::vector<Mat> x2;
double inlierThr;

int inputs() const { return 9; } // There are 9 parameters of the model
int values() const { return x1.size(); } // The number of observations
};

#endif // LMA_LINES_H
