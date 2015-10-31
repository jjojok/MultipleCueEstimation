#ifndef LMA_LINES_H
#define LMA_LINES_H

/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++
* Code partially from: https://github.com/daviddoria/Examples/blob/master/c%2B%2B/Eigen/LevenbergMarquardt/CurveFitting.cpp
* +++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

#include <iostream>

#include "DataStructs.h"
#include "FEstimatorHLines.h"


struct LineFunctor : Functor<double>
{
    int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
    {
        Mat newH = Mat::ones(3,3,CV_64FC1);

        newH.at<double>(0,0) = x(0);
        newH.at<double>(0,1) = x(1);
        newH.at<double>(0,2) = x(2);

        newH.at<double>(1,0) = x(3);
        newH.at<double>(1,1) = x(4);
        newH.at<double>(1,2) = x(5);

        newH.at<double>(2,0) = x(6);
        newH.at<double>(2,1) = x(7);
        newH.at<double>(2,2) = x(8);

        homogMat(newH);

        Mat H_inv = newH.inv(DECOMP_SVD);

        lineCorrespStruct lc;

        for(unsigned int i = 0; i < this->lines->lineCorrespondencies.size(); ++i)
        {
            lc = this->lines->lineCorrespondencies.at(i);
            fvec(i) = sampsonDistanceHomography(newH, H_inv, lc.line1Start, lc.line1End, lc.line2Start, lc.line2End);
        }

        return 0;
    }

lineSubsetStruct *lines;
FEstimatorHLines *estimator;

int inputs() const { return 9; } // There are 9 parameters of the model
int values() const { return lines->lineCorrespondencies.size(); } // The number of observations
};

#endif // LMA_LINES_H
