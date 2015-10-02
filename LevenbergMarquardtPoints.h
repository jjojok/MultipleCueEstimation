#ifndef LMA_POINTS_H
#define LMA_POINTS_H

/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++
* Code partially from: https://github.com/daviddoria/Examples/blob/master/c%2B%2B/Eigen/LevenbergMarquardt/CurveFitting.cpp
* +++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

#include <iostream>

#include "Utility.h"

struct PointFunctor : Functor<double>
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

        //Mat newH_inv = newH.inv(DECOMP_SVD);

        pointCorrespStruct pc;

        for(unsigned int i = 0; i < this->points->pointCorrespondencies.size(); ++i)
        {
            pc = this->points->pointCorrespondencies.at(i);
            fvec(i) = sampsonDistanceHomography(newH, matVector(pc.x1), matVector(pc.x2));
        }

        return 0;
    }

pointSubsetStruct *points;

int inputs() const { return 9; } // There are 9 parameters of the model
int values() const { return points->pointCorrespondencies.size(); } // The number of observations
};

#endif // LMA_POINTS_H
