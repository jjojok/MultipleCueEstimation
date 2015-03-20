#ifndef LMA_LINES_H
#define LMA_LINES_H

/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++
* Code partially from: https://github.com/daviddoria/Examples/blob/master/c%2B%2B/Eigen/LevenbergMarquardt/CurveFitting.cpp
* +++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

#include <iostream>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include <eigen3/unsupported/Eigen/NonLinearOptimization>
#include <eigen3/unsupported/Eigen/NumericalDiff>
#include "FEstimatorLines.h"

// Generic functor
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
    typedef _Scalar Scalar;
    enum {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };
    typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

    int m_inputs, m_values;

    Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
    Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

    int inputs() const { return m_inputs; }
    int values() const { return m_values; }

};


struct LineFunctor : Functor<double>
{
    int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
    {
        Mat newH = Mat::zeros(3,3,CV_32FC1), H_T, H_invT;

        newH.at<float>(0,0) = x(0);
        newH.at<float>(0,1) = x(1);
        newH.at<float>(0,2) = x(2);

        newH.at<float>(1,0) = x(3);
        newH.at<float>(1,1) = x(4);
        newH.at<float>(1,2) = x(5);

        newH.at<float>(2,0) = x(6);
        newH.at<float>(2,1) = x(7);
        newH.at<float>(2,2) = x(8);

        H_T = newH.t();
        H_invT = newH.inv(DECOMP_SVD).t();

        lineCorrespStruct lc;

        for(unsigned int i = 0; i < this->lines->lineCorrespondencies.size(); ++i)
        {
            lc = this->lines->lineCorrespondencies.at(i);

            Mat A = H_T*crossProductMatrix(lc.line2Start)*lc.line2End;
            Mat start1 = lc.line1Start.t()*A;
            Mat end1 = lc.line1End.t()*A;
            Mat B = H_invT*crossProductMatrix(lc.line1Start)*lc.line1End;
            Mat start2 = lc.line2Start.t()*B;
            Mat end2 = lc.line2End.t()*B;

            fvec(2*i) = Mat(start1+start2).at<float>(0,0);
            fvec(2*i+1) = Mat(end1+end2).at<float>(0,0);
        }

        return 0;
    }

lineSubsetStruct *lines;
FEstimatorLines *estimator;

int inputs() const { return 9; } // There are 9 parameters of the model
int values() const { return lines->lineCorrespondencies.size()*2; } // The number of observations
};

#endif // LMA_LINES_H
