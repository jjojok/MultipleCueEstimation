#ifndef DATASTRUCTURES_H
#define DATASTRUCTURES_H

#include <opencv2/opencv.hpp>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include <eigen3/unsupported/Eigen/NonLinearOptimization>
#include <eigen3/unsupported/Eigen/NumericalDiff>

using namespace cv;

struct lineCorrespStruct {
    double line1Angle, line2Angle, line1Length, line2Length;
    Mat line1StartNormalized, line1EndNormalized, line2StartNormalized, line2EndNormalized;
    Mat line1Start, line1End, line2Start, line2End;
    int id;
};

struct pointCorrespStruct {
    Point2d x1, x2;
    int id;
};

struct lineCorrespSubsetError {
    int lineCorrespIdx;
    double lineCorrespError;
};

struct lineSubsetStruct {
    std::vector<lineCorrespStruct> lineCorrespondencies;
    double qualityMeasure;
    double meanSquaredSymmeticTransferError;
    Mat Hs, Hs_normalized;
};

struct pointSubsetStruct {
    std::vector<pointCorrespStruct> pointCorrespondencies;
    double qualityMeasure;
    double meanSquaredSymmeticTransferError;
    Mat Hs, Hs_normalized;
};

struct segmentStruct {
    int id;
    int area;
    Point startpoint;
    int contours_idx;
};

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

#endif
