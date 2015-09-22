#include "SevenpointLevenbergMarquardt.h"

#include <oct.h>
#include <octave.h>
#include <parse.h>
#include <toplev.h>
#include <ov.h>

bool SevenpointLevenbergMarquardt(std::vector<double>* F, std::vector<double> x1, std::vector<double> y1, std::vector<double> x2, std::vector<double> y2)
{
    //run octave code

    string_vector argv (2);
    argv(0) = "embedded";
    argv(1) = "-q";

    octave_main (2, argv.c_str_vec (), 1);  //Start octave

    source_file("clm_c_bridge.m");

//    Array<double> X1;
//    Array<double> X2;
//    Array<double> Y1;
//    Array<double> Y2;

    Matrix X1o = Matrix(x1.size(),1);
    Matrix X2o = Matrix(x1.size(),1);
    Matrix Y1o = Matrix(x1.size(),1);
    Matrix Y2o = Matrix(x1.size(),1);

    octave_value_list X;

    octave_idx_type ii = 0;

    for (int i = 0; i < x1.size(); i++) {
    //for (octave_idx_type ii = 0; i < n; i++) //convert points to octave variables
//        X1(ii) = (x1.at(i));
//        Y1(ii) = (y1.at(i));
//        X2(ii) = (x2.at(i));
//        Y2(ii) = (y2.at(i));
        X1o(ii,0) = (x1.at(i));
        Y1o(ii,0) = (y1.at(i));
        X2o(ii,0) = (x2.at(i));
        Y2o(ii,0) = (y2.at(i));
        ii++;
    }

    X(0) = X1o;
    X(1) = Y1o;
    X(2) = X2o;
    X(3) = Y2o;

    octave_value_list octF = feval("SPLM", X, 4);
    Array<octave_value> octFm = octF(0).array_value();

    if (!error_state && octF.length() > 0) {
        for(int i = 0; i < 9; i++)
            F->push_back(octFm(i).double_value());
        return true;
    } else {
        return false;
    }

}
