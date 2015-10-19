#include "SevenpointLevenbergMarquardt.h"

#include <oct.h>
#include <octave.h>
#include <parse.h>
#include <toplev.h>
#include <ov.h>

void SevenpointLevenbergMarquardtExit() {
    clean_up_and_exit(0);
}

void SevenpointLevenbergMarquardtInit() {

    //run octave code
    string_vector argv (2);
    argv(0) = "embedded";
    argv(1) = "-q";

    octave_main (2, argv.c_str_vec (), 1);  //Start octave

    source_file("clm.m");
}

bool SevenpointLevenbergMarquardt(std::vector<double>* F, std::vector<double> x1, std::vector<double> y1, std::vector<double> x2, std::vector<double> y2, double F0, int maxIter, double stopDist)
{

    Matrix X1o = Matrix(x1.size(),1);
    Matrix X2o = Matrix(x1.size(),1);
    Matrix Y1o = Matrix(x1.size(),1);
    Matrix Y2o = Matrix(x1.size(),1);

    octave_idx_type ii = 0;

    for (int i = 0; i < x1.size(); i++) {
        X1o(ii,0) = (x1.at(i));
        Y1o(ii,0) = (y1.at(i));
        X2o(ii,0) = (x2.at(i));
        Y2o(ii,0) = (y2.at(i));
        ii++;
    }

    Matrix FS = Matrix(9,1);

    double norm = 0;

    for(int i = 0; i < 9; i++)
    {
        norm += F->at(i)*F->at(i);
        FS(i,0) = F->at(i);
    }

    //FS /= sqrt(norm); //norm to ||F|| = 1;

    // function u = CLM (FS, X1, Y1, X2, Y2, F0, maxIter, stopDist)
    octave_value_list params;
    params(0) = FS;
    params(1) = X1o;
    params(2) = Y1o;
    params(3) = X2o;
    params(4) = Y2o;
    params(5) = F0;
    params(6) = maxIter;
    params(7) = stopDist;

    octave_value_list octF = feval("CLM", params, 8);
    Array<octave_value> octFm = octF(0).array_value();

    if (!error_state && octF.length() > 0) {
        for(int i = 0; i < 9; i++)
            F->at(i) = octFm(i).double_value();
        return true;
    } else {
        return false;
    }

}
