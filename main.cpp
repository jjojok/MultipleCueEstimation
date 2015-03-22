#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "MultipleCueEstimation.h"
#include "Utility.h"

//TODO:
//- fix ground truth computation
//- Change random point error computation: move only a small distance on the epipolar line (only one random point)
//- use FindHomography from oopencv to use points in general position for computation

//MAYBE:
//- tweak dinamic line corresp. reduction
//- Use levenberg marquardt in Eigen to solve minimization in ransac

Mat *getGroundTruthP(Mat K1, Mat K2, Mat R1w, Mat R2w, Mat T1w, Mat T2w) {      //Compute F from K,R,t in world coords, see A_General_Expression_of_the_Fundamental_Matrix_for_Both_Perspective_and_Affine_Cameras p2-3

    R1w = R1w.t();      //Strecha Rotation matrices are transposed
    R2w = R2w.t();

    Mat P1;
    Mat P2;

    hconcat(R1w, T1w, P1);
    hconcat(R2w, T2w, P2);

    P1 = K1*P1;
    P2 = K2*P2;

    Mat P1p = P1.t()*(P1*P1.t()).inv(DECOMP_SVD);   //Pseudo inverse of P1
    Mat P2p = P2.t()*(P2*P2.t()).inv(DECOMP_SVD);   //Pseudo inverse of P2

    Mat p = (Mat::eye(4,4,CV_64FC1)-P1p*P1)*Mat::ones(4,1,CV_64FC1);

    Mat * F = new Mat((crossProductMatrix(P1*p)*P1*P2p));
    *F = *F / F->at<double>(2,2);

    if (LOG_DEBUG) {
        std::cout << "Computation of ground truth: " << std::endl;
        std::cout << "-- F = " << std::endl << *F << std::endl;
        std::cout << std::endl;
    }

    return F;

}


Mat *getGroundTruthKRt(Mat K1, Mat K2, Mat R1w, Mat R2w, Mat T1w, Mat T2w) {      //Compute F from K,R,t in world coords

    R1w = R1w.t();      //Strecha Rotation matrices are transposed
    R2w = R2w.t();

    Mat R12 = R1w.t()*R2w;                    //Rotation from camera 1 to camera 2
    Mat T12 = -R1w*Mat(T2w - T1w).rowRange(0,3);   //Translation from camera 1 to camera 2

    Mat P2;
    hconcat(R12, T12, P2);
    P2 = K2*P2;                              //P2 = K2[R12|t12]

    Mat P1p, C;
    vconcat(K1.inv(DECOMP_SVD), Mat::zeros(1,3,CV_64FC1), P1p);         //Pseudo inverse of P1 (P1^t = [K1^(-1); 0^t]
    vconcat(Mat::zeros(3,1,CV_64FC1), Mat::ones(1,1,CV_64FC1), C);      //Camera center image 1


    Mat * F = new Mat(crossProductMatrix(P2*C)*P2*P1p);     //F = [P'*C]x*P'*P^+(See Hartley, Zisserman: p. 244)
    *F = *F / F->at<double>(2,2);
    //*F = *F * 10;
    //F->at<double>(2,2) = 1.0;

    if (LOG_DEBUG) {
        std::cout << "Computation of ground truth: " << std::endl;
        std::cout << "-- K1 = " << std::endl << K1 << std::endl;
        std::cout << "-- K2 = " << std::endl << K2 << std::endl;
        std::cout << "-- T1w = " << std::endl << T1w << std::endl;
        std::cout << "-- T2w = " << std::endl << T2w << std::endl;
        std::cout << "-- R1w = " << std::endl << R1w << std::endl;
        std::cout << "-- R2w = " << std::endl << R2w << std::endl;
        std::cout << "-- R12 = " << std::endl << R12 << std::endl;
        std::cout << "-- T12 = " << std::endl << T12 << std::endl;
        std::cout << "-- P1p = " << std::endl << P1p << std::endl;
        std::cout << "-- P2 = " << std::endl << P2 << std::endl;
        std::cout << "-- F = " << std::endl << *F << std::endl;
        std::cout << std::endl;
    }

    return F;

}

int main(int argc, char** argv )
{
    //freopen( "error.txt", "w", stderr );

    if (argc != 3 && argc != 4 && argc != 6) {
        std::cerr << "Usage: MultipleCueEstimation <path to first image> <path to second image> <optional: path to first camera matrix> <optional: path to second camera matrix>" << std::endl;
        return 0;
    }

    Mat image_1_color, image_2_color;
    int computations = 0;
    MultipleCueEstimation* mce;

    if(argc == 3) { //Two images in one
        Mat image_color = imread(argv[1], CV_LOAD_IMAGE_COLOR);

        image_1_color = image_color(Rect(0, 0, image_color.cols/2, image_color.rows));
        image_2_color = image_color(Rect(image_color.cols/2, 0, image_color.cols/2, image_color.rows));
        computations = atoi(argv[2]);
    } else {
        image_1_color = imread(argv[1], CV_LOAD_IMAGE_COLOR).clone();
        image_2_color = imread(argv[2], CV_LOAD_IMAGE_COLOR).clone();
        computations = atoi(argv[3]);
    }

    if(argc < 6) {
        mce = new MultipleCueEstimation(&image_1_color, &image_2_color, computations);
    } else {
//        Mat P1w = MatFromFile(argv[4], 3); //P1 in world coords
//        Mat P2w = MatFromFile(argv[5], 3); //P2 in world coords
        //Mat K1 = Mat::zeros(3,3,CV_64FC1), K2 = Mat::zeros(3,3,CV_64FC1), R1w = Mat::zeros(3,3,CV_64FC1), R2w = Mat::zeros(3,3,CV_64FC1), t1w = Mat::zeros(3,1,CV_64FC1), t2w = Mat::zeros(3,1,CV_64FC1);
        Mat K1, K2, R1w, R2w, t1w, t2w;
        Mat K12, K22, R1w2, R2w2, t1w2, t2w2;
        if(ImgParamsFromFile(argv[4], K1, R1w, t1w) && ImgParamsFromFile(argv[5], K2, R2w, t2w)) {
            getGroundTruthKRt(K1, K2, R1w, R2w, t1w, t2w);
            ImgParamsFromFile(argv[5], K22, R2w2, t2w2);
            ImgParamsFromFile(argv[4], K12, R1w2, t1w2);
            getGroundTruthP(K12, K22, R1w2, R2w2, t1w2, t2w2);
            mce = new MultipleCueEstimation(&image_1_color, &image_2_color, computations, getGroundTruthKRt(K1, K2, R1w, R2w, t1w, t2w));
        }
    }
    if(mce != 0) mce->run();
}
