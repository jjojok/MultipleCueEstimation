#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "MultipleCueEstimation.h"
#include "Utility.h"
#include "Statics.h"

#include <iostream>
#include <fstream>
#include "clm/SevenpointLevenbergMarquardt.h"

//TODO:
//- Fro refinement: select best F based on inlaiers not genral error
//- Find a good starting value for ransac thr according to noralization
//- Debug output: number of correct matches for each estimation
//- Use sampson error for HPoints (Hartley, Ziss: p98 (Homography), p287 (Fundamental)
//- Discard global optimization if result is worse then one of the intermediate results
//- Look at http://users.ics.forth.gr/~lourakis/fundest/ and levmar and http://users.ics.forth.gr/~lourakis/homest/
//- Refine F: Compute Projection matrices and take mean of rotations/translations (P1 = [I|0]; P2 = [[e2]xF12 | e2])

//MAYBE:
//- F from curves: FindContours -> compute slope for every point on contour, match slopes (include in matching: slope, point distance, color, intensety...)

//General Info:
//- Sampson/Reprojection error X = (x1, y1, x2, y2) in homogenen korrds
//- LM: Optimize all 9 matrix elements


Mat *getGroundTruthKRt(Mat K1, Mat K2, Mat Rw1, Mat Rw2, Mat Tw1, Mat Tw2) {      //Compute F from K,R,t in world coords

    Mat R12 = Rw2.t()*Rw1;                  //Rotation from camera 1 to camera 2
    Mat T12 = Rw2.t()*Mat(Tw2 - Tw1);      //Translation from camera 1 to camera 2

    Mat P2;
    hconcat(R12, T12, P2);
    P2 = K2*P2;

    Mat P1p, C;
    vconcat(K1.inv(DECOMP_SVD), Mat::zeros(1,3,CV_64FC1), P1p);         //Pseudo inverse of P1 (P1^t = [K1^(-1); 0^t]
    vconcat(Mat::zeros(3,1,CV_64FC1), Mat::ones(1,1,CV_64FC1), C);      //Camera center image 1

    Mat * F = new Mat(crossProductMatrix(P2*C)*P2*P1p);     //F = [P'*C]x*P'*P^+(See Hartley, Zisserman: p. 244)
    *F = *F / F->at<double>(2,2);

//    if (LOG_DEBUG) {
//        std::cout << "Computation of ground truth: " << std::endl;
//        std::cout << "-- K1 = " << std::endl << K1 << std::endl;
//        std::cout << "-- K2 = " << std::endl << K2 << std::endl;
//        std::cout << "-- T1w = " << std::endl << T1w << std::endl;
//        std::cout << "-- T2w = " << std::endl << T2w << std::endl;
//        std::cout << "-- R1w = " << std::endl << R1w << std::endl;
//        std::cout << "-- R2w = " << std::endl << R2w << std::endl;
//        std::cout << "-- R12 = " << std::endl << R12 << std::endl;
//        std::cout << "-- T12 = " << std::endl << T12 << std::endl;
//        std::cout << "-- P1p = " << std::endl << P1p << std::endl;
//        std::cout << "-- P2 = " << std::endl << P2 << std::endl;
//        std::cout << "-- F = " << std::endl << *F << std::endl;
//        std::cout << std::endl;
//    }

    return F;

}

int main(int argc, char** argv )
{
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
        Mat K1, K2, R1w, R2w, t1w, t2w;
        if(ImgParamsFromFile(argv[4], K1, R1w, t1w) && ImgParamsFromFile(argv[5], K2, R2w, t2w)) {
            mce = new MultipleCueEstimation(&image_1_color, &image_2_color, computations, getGroundTruthKRt(K1, K2, R1w, R2w, t1w, t2w));
        }
    }



    if(mce != 0) {


        time_t startTime = time(0);
        mce->compute();


        //Output results
        if(argc == 6) {
//            std::cout << "first image, second image, featureCountCombined, trueFeatureCountCombined, Fgt: trueSampsonErr, refined F: inlierCountCombined, refined F: trueInlierCountCombined, refined F: sampsonErrCombined, refined F: trueSampsonErr,";

//            for(int i = 0; i < mce->getEstimations().size(); i++){
//                FEstimationMethod estimation = mce->getEstimations().at(i);
//                std::cout << estimation.name << ": featureCountGood, " << estimation.name << ": trueFeatureCountGood, "
//                          << estimation.name << ": featureCountComplete, " << estimation.name << ": trueFeatureCountComplete, "
//                          << estimation.name << ": inlierCountOwnGood, " << estimation.name << ": trueInlierCountOwnGood, "
//                          << estimation.name << ": inlierCountOwnComplete, " << estimation.name << ": trueInlierCountOwnComplete, "
//                          << estimation.name << ": inlierCountCombined, " << estimation.name << ": trueInlierCountCombined, "
//                          << estimation.name << ": sampsonErrOwn, " << estimation.name << ": sampsonErrComplete, "
//                          << estimation.name << ": sampsonErrCombined, " << estimation.name << ": sampsonErrStdDevCombined, " << estimation.name << ": trueSampsonErr, " << estimation.name << ": quality, ";
//            }

            //std::cout << "Time (min)" << std::endl;

            std::cout << argv[1] << "," << argv[2] << "," << mce->combinedFeatures << "," << mce->combinedFeaturesCorrect << "," << mce->groundTruthSampsonDistCombined << "," << mce->refinedFInlierCombined << "," << mce->refinedFTrueInlierCombined << "," << mce->refinedFSampsonDistCombined << "," << mce->refinedFSampsonDistCorrect << ",";
            for(int i = 0; i < mce->getEstimations().size(); i++){
                FEstimationMethod estimation = mce->getEstimations().at(i);
                if(!estimation.isSuccessful()) std::cout << ",,,,,,,,,,,,,,,,";
                else {
                    std::cout << estimation.featureCountGood << "," << estimation.trueFeatureCountGood << ","
                              << estimation.featureCountComplete << "," << estimation.trueFeatureCountComplete << ","
                              << estimation.inlierCountOwnGood << "," << estimation.trueInlierCountOwnGood << ","
                              << estimation.inlierCountOwnComplete << "," << estimation.trueInlierCountOwnComplete << ","
                              << estimation.inlierCountCombined << "," << estimation.trueInlierCountCombined << ","
                              << estimation.sampsonErrOwn << "," << estimation.sampsonErrComplete << ","
                              << estimation.sampsonErrCombined << "," << estimation.sampsonErrStdDevCombined << ","
                              << estimation.trueSampsonErr << "," << estimation.quality << ",";
                }
            }
            std::cout << (time(0) - startTime)/60.0 << std::endl;
        }
    }
    if(VISUAL_DEBUG) waitKey(0);
    SevenpointLevenbergMarquardtExit();        //prevent octave from crashing
}
