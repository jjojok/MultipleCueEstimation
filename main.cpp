#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "MultipleCueEstimation.h"
#include "Utility.h"
#include "Statics.h"

#include <iostream>
#include <fstream>
#include "clm/SevenpointLevenbergMarquardt.h"

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
            mce->K1_gt = K1.clone();
            mce->K2_gt = K2.clone();
        }
    }



    if(mce != 0) {


        time_t startTime = time(0);
        mce->compute();


        //Output results
        if(argc == 6) {
//            std::cout << "first image, second image, featureCountCombined, trueFeatureCountCombined, Fgt: trueSampsonErr, Fgt: trueErrorStdDev, , Fgt: trueRootSampsonErr, Fgt: trueRootErrorStdDev, refined F: inlierCountCombined, refined F: trueInlierCountCombined, refined F: sampsonErrCombined, refined F: trueSampsonErr, refined F: trueSampsonErrStdDev, refined F: trueRootSampsonErr, refined F: trueRootSampsonErrStdDev,";

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

            std::cout << argv[1] << "," << argv[2] << "," << mce->combinedFeatures << "," << mce->combinedFeaturesCorrect << ","
                                 << mce->groundTruthSampsonDistCorrect << "," << mce->groundTruthSampsonDistCorrect << "," << mce->groundTruthRootDistCorrect << "," << mce->groundTruthRootDistCorrect << "," << mce->refinedFInlierCombined << ","
                                 << mce->refinedFTrueInlierCombined << "," << mce->refinedFSampsonDistCombined << ","
                                 << mce->refinedFSampsonDistCorrect << "," << mce->refinedFSampsonErrStdDevCorrect << "," << mce->refinedFRootDistCorrect << "," << mce->refinedFRootErrStdDevCorrect << ",";


            for(int i = 0; i < mce->getEstimations().size(); i++){
                FEstimationMethod estimation = mce->getEstimations().at(i);
                if(!estimation.isSuccessful()) std::cout << ",,,,,,,,,,,,,,,,,,,";
                else {
                    std::cout << estimation.featureCountGood << "," << estimation.trueFeatureCountGood << ","
                              << estimation.featureCountComplete << "," << estimation.trueFeatureCountComplete << ","
                              << estimation.inlierCountOwnGood << "," << estimation.trueInlierCountOwnGood << ","
                              << estimation.inlierCountOwnComplete << "," << estimation.trueInlierCountOwnComplete << ","
                              << estimation.inlierCountCombined << "," << estimation.trueInlierCountCombined << ","
                              << estimation.sampsonErrOwn << "," << estimation.sampsonErrComplete << ","
                              << estimation.sampsonErrCombined << "," << estimation.sampsonErrStdDevCombined << ","
                              << estimation.trueSampsonErr << "," << estimation.trueSampsonErrStdDev << ","
                              << estimation.trueRootSampsonErr << "," << estimation.trueRootSampsonErrStdDev << ","
                              << estimation.quality << ",";
                }
            }
            std::cout << (time(0) - startTime)/60.0 << std::endl;
        }
    }
    if(SHOW_DEBUG_IMG) waitKey(0);
    SevenpointLevenbergMarquardtExit();        //prevent octave from crashing
}
