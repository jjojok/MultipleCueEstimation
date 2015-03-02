#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "MultipleCueEstimation.h"
#include "Utility.h"

//TODO:
//Lines:
//- Check if LmedS is correct
//- Check if the line projectionDistance is correct (maybe visual)
//- Try implementing ransac instead of LMeds
//- Maybe tweak line corresp. reduction



Mat *getGroundTruth(Mat P1w, Mat P2w) {      //Compute F from Ps in world coords
    Mat T1w, R1w;   //World rotation, translation
    Mat K1; //calibration matrix

    decomposeProjectionMatrix(P1w, K1, R1w, T1w);

    T1w = T1w/T1w.at<float>(3);      //convert to homogenius coords

    if (LOG_DEBUG) {
        std::cout << "P1w = " << std::endl << P1w << std::endl;
        std::cout << "T1w = " << std::endl << T1w << std::endl;
        std::cout << "R1w = " << std::endl << R1w << std::endl;
        std::cout << "P2w = " << std::endl << P2w << std::endl;
    }

    Mat* F = new Mat(crossProductMatrix(P2w*T1w)*P2w*P1w.inv(DECOMP_SVD)); //(See Hartley, Zisserman: p. 244)

    *F = *F / F->at<float>(2,2);       //convert to homogenius coords

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
        Mat P1w = MatFromFile(argv[4], 3); //P1 in world coords
        Mat P2w = MatFromFile(argv[5], 3); //P2 in world coords

        mce = new MultipleCueEstimation(&image_1_color, &image_2_color, computations, getGroundTruth(P1w, P2w));
    }
    mce->run();
}
