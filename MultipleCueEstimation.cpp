#include "MultipleCueEstimation.h"

using namespace cv;

MultipleCueEstimation::MultipleCueEstimation()
{
    compareWithGroundTruth = false;
    computations = 0;
}

void MultipleCueEstimation::run() {
    Mat Fgt;       //Ground truth Fundamental matrix
    std::vector<FEstimationMethod> fundamentalMatrices;
    if (loadData()) {
        if(computations & F_FROM_POINTS) {
            fundamentalMatrices.push_back(*calcFfromPoints());
        }
        if(computations & F_FROM_LINES) {
            FEstimationMethod* lines = calcFfromLines();
            fundamentalMatrices.push_back(*lines);
            Mat transformed;
            warpPerspective(image_2, transformed, lines->getF(), Size(image_2.cols,image_2.rows));
            showImage("H21", transformed);
        }
        if(computations & F_FROM_PLANES) {
            fundamentalMatrices.push_back(*calcFfromPlanes());
        }

        for (std::vector<FEstimationMethod>::iterator it = fundamentalMatrices.begin() ; it != fundamentalMatrices.end(); ++it) {
            //it->error = averageSquaredError(Fgt,it->F)[0];
            double error = epipolarSADError(it->getF(), x1, x2);
            std::cout << "Estimation: " << it->name << " = " << std::endl << it->getF() << std::endl << "Error: " << error << std::endl;
        }

        if (compareWithGroundTruth) {   //Compare to ground truth
            Fgt = getGroundTruth();
            double error = epipolarSADError(Fgt, x1, x2);
            std::cout << "F_groundtruth = " << std::endl << Fgt << std::endl << "Error: " << error << std::endl;
        }

        //rectify(x1, x2, Fgt, image_1, 1, "Image 1 rect Fgt");
        //rectify(x1, x2, Fpt, image_1, 1, "Image 1 rect Fpt");

        //drawEpipolarLines(x1, x2, Fgt, image_1.clone(), image_2.clone());

        waitKey(0);
        //Mat h = findHomography(x1, x2, noArray(), CV_RANSAC, 3);
    }

}

int MultipleCueEstimation::loadData() {
    if (arguments != 4 && !compareWithGroundTruth) {
        std::cout << "Usage: MultipleCueEstimation <path to first image> <path to second image> <optional: path to first camera matrix> <optional: path to second camera matrix>" << std::endl;
        return 0;
    }

    image_1 = imread(path_img1, CV_LOAD_IMAGE_GRAYSCALE);
    image_2 = imread(path_img2, CV_LOAD_IMAGE_GRAYSCALE);

    image_1_color = imread(path_img1, CV_LOAD_IMAGE_COLOR);
    image_2_color = imread(path_img2, CV_LOAD_IMAGE_COLOR);

    if ( !image_1.data || !image_2.data || !image_1_color.data || !image_2_color.data )
    {
        printf("No image data \n");
        return 0;
    }

    if(VISUAL_DEBUG) {
        showImage("Image 1 original", image_1);

        showImage("Image 2 original", image_2);

    }

    return 1;
}

FEstimationMethod* MultipleCueEstimation::calcFfromPoints() {
    FEstimatorPoints* estimatorPoints = new FEstimatorPoints(image_1, image_2, image_1_color, image_2_color, "F_points");
    estimatorPoints->compute();
    x1 = estimatorPoints->getX1();
    x2 = estimatorPoints->getX2();
    return estimatorPoints;
}

FEstimationMethod* MultipleCueEstimation::calcFfromLines() {     // From Paper: "Robust line matching in image pairs of scenes with dominant planes", Sagúes C., Guerrero J.J.
    FEstimatorLines* estomatorLines = new FEstimatorLines(image_1, image_2, image_1_color, image_2_color, "F_lines");
    estomatorLines->compute();
    return estomatorLines;
}

FEstimationMethod* MultipleCueEstimation::calcFfromPlanes() {    // From: 1. two Homographies (one in each image), 2. Planes as additinal point information (point-plane dualism)
    FEstimatorPlanes* estomatorPlanes = new FEstimatorPlanes(image_1, image_2, image_1_color, image_2_color, "F_Planes");
    estomatorPlanes->compute();
    return estomatorPlanes;
}

FEstimationMethod* MultipleCueEstimation::calcFfromConics() {    // Maybe: something with vanishing points v1*w*v2=0 (Hartley, Zissarmen p. 235ff)

}

FEstimationMethod* MultipleCueEstimation::calcFfromCurves() {    // First derivative of corresponding curves are gradients to the epipolar lines

}

Mat MultipleCueEstimation::refineF() {    //Reduce error of F AFTER computing it seperatly form different sources

}

Mat MultipleCueEstimation::getGroundTruth() {
    Mat P1w = MatFromFile(path_P1, 3); //P1 in world coords
    Mat P2w = MatFromFile(path_P2, 3); //P2 in world coords
    Mat T1w, R1w;   //World rotation, translation
    Mat K1; //calibration matrix
    Mat Trel; //Relative translation

    if (LOG_DEBUG) {
        std::cout << "P1w = " << std::endl << P1w << std::endl;
    }

    decomposeProjectionMatrix(P1w, K1, R1w, T1w, noArray(), noArray(), noArray(), noArray() );

    T1w = T1w/T1w.at<float>(3);      //convert to homogenius coords

    if (LOG_DEBUG) {
        std::cout << "T1w = " << std::endl << T1w << std::endl;
        std::cout << "R1w = " << std::endl << R1w << std::endl;
        std::cout << "Trel = " << std::endl << Trel << std::endl;
    }

    Mat F = crossProductMatrix(P2w*T1w)*P2w*P1w.inv(DECOMP_SVD); //(See Hartley, Zisserman: p. 244)

    return F / F.at<float>(2,2);       //Set global arbitrary scale factor 1 -> easier to compare
}



