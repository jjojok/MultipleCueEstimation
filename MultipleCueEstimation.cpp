#include "MultipleCueEstimation.h"
#include "LevenbergMarquardtGeneral.h"

using namespace cv;

MultipleCueEstimation::MultipleCueEstimation(Mat *img1, Mat *img2, int comp) {
    image_1_color = img1->clone();
    image_2_color = img2->clone();
    compareWithGroundTruth = false;
    computations = comp;
}

MultipleCueEstimation::MultipleCueEstimation(Mat *img1, Mat *img2, int comp, Mat *F_groudtruth)
{
    image_1_color = img1->clone();
    image_2_color = img2->clone();
    compareWithGroundTruth = false;
    computations = comp;
    compareWithGroundTruth = true;
    Fgt = F_groudtruth->clone();
}

void MultipleCueEstimation::run() {
    std::vector<FEstimationMethod> estimations;
    if (checkData()) {
        if(computations & F_FROM_POINTS) {
            FEstimationMethod* points = calcFfromPoints();
            estimations.push_back(*points);
        }
        if(computations & F_FROM_LINES_VIA_H) {
            FEstimationMethod* lines = calcFfromLines();
            if(lines->isSuccessful()) estimations.push_back(*lines);
        }
        if(computations & F_FROM_PLANES_VIA_H) {
            estimations.push_back(*calcFfromPlanes());
        }

        if (compareWithGroundTruth) {
            std::cout << "Ground truth = " << std::endl << Fgt << std::endl;
            if(VISUAL_DEBUG) {
                //rectify(x1, x2, Fgt, image_1, image_2, "Rectified ground truth");
                drawEpipolarLines(x1,x2, Fgt, image_1, image_2, "F_groundtruth");
            }
        }

        //double error = epipolarSADError(Fgt, x1, x2);

        for (std::vector<FEstimationMethod>::iterator it = estimations.begin() ; it != estimations.end(); ++it) {
            std::cout << "Estimation: " << it->name << " = " << std::endl << it->getF() << std::endl;
            if (compareWithGroundTruth) {
                double error1 = randomSampleSymmeticTransferError(Fgt, it->getF(), image_1, NUM_SAMPLES_F_COMARATION);
                double error2 = squaredError(Fgt, it->getF());
                std::cout << "Random sample epipolar error: " << error1 << ", Squated distance: " << error2 << std::endl;
            }
            if(VISUAL_DEBUG) {
                //rectify(x1, x2, it->getF(), image_1, image_2, "Rectified "+it->name);
                drawEpipolarLines(x1,x2, it->getF(), image_1, image_2, it->name);
            }
        }

        Mat F = refineF(estimations);
        std::cout << "Refined F = " << std::endl << F << std::endl;
        if (compareWithGroundTruth) {
            double error1 = randomSampleSymmeticTransferError(Fgt, F, image_1, NUM_SAMPLES_F_COMARATION);
            double error2 = squaredError(Fgt, F);
            std::cout << "Random sample epipolar error: " << error1 << ", Squated distance: " << error2 << std::endl;
        }

        waitKey(0);
    }

}

int MultipleCueEstimation::checkData() {
    if(!image_1_color.data || !image_2_color.data)
    {
        std::cerr << "No image data!" << std::endl;
        return 0;
    }

    if(compareWithGroundTruth && !Fgt.data) {
        std::cerr << "No ground truth data!" << std::endl;
        return 0;
    }

    if(image_1_color.channels() > 1) {
        cvtColor(image_1_color, image_1, CV_BGR2GRAY);
        cvtColor(image_2_color, image_2, CV_BGR2GRAY);
    } else {
        image_1 = image_1_color;
        image_2 = image_2_color;
    }

    if(VISUAL_DEBUG) {
        showImage("Image 1 original", image_1_color);
        showImage("Image 2 original", image_2_color);
    }

    return 1;
}

FEstimationMethod* MultipleCueEstimation::calcFfromPoints() {
    FEstimatorPoints* estimatorPoints = new FEstimatorPoints(image_1, image_2, image_1_color, image_2_color, "F_points");
    estimatorPoints->compute();
    return estimatorPoints;
}

FEstimationMethod* MultipleCueEstimation::calcFfromLines() {     // From Paper: "Robust line matching in image pairs of scenes with dominant planes", Sagúes C., Guerrero J.J.
    FEstimatorHLines* estomatorLines = new FEstimatorHLines(image_1, image_2, image_1_color, image_2_color, "F_lines");
    estomatorLines->compute();
    return estomatorLines;
}

FEstimationMethod* MultipleCueEstimation::calcFfromPlanes() {    // From: 1. two Homographies (one in each image), 2. Planes as additinal point information (point-plane dualism)
    FEstimatorHPlanes* estomatorPlanes = new FEstimatorHPlanes(image_1, image_2, image_1_color, image_2_color, "F_Planes");
    estomatorPlanes->compute();
    return estomatorPlanes;
}

FEstimationMethod* MultipleCueEstimation::calcFfromConics() {    // Maybe: something with vanishing points v1*w*v2=0 (Hartley, Zissarmen p. 235ff)

}

FEstimationMethod* MultipleCueEstimation::calcFfromCurves() {    // First derivative of corresponding curves are gradients to the epipolar lines

}

Mat MultipleCueEstimation::refineF(std::vector<FEstimationMethod> estimations) {    //Reduce error of F AFTER computing it seperatly form different sources

    if (LOG_DEBUG) std::cout << std::endl << "Refinement of computed Fundamental Matrices" << std::endl;

    //Select F with smallest error with respect to all features as starting point

    FEstimationMethod bestMethod = *estimations.begin();
    Mat refinedF = Mat(3,3,CV_64FC1);
    int numValues = 0;

    for(std::vector<FEstimationMethod>::iterator estimationIter = estimations.begin(); estimationIter != estimations.end(); ++estimationIter) {
        if(estimationIter->computeMeanError(estimations) < bestMethod.getError()) bestMethod = *estimationIter;
        numValues += estimationIter->getFeaturesImg1().size();
    }

    Eigen::VectorXd x(9);

    x(0) = bestMethod.getF().at<double>(0,0);
    x(1) = bestMethod.getF().at<double>(0,1);
    x(2) = bestMethod.getF().at<double>(0,2);

    x(3) = bestMethod.getF().at<double>(1,0);
    x(4) = bestMethod.getF().at<double>(1,1);
    x(5) = bestMethod.getF().at<double>(1,2);

    x(6) = bestMethod.getF().at<double>(2,0);
    x(7) = bestMethod.getF().at<double>(2,1);
    x(8) = bestMethod.getF().at<double>(2,2);

    GeneralFunctor functor;
    functor.estimations = &estimations;
    functor.numValues = numValues;
    Eigen::NumericalDiff<GeneralFunctor> numDiff(functor, 1.0e-3); //epsilon
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<GeneralFunctor>,double> lm(numDiff);

    lm.parameters.ftol = 1.0e-10;
    lm.parameters.xtol = 1.0e-10;
    //lm.parameters.epsfcn = 1.0e-3;
    lm.parameters.maxfev = 4000; // Max iterations
    Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(x);

    if (LOG_DEBUG) std::cout << "-- LMA Iterations: " << lm.nfev << ", Status: " << status << std::endl;

    refinedF.at<double>(0,0) = x(0);
    refinedF.at<double>(0,1) = x(1);
    refinedF.at<double>(0,2) = x(2);

    refinedF.at<double>(1,0) = x(3);
    refinedF.at<double>(1,1) = x(4);
    refinedF.at<double>(1,2) = x(5);

    refinedF.at<double>(2,0) = x(6);
    refinedF.at<double>(2,1) = x(7);
    refinedF.at<double>(2,2) = x(8);

    if (LOG_DEBUG) std::cout <<"Error: " << estimations.begin()->computeMeanError(estimations, refinedF) << std::endl;

    return refinedF;
}



