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

Mat MultipleCueEstimation::compute() {
    if (checkData()) {
        if(computations & F_FROM_POINTS) {
            FEstimationMethod* points = calcFfromPoints();
            estimations.push_back(*points);
        }
        if(computations & F_FROM_LINES_VIA_H) {
            FEstimationMethod* lines = calcFfromHLines();
            if(lines->isSuccessful()) estimations.push_back(*lines);
        }
        if(computations & F_FROM_POINTS_VIA_H) {
            FEstimationMethod* Hpoints = calcFfromHPoints();
            if(Hpoints->isSuccessful()) estimations.push_back(*Hpoints);
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
                it->meanSquaredRSSTError = randomSampleSymmeticTransferError(Fgt, it->getF(), image_1_color, image_2_color, NUM_SAMPLES_F_COMARATION);
                double error2 = squaredError(Fgt, it->getF());
                if(LOG_DEBUG) std::cout << "Random sample epipolar error: " << it->meanSquaredRSSTError << ", Squated distance: " << error2 << ", Mean squared symmetric tranfer error: " << it->getError() << std::endl;
            }
            if(VISUAL_DEBUG) {
                //rectify(x1, x2, it->getF(), image_1, image_2, "Rectified "+it->name);
                drawEpipolarLines(x1,x2, it->getF(), image_1, image_2, it->name);
            }
        }

        F = refineF(estimations);
        std::cout << "Refined F = " << std::endl << F << std::endl;
        if (compareWithGroundTruth) {
            meanSquaredRSSTError = randomSampleSymmeticTransferError(Fgt, F, image_1_color, image_2_color, NUM_SAMPLES_F_COMARATION);
            double error2 = squaredError(Fgt, F);
            if(LOG_DEBUG) std::cout << "Random sample epipolar error: " << meanSquaredRSSTError << ", Squated distance: " << error2 << std::endl;
        }
        if(VISUAL_DEBUG) {
            //rectify(x1, x2, it->getF(), image_1, image_2, "Rectified "+it->name);
            drawEpipolarLines(x1,x2, F, image_1, image_2, "Refined F");
            waitKey(0);
        }

        if(LOG_DEBUG) std::cout << "done." << std::endl;
    }
    return F;
}

int MultipleCueEstimation::checkData() {
    if(!image_1_color.data || !image_2_color.data)
    {
        std::cerr << "No image data!" << std::endl;
        return 0;
    }

    if(image_1_color.cols != image_2_color.cols || image_1_color.rows != image_2_color.rows)
    {
        std::cerr << "Image sizes do not match!" << std::endl;
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
    x1 = estimatorPoints->getUsedX1();
    x2 = estimatorPoints->getUsedX2();
    return estimatorPoints;
}

FEstimationMethod* MultipleCueEstimation::calcFfromHLines() {     // From Paper: "Robust line matching in image pairs of scenes with dominant planes", Sagúes C., Guerrero J.J.
    FEstimatorHLines* estomatorLines = new FEstimatorHLines(image_1, image_2, image_1_color, image_2_color, "F_Hlines");
    estomatorLines->compute();
    return estomatorLines;
}

FEstimationMethod* MultipleCueEstimation::calcFfromHPoints() {    // From: two Homographies (one in each image) computed from points in general position
    FEstimatorHPoints* estomatorHPoints = new FEstimatorHPoints(image_1, image_2, image_1_color, image_2_color, "F_HPoints");
    estomatorHPoints->compute();
    return estomatorHPoints;
}

FEstimationMethod* MultipleCueEstimation::calcFfromHPlanes() {    // From: two Homographies (one in each image)
    FEstimatorHPlanes* estomatorPlanes = new FEstimatorHPlanes(image_1, image_2, image_1_color, image_2_color, "F_HPlanes");
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
    Mat refinedF = Mat::ones(3,3,CV_64FC1);
    int numValues = 0;

    for(std::vector<FEstimationMethod>::iterator estimationIter = estimations.begin(); estimationIter != estimations.end(); ++estimationIter) {
        estimationIter->meanSquaredCSTError = computeCombinedMeanSquaredError(estimations, estimationIter->getF());
        if (LOG_DEBUG) std::cout << "Computing meanSquaredCSTError for " << estimationIter->name << ": " << estimationIter->meanSquaredCSTError << std::endl;
        if(bestMethod.meanSquaredCSTError > estimationIter->meanSquaredCSTError) {
            bestMethod = *estimationIter;
        }
        numValues += estimationIter->getFeaturesImg1().size();
    }

    refinedF = bestMethod.getF().clone();

    if (LOG_DEBUG) std::cout << "Starting point for optimization: " << bestMethod.name << std::endl;
    double lastError = 0;
    int stableSolutions = 0;
    int iterations = 1;

    do {

        Eigen::VectorXd x(9);

        x(0) = refinedF.at<double>(0,0);
        x(1) = refinedF.at<double>(0,1);
        x(2) = refinedF.at<double>(0,2);

        x(3) = refinedF.at<double>(1,0);
        x(4) = refinedF.at<double>(1,1);
        x(5) = refinedF.at<double>(1,2);

        x(6) = refinedF.at<double>(2,0);
        x(7) = refinedF.at<double>(2,1);

        std::vector<Mat> goodCombindX1;
        std::vector<Mat> goodCombindX2;
        findGoodCombinedMatches(estimations, goodCombindX1, goodCombindX2, refinedF, 0.01);
        if (LOG_DEBUG) std::cout << "-- Refinement Iteration " << iterations << ", Refined feature count: " << goodCombindX1.size() << "/" << numValues << std::endl;

        GeneralFunctor functor;
        //functor.estimations = &estimations;
        functor.x1 = goodCombindX1;
        functor.x2 = goodCombindX2;
        functor.numValues = goodCombindX1.size();
        //functor.numValues = numValues;
        Eigen::NumericalDiff<GeneralFunctor> numDiff(functor, 1.0e-6); //epsilon
        Eigen::LevenbergMarquardt<Eigen::NumericalDiff<GeneralFunctor>,double> lm(numDiff);
        lm.parameters.ftol = 1.0e-15;
        lm.parameters.xtol = 1.0e-15;
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

        enforceRankTwoConstraint(refinedF);

        //meanSquaredCSTError = computeCombinedMeanSquaredError(estimations, refinedF);
        meanSquaredCSTError = computeCombinedMeanSquaredError(goodCombindX1, goodCombindX2, refinedF);

        if((lastError - meanSquaredCSTError)/meanSquaredCSTError < 0.01) stableSolutions++;
        else stableSolutions = 0;

        if (LOG_DEBUG) std::cout <<"Mean sqared symmetic transfer error: " << meanSquaredCSTError << ", rel. error change: " << (lastError - meanSquaredCSTError)/meanSquaredCSTError << ", stable solutions: " << stableSolutions << std::endl;

        lastError = meanSquaredCSTError;

        iterations++;

    } while(stableSolutions < 3 && iterations < MAX_REFINEMENT_ITERATIONS);

    return refinedF;
}

double MultipleCueEstimation::getMeanSquaredCSTError() {
    return meanSquaredCSTError;
}

double MultipleCueEstimation::getMeanSquaredRSSTError() {
    return meanSquaredRSSTError;
}

std::vector<FEstimationMethod> MultipleCueEstimation::getEstimations() {
    return estimations;
}

