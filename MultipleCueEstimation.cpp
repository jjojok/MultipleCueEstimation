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
            if(points->isSuccessful()) estimations.push_back(*points);
            debug_estimations.push_back(*points);
        }
        if(computations & F_FROM_LINES_VIA_H) {
            FEstimationMethod* lines = calcFfromHLines();
            if(lines->isSuccessful()) estimations.push_back(*lines);
            debug_estimations.push_back(*lines);
        }
        if(computations & F_FROM_POINTS_VIA_H) {
            FEstimationMethod* Hpoints = calcFfromHPoints();
            if(Hpoints->isSuccessful()) estimations.push_back(*Hpoints);
            debug_estimations.push_back(*Hpoints);
        }

        if (compareWithGroundTruth) {
            std::cout << "Ground truth = " << std::endl << Fgt << std::endl;
            if(VISUAL_DEBUG) {
                //rectify(x1, x2, Fgt, image_1, image_2, "Rectified ground truth");
                drawEpipolarLines(x1,x2, Fgt, image_1, image_2, "F_groundtruth");
            }
        }

        std::vector<Mat> x1combined, x2combined;

        for(std::vector<FEstimationMethod>::iterator estimationIter = estimations.begin(); estimationIter != estimations.end(); ++estimationIter) {
            for(int i = 0; i < estimationIter->getFeaturesImg1().size(); i++) {
                x1combined.push_back(estimationIter->getFeaturesImg1().at(i));
                x2combined.push_back(estimationIter->getFeaturesImg2().at(i));
            }
        }

        //double error = epipolarSADError(Fgt, x1, x2);



        F = refineF(estimations);

        for (std::vector<FEstimationMethod>::iterator it = estimations.begin() ; it != estimations.end(); ++it) {
            std::cout << "Estimation: " << it->name << " = " << std::endl << it->getF() << std::endl;
            if (compareWithGroundTruth) {
                //it->meanSquaredRSSTError = randomSampleSymmeticTransferError(Fgt, it->getF(), image_1_color, image_2_color, NUM_SAMPLES_F_COMARATION);
                it->meanSquaredRSSTError = -2;
                double error2 = squaredError(Fgt, it->getF());
                if(LOG_DEBUG) std::cout << "Random sample epipolar error: " << it->meanSquaredRSSTError << ", Squated distance: " << error2 << ", Mean squared symmetric tranfer error: " << it->getError() << std::endl;
                double error3 = meanSampsonFDistanceGoodMatches(Fgt, it->getF(), x1combined, x2combined);
            }
            if(VISUAL_DEBUG) {
                //rectify(x1, x2, it->getF(), image_1, image_2, "Rectified "+it->name);
                drawEpipolarLines(x1,x2, it->getF(), image_1, image_2, it->name);
            }
        }

        if(F.data) {
            std::cout << "Refined F = " << std::endl << F << std::endl;
            if (compareWithGroundTruth) {
                //meanSquaredRSSTError = randomSampleSymmeticTransferError(Fgt, F, image_1_color, image_2_color, NUM_SAMPLES_F_COMARATION);
                meanSquaredRSSTError = -2;
                double error2 = squaredError(Fgt, F);
                if(LOG_DEBUG) std::cout << "Random sample epipolar error: " << meanSquaredRSSTError << ", Squated distance: " << error2 << std::endl;
                double error3 = meanSampsonFDistanceGoodMatches(Fgt, F, x1combined, x2combined);
            }
            if(VISUAL_DEBUG) {
                //rectify(x1, x2, it->getF(), image_1, image_2, "Rectified "+it->name);
                drawEpipolarLines(x1,x2, F, image_1, image_2, "Refined F");
            }
        }

        if(LOG_DEBUG) std::cout << "done." << std::endl;
        if(VISUAL_DEBUG) waitKey(0);
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

}

FEstimationMethod* MultipleCueEstimation::calcFfromConics() {    // Maybe: something with vanishing points v1*w*v2=0 (Hartley, Zissarmen p. 235ff)

}

FEstimationMethod* MultipleCueEstimation::calcFfromCurves() {    // First derivative of corresponding curves are gradients to the epipolar lines

}

Mat MultipleCueEstimation::refineF(std::vector<FEstimationMethod> &estimations) {    //Reduce error of F AFTER computing it seperatly form different sources

    if (LOG_DEBUG) std::cout << std::endl << "Refinement of computed Fundamental Matrices" << std::endl;

    std::vector<Mat> FundMats;
    std::vector< std::vector<Mat> > inliersX1;
    std::vector< std::vector<Mat> > inliersX2;

    combinePointCorrespondecies();

    //Select F with smallest error with respect to all features as starting point

    if(estimations.size() == 0) {
        if (LOG_DEBUG) std::cout << "-- No Fundamental Matrix found!" << std::endl;
        return Mat();
    }

    std::vector<FEstimationMethod>::iterator bestMethod = estimations.begin();
    Mat refinedF = Mat::ones(3,3,CV_64FC1);
    int numValues = 0;

    for(std::vector<FEstimationMethod>::iterator estimationIter = estimations.begin(); estimationIter != estimations.end(); ++estimationIter) {
        estimationIter->meanSquaredCSTError = errorFunctionCombinedMeanSquared(x1Combined, x2Combined, estimationIter->getF());
        if (LOG_DEBUG) std::cout << "Computing mean squared error of combined matches for " << estimationIter->name << ": " << estimationIter->meanSquaredCSTError << std::endl;
        if(bestMethod->meanSquaredCSTError > estimationIter->meanSquaredCSTError) {
            bestMethod = estimationIter;
        }
        numValues += estimationIter->getFeaturesImg1().size();
    }

    if(numValues < 8) {
        if (LOG_DEBUG) std::cout << "-- Not enough features!" << std::endl;
        return Mat();
    }

    refinedF = bestMethod->getF().clone();

    if (LOG_DEBUG) std::cout << "-- Starting point for optimization: " << bestMethod->name << std::endl;

    double errorThr = 0.5*sqrt(bestMethod->meanSquaredCSTError);

    levenbergMarquardt(refinedF, x1Combined, x2Combined, errorThr, 3, 0.05, 3.0);

    double generalError = errorFunctionCombinedMeanSquared(x1Combined, x2Combined, refinedF);
    if (LOG_DEBUG) std::cout << "Computing mean squared error of combined matches for refined F: " << generalError << std::endl;

    return refinedF;
}

double MultipleCueEstimation::getMeanSquaredCSTError() {
    return meanSquaredCombinedError;
}

double MultipleCueEstimation::getMeanSquaredRSSTError() {
    return meanSquaredRSSTError;
}

std::vector<FEstimationMethod> MultipleCueEstimation::getEstimations() {
    return debug_estimations;
}

void MultipleCueEstimation::combinePointCorrespondecies() {
    for(std::vector<FEstimationMethod>::iterator estimationIter = estimations.begin(); estimationIter != estimations.end(); ++estimationIter) {
        int cnt = 0;
        for(int i = 0; i < estimationIter->getFeaturesImg1().size(); i++) {
            Mat x11 = estimationIter->getFeaturesImg1().at(i);
            Mat x12 = estimationIter->getFeaturesImg2().at(i);
            bool add = true;
            for(int j = 0; j < x1Combined.size(); j++) {
                if(isEqualPointCorresp(x11, x12, x1Combined.at(j), x2Combined.at(j))) {
                    add = false;
                    break;
                }
            }
            if(add) {
                x1Combined.push_back(x11);
                x2Combined.push_back(x12);
                cnt++;
            }
        }
        if(LOG_DEBUG) std::cout << estimationIter->name << ": Added " << cnt << "/" << estimationIter->getFeaturesImg1().size() << " Point correspondencies to combined feature vector" << std::endl;
    }
}

Mat MultipleCueEstimation::levenbergMarquardt(Mat Flm, std::vector<Mat> x1, std::vector<Mat> x2, double errorThr, int minFeatureChange, double minErrorChange, double lmErrorThr) {
    double lastError = 0;
    int stableSolutions = 0;
    int iterations = 1;
    int lastFeatureCount = 0;
    double meanSquaredCombinedErr = 0;

    do {

        Eigen::VectorXd x(9);

        x(0) = Flm.at<double>(0,0);
        x(1) = Flm.at<double>(0,1);
        x(2) = Flm.at<double>(0,2);

        x(3) = Flm.at<double>(1,0);
        x(4) = Flm.at<double>(1,1);
        x(5) = Flm.at<double>(1,2);

        x(6) = Flm.at<double>(2,0);
        x(7) = Flm.at<double>(2,1);
        x(8) = Flm.at<double>(2,2);

        std::vector<Mat> goodCombindX1;
        std::vector<Mat> goodCombindX2;

        homogMat(Flm);

        findGoodCombinedMatches(x1, x2, goodCombindX1, goodCombindX2, Flm, errorThr/(iterations));
        if (LOG_DEBUG) std::cout << "-- Refinement Iteration " << iterations << ", Refined feature count: " << goodCombindX1.size() << "/" << x1.size() << ", error threshold: " << errorThr/(iterations) << std::endl;
        //thr = 3.0;

        if(goodCombindX1.size() <= NUMERICAL_OPTIMIZATION_MIN_MATCHES) break;

        GeneralFunctor functor;
        functor.x1 = goodCombindX1;
        functor.x2 = goodCombindX2;
        functor.inlierThr = lmErrorThr;
        Eigen::NumericalDiff<GeneralFunctor> numDiff(functor, 1.0e-6); //epsilon
        Eigen::LevenbergMarquardt<Eigen::NumericalDiff<GeneralFunctor>,double> lm(numDiff);
        lm.parameters.ftol = 1.0e-15;
        lm.parameters.xtol = 1.0e-15;
        lm.parameters.maxfev = 4000; // Max iterations
        Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(x);

        if (LOG_DEBUG) std::cout << "-- LMA Iterations: " << lm.nfev << ", Status: " << status << std::endl;

        Flm.at<double>(0,0) = x(0);
        Flm.at<double>(0,1) = x(1);
        Flm.at<double>(0,2) = x(2);

        Flm.at<double>(1,0) = x(3);
        Flm.at<double>(1,1) = x(4);
        Flm.at<double>(1,2) = x(5);

        Flm.at<double>(2,0) = x(6);
        Flm.at<double>(2,1) = x(7);
        Flm.at<double>(2,2) = x(8);

        enforceRankTwoConstraint(Flm);

        //meanSquaredCSTError = computeCombinedMeanSquaredError(estimations, refinedF);
        meanSquaredCombinedErr = errorFunctionCombinedMeanSquared(goodCombindX1, goodCombindX2, Flm);

        double dError = (lastError - meanSquaredCombinedErr)/meanSquaredCombinedErr;

        if((dError >= 0 && dError < minErrorChange) || abs(lastFeatureCount - goodCombindX1.size()) <= minFeatureChange) stableSolutions++;   //abs(lastFeatureCount - goodCombindX1.size()) < 4
        else stableSolutions = 0;

        lastFeatureCount = goodCombindX1.size();

        if (LOG_DEBUG) std::cout <<"-- Error: " << meanSquaredCombinedErr << ", rel. error change: " << dError << ", stable solutions: " << stableSolutions << std::endl;

        lastError = meanSquaredCombinedErr;

        iterations++;

    } while(stableSolutions < 3 && iterations < NUMERICAL_OPTIMIZATION_MAX_ITERATIONS);

    return Flm;
}


