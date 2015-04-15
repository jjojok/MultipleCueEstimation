#include "MultipleCueEstimation.h"

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
            estimations.push_back(*lines);
        }
        if(computations & F_FROM_POINTS_VIA_H) {
            FEstimationMethod* Hpoints = calcFfromHPoints();
            estimations.push_back(*Hpoints);
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
            if(it->isSuccessful()) {
                std::cout << "Estimation: " << it->name << " = " << std::endl << it->getF() << std::endl;
                if (compareWithGroundTruth) {
                    //it->meanSquaredRSSTError = randomSampleSymmeticTransferError(Fgt, it->getF(), image_1_color, image_2_color, NUM_SAMPLES_F_COMARATION);
                    it->meanSquaredRSSTError = -2;
                    double error2 = squaredError(Fgt, it->getF());
                    if(LOG_DEBUG) std::cout << "Random sample epipolar error: " << it->meanSquaredRSSTError << ", Squated distance: " << error2 << ", Mean squared symmetric tranfer error: " << it->getError() << std::endl;
                    meanSampsonFDistanceGoodMatches(Fgt, it->getF(), x1combined, x2combined, it->meanSampsonDistanceGoodPointMatches, it->goodPointMatchesCount);
                }
                if(VISUAL_DEBUG) {
                    //rectify(x1, x2, it->getF(), image_1, image_2, "Rectified "+it->name);
                    drawEpipolarLines(x1,x2, it->getF(), image_1, image_2, it->name);
                }
            }
        }

        if(F.data) {
            std::cout << "Refined F = " << std::endl << F << std::endl;
            if (compareWithGroundTruth) {
                //meanSquaredRSSTError = randomSampleSymmeticTransferError(Fgt, F, image_1_color, image_2_color, NUM_SAMPLES_F_COMARATION);
                meanSquaredRSSTError = -2;
                double error2 = squaredError(Fgt, F);
                if(LOG_DEBUG) std::cout << "Random sample epipolar error: " << meanSquaredRSSTError << ", Squated distance: " << error2 << std::endl;
                double error3;
                int cnt;
                meanSampsonFDistanceGoodMatches(Fgt, F, x1combined, x2combined, error3, cnt);
            }
            if(VISUAL_DEBUG) {
                //rectify(x1, x2, it->getF(), image_1, image_2, "Rectified "+it->name);
                drawEpipolarLines(x1,x2, F, image_1, image_2, "Refined F");
            }
        }

        if (compareWithGroundTruth) {
            std::cout << "Ground truth = " << std::endl << Fgt << std::endl;
            if(VISUAL_DEBUG) {
                //rectify(x1, x2, Fgt, image_1, image_2, "Rectified ground truth");
                drawEpipolarLines(x1,x2, Fgt, image_1, image_2, "F_groundtruth");
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

    std::vector<Mat> goodCombindX1;
    std::vector<Mat> goodCombindX2;

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
        if(estimationIter->isSuccessful()) {
            errorFunctionCombinedMeanSquared(x1Combined, x2Combined, estimationIter->getF(), estimationIter->meanSquaredCSTError, estimationIter->meanSquaredCSTErrorInliers, 3.0, estimationIter->meanSquaredCSTErrorStandardDeviation);
            errorFunctionCombinedMean(x1Combined, x2Combined, estimationIter->getF(), estimationIter->meanCSTError, estimationIter->meanCSTErrorInliers, 3.0, estimationIter->meanCSTErrorStandardDeviation);
            if (LOG_DEBUG) std::cout << "Computing mean squared error of combined matches for " << estimationIter->name << ": " << estimationIter->meanSquaredCSTError << " Std. dev: " << estimationIter->meanSquaredCSTErrorStandardDeviation << ", inliers: " << estimationIter->meanSquaredCSTErrorInliers << std::endl;
            if (LOG_DEBUG) std::cout << "Computing mean error of combined matches for " << estimationIter->name << ": " << estimationIter->meanCSTError << " Std. dev: " << estimationIter->meanCSTErrorStandardDeviation << ", inliers: " << estimationIter->meanCSTErrorInliers << std::endl;
            if(bestMethod->meanSquaredCSTErrorInliers < estimationIter->meanSquaredCSTErrorInliers) {
                bestMethod = estimationIter;
            }
            numValues += estimationIter->getFeaturesImg1().size();
        }
    }

    if(numValues < 8) {
        if (LOG_DEBUG) std::cout << "-- Not enough features!" << std::endl;
        return Mat();
    }

    refinedF = bestMethod->getF().clone();

    if (LOG_DEBUG) std::cout << "-- Starting point for optimization: " << bestMethod->name << std::endl;

    double errorThr = bestMethod->meanSquaredCSTError;
    int inliers = 0;
    int oldInliers = 0;
    int stableSolutions = 0;
    double lmErrorThr = bestMethod->meanSquaredCSTError;
    double stdDeviation = 0;
    double stdDeviationOld = 0;
    double stdDeviationChange = 0;
    double error = 0;
    double errorOld = 0;
    double errorChange = 0;
    int inlierChange = 0;
    bool inliersChaged = false;
    int  iterations = 1;
    int goodSolutions = 0;
    Mat lastGoodF = refinedF.clone();
    double bestError = 999999999;
    double bestStdDeviation = 999999999;
    int goodSolutionCount = 0;
    double errorThr2 = 3.0;

    std::vector<Point2d> x1CombinedTmp;
    std::vector<Point2d> x2CombinedTmp;

    std::vector<fundamentalMatrix> fundMats;

    matToPoint(x1Combined, x1CombinedTmp);
    matToPoint(x2Combined, x2CombinedTmp);

    bestError = 0;
    double errorMeasure;
    double squaredErrorThr = 0.5;
    int debugUsed;
    double debugErr;
    double smallestTrueErr = 1e50;

    int remove = ((int)std::ceil(x1Combined.size()*0.01));

    do {
        if(LOG_DEBUG) std::cout << "-- Iteration: " << iterations << ", refined number of matches: " << x1CombinedTmp.size() << std::endl;


        fundamentalMatrix* fm = new fundamentalMatrix;

        fm->F = findFundamentalMat(x1CombinedTmp, x2CombinedTmp);

        if(!fm->F.data) {
            if (LOG_DEBUG) std::cout << "-- Computed F has no data!" << std::endl;

            for(int i = 0; i < remove; i++) {
                int idx = std::rand()%x1CombinedTmp.size();
                x1CombinedTmp.erase(x1CombinedTmp.begin()+idx);
                x2CombinedTmp.erase(x2CombinedTmp.begin()+idx);
            }

        } else {

            fm->inlier = 0;
            fm->inlierMeanSquaredErrror = 0;
            fm->inlierStdDeviation = 0;
            fm->meanSquaredErrror = 0;
            fm->stdDeviation = 0;

            std::vector<correspSubsetError> errors;

            for(int i = 0; i < x1CombinedTmp.size(); i++) {
                correspSubsetError err;
                err.correspIdx = i;
                err.correspError = errorFunctionFPointsSquared(fm->F, matVector(x1CombinedTmp.at(i)), matVector(x2CombinedTmp.at(i)));
                errors.push_back(err);
            }

            std::sort(errors.begin(), errors.end(), compareCorrespErrors);

            errors.erase(errors.begin()+remove, errors.end());

            for(int i = 0; i < errors.size(); i++) {
                x1CombinedTmp.erase(x1CombinedTmp.begin()+errors.at(i).correspIdx);
                x2CombinedTmp.erase(x2CombinedTmp.begin()+errors.at(i).correspIdx);
            }

            std::vector<double> squaredErrorVect = computeCombinedSquaredErrorVect(x1Combined, x2Combined, fm->F);

            for(int i = 0; i < squaredErrorVect.size(); i++) {
                if(squaredErrorVect.at(i) < squaredErrorThr) {
                    fm->inlierMeanSquaredErrror +=squaredErrorVect.at(i);
                    fm->inlier++;
                }
            }
            fm->inlierMeanSquaredErrror/=fm->inlier;
            for(int i = 0; i < squaredErrorVect.size(); i++) {
                if(squaredErrorVect.at(i) < squaredErrorThr) {
                    fm->inlierStdDeviation += std::pow(squaredErrorVect.at(i) - fm->inlierMeanSquaredErrror, 2);
                }
            }
            fm->inlierStdDeviation = std::sqrt(fm->inlierStdDeviation/(fm->inlier-1));

            errorMeasure = std::sqrt(fm->inlierStdDeviation*fm->inlierStdDeviation + fm->inlierMeanSquaredErrror*fm->inlierMeanSquaredErrror);
            std::cout << "-- inlierMeanError: " << fm->inlierMeanSquaredErrror << ", inlierStdDev: " << fm->inlierStdDeviation << " inlier: " << fm->inlier << ", error measure: " << errorMeasure<< std::endl;

            int generalInliers;
            errorFunctionCombinedMeanSquared(x1Combined, x2Combined, fm->F, fm->meanSquaredErrror, generalInliers, squaredErrorThr, fm->stdDeviation);
            errorMeasure = std::sqrt(fm->stdDeviation*fm->stdDeviation + fm->meanSquaredErrror*fm->meanSquaredErrror);
            if (LOG_DEBUG) {
                std::cout << "-- Computing mean squared error of combined matches for refined F: " << fm->meanSquaredErrror << ", Std. dev: " << fm->stdDeviation << ", inliers: " << generalInliers << ", error measure: " << errorMeasure << std::endl;
                if(compareWithGroundTruth) {
                    meanSampsonFDistanceGoodMatches(Fgt, fm->F, x1Combined, x2Combined, debugErr, debugUsed);
                }
                //std::cout << std::endl;
            }

            if(debugErr < smallestTrueErr) smallestTrueErr = debugErr;

//            if(errorMeasure < bestErrorMeasure) {
//                refinedF = fm->F.clone();
//                bestStdDev = stdDeviation;
//                mostInliers = generalInliers;
//                bestError = generalError;
//                bestIter = iterations;
//                bestErrorMeasure = errorMeasure;
//            }
            //errorFunctionCombinedMean(x1Combined, x2Combined, fm->F, generalError, generalInliers, squaredErrorThr, stdDeviation);
            //if (LOG_DEBUG) std::cout << "-- Computing mean error of combined matches for refined F: " << generalError << ", Std. dev: " << stdDeviation << ", inliers: " << generalInliers << std::endl;

            fundMats.push_back(*fm);

        }

        iterations++;

    } while(x1CombinedTmp.size() > 2*remove);

    std::sort(fundMats.begin(), fundMats.end(), compareFundMatSets);

    if(compareWithGroundTruth && LOG_DEBUG) std::cout << "-- smallest error to ground trouth: " << smallestTrueErr << std::endl;

    std::cout << std::endl;

    double meanErrormeasure = 0;
    double meanMean = 0;
    double meanStdDev = 0;
    double errormeasures = std::min((int)fundMats.size(), 30);

    for(int i = 0; i < errormeasures; i++) {
        fundamentalMatrix fm = fundMats.at(i);
        meanErrormeasure += fnorm(fm.meanSquaredErrror, fm.stdDeviation);
        meanMean += fm.meanSquaredErrror;
        meanStdDev += fm.stdDeviation;
        if (LOG_DEBUG) {
            std::cout << "-- FundMat " << i << ", inlier: " << fm.inlier << ", inlier error: " << fm.inlierMeanSquaredErrror << ", inlier std. dev.: " << fm.inlierStdDeviation << ", error: " << fm.meanSquaredErrror << ", std. dev.: " << fm.stdDeviation << std::endl;
            std::cout << "-- inlier errormeasure: " << fnorm(fm.inlierMeanSquaredErrror, fm.inlierStdDeviation) << ", errormeasure: " << fnorm(fm.meanSquaredErrror, fm.stdDeviation) << std::endl;
            if(compareWithGroundTruth) {
                meanSampsonFDistanceGoodMatches(Fgt, fm.F, x1Combined, x2Combined, debugErr, debugUsed);
            }
            std::cout << std::endl;
        }
    }

    meanErrormeasure/=errormeasures;
    meanMean/=errormeasures;
    meanStdDev/=errormeasures;

    if (LOG_DEBUG) std::cout << "-- Mean Error measure: " << meanErrormeasure << ", mean mean: " << meanMean << ", mean std. dev.: " << meanStdDev << std::endl;

    fundamentalMatrix bestFm;
    double smallesDist = 1e50;
    double dist;
    int distUsed;
    std::vector<fundamentalMatrix> bestFundMats;

    for(int i = 0; i < errormeasures; i++) {
        fundamentalMatrix fm = fundMats.at(i);
        //if(fnorm(fm.meanSquaredErrror, fm.stdDeviation) < meanErrormeasure) {
        if(fm.meanSquaredErrror < meanMean && fm.stdDeviation < meanStdDev) {
            bestFundMats.push_back(fm);
            //bestFm = fundMats.at(i);
            //break;
        }
    }

    bestFm = bestFundMats.at(0);

    std::cout << "-- Curresnt best FundMat, inlier: " << bestFm.inlier << ", inlier error: " << bestFm.inlierMeanSquaredErrror << ", inlier std. dev.: " << bestFm.inlierStdDeviation << ", error: " << bestFm.meanSquaredErrror << ", std. dev.: " << bestFm.stdDeviation << std::endl;
    if(compareWithGroundTruth) {
        meanSampsonFDistanceGoodMatches(Fgt, bestFundMats.at(0).F, x1Combined, x2Combined, debugErr, debugUsed);
    }

    if (LOG_DEBUG) std::cout << "-- Finding F closest to current best solution: " << std::endl;
    for(int i = 0; i < bestFundMats.size(); i++) {
        fundamentalMatrix fm = bestFundMats.at(i);
        meanSampsonFDistanceGoodMatches(bestMethod->getF(), fm.F, x1Combined, x2Combined, dist, distUsed);
        if(dist < smallesDist) {
            smallesDist = dist;
            bestFm = fm;
        }
    }

    refinedF = bestFm.F;

    if(LOG_DEBUG) std::cout << "-- Best error: " << bestFm.meanSquaredErrror << ", std. dev.: " << bestFm.stdDeviation << ", inliers: " << bestFm.inlier << std::endl;
    meanSampsonFDistanceGoodMatches(Fgt, refinedF, x1Combined, x2Combined, debugErr, debugUsed);

    findGoodCombinedMatches(x1Combined, x2Combined, goodCombindX1, goodCombindX2, refinedF, squaredErrorThr);

    errorFunctionCombinedMeanSquared(goodCombindX1, goodCombindX1, refinedF, errorThr2, inliers, squaredErrorThr, stdDeviation);

    drawEpipolarLines(x1, x2, refinedF, image_1_color, image_2_color, "Standard Deviation min imization result");

    visualizePointMatches(image_1_color, image_2_color, goodCombindX1, goodCombindX2, 3, true, "Standard Deviation min imization result");

    double errorThr3 = 2.0;

    levenbergMarquardt(refinedF, x1Combined, x2Combined, goodCombindX1, goodCombindX2, errorThr3, 3, 0.01, 0.0, 1.0, inliers, 3, 300, 0.0, stdDeviation, error);

    visualizePointMatches(image_1_color, image_2_color, goodCombindX1, goodCombindX2, 3, true, "Refined F used matches");

    double generalError;
    int generalInliers;
    errorFunctionCombinedMeanSquared(x1Combined, x2Combined, refinedF, generalError, generalInliers, squaredErrorThr, stdDeviation);
    if (LOG_DEBUG) std::cout << "Computing mean squared error of combined matches for refined F: " << generalError << ", Std. dev: " << stdDeviation << ", inliers: " << generalInliers << std::endl;
    errorFunctionCombinedMean(x1Combined, x2Combined, refinedF, generalError, generalInliers, sqrt(squaredErrorThr), stdDeviation);
    if (LOG_DEBUG) std::cout << "Computing mean error of combined matches for refined F: " << generalError << ", Std. dev: " << stdDeviation << ", inliers: " << generalInliers << std::endl;

    return refinedF;
}

double MultipleCueEstimation::getMeanSquaredCSTError() {
    return meanSquaredCombinedError;
}

double MultipleCueEstimation::getMeanSquaredRSSTError() {
    return meanSquaredRSSTError;
}

std::vector<FEstimationMethod> MultipleCueEstimation::getEstimations() {
    return estimations;
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

void MultipleCueEstimation::levenbergMarquardt(Mat &Flm, std::vector<Mat> x1, std::vector<Mat> x2, std::vector<Mat> &goodCombindX1, std::vector<Mat> &goodCombindX2, double &errorThr, int minFeatureChange, double minErrorChange, double lmErrorThr, double errorDecay, int &inliers, int minStableSolutions, int maxIterations, double maxError, double &stdDeviation, double &error) {
    double lastError = 0;
    int stableSolutions = 0;
    int iterations = 1;
    int featureChange = 0;
    stdDeviation = 0;
    inliers = 0;
    error = 0;
//    int nextIterInliers = 0;
//    Mat oldFlm;

    if (LOG_DEBUG) std::cout << "-- Running Levenberg Marquardt loop, min feature change: " << minFeatureChange << ", min rel. error change: " << minErrorChange << ", threshold for LM error: " << lmErrorThr << std::endl;


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

        goodCombindX1.clear();
        goodCombindX2.clear();

        homogMat(Flm);

//        oldFlm = Flm.clone();

        findGoodCombinedMatches(x1, x2, goodCombindX1, goodCombindX2, Flm, errorThr);

        featureChange = featureChange - goodCombindX1.size();

        if (LOG_DEBUG) std::cout << "-- Refinement Iteration " << iterations << "/" << maxIterations << ", Refined feature count: " << goodCombindX1.size() << "/" << x1.size() << ", feature change: " << featureChange << ", error threshold: " << errorThr << ", max Error: " << maxError << std::endl;
        //thr = 3.0;

//        if(goodCombindX1.size() <= 50) {
//            if (LOG_DEBUG) std::cout << "-- Less then 50 matches left, iteration finieshed" << std::endl;
//            break;
//        }

        GeneralFunctor functor;
        functor.x1 = x1;//goodCombindX1;
        functor.x2 = x2;//goodCombindX2;
        functor.inlierThr = lmErrorThr;
        Eigen::NumericalDiff<GeneralFunctor> numDiff(functor, 1.0e-6); //epsilon
        Eigen::LevenbergMarquardt<Eigen::NumericalDiff<GeneralFunctor>,double> lm(numDiff);
        lm.parameters.ftol = 1.0e-15;
        lm.parameters.xtol = 1.0e-15;
        lm.parameters.maxfev = 2000; // Max iterations
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

        errorFunctionCombinedMean(goodCombindX1, goodCombindX2, Flm, error, inliers, errorThr, stdDeviation);

//        if(lmErrorThr > 0.8 && ((double)nextIterInliers)/inliers < 0.4) {
//            if(LOG_DEBUG) std::cout <<"-- Inliers droped from " << inliers << " to " << nextIterInliers << ", Iteration stopped" << std::endl;
//            Flm = oldFlm;
//            break;
//        }

//        inliers = nextIterInliers;

        double dError = (lastError - error)/error;

        if(((dError >= 0 && dError < minErrorChange) || abs(featureChange) <= minFeatureChange || error <= maxError)) stableSolutions++;   //abs(lastFeatureCount - goodCombindX1.size()) < 4
        else stableSolutions = 0;

        if (LOG_DEBUG) std::cout <<"-- Error: " << error << ", standard dev.: " << stdDeviation << ", rel. error change: " << dError << ", inliers: " << inliers << ", stable solutions: " << stableSolutions << "/" << minStableSolutions << std::endl;

        featureChange = goodCombindX1.size();

        lastError = error;

        if(errorDecay == 0.0) errorThr = errorThr/(iterations);
        else errorThr *= errorDecay;

        iterations++;

    } while(stableSolutions < minStableSolutions && iterations < maxIterations);
}

void MultipleCueEstimation::levenbergMarquardtStandardDeviation(Mat &Flm, std::vector<Mat> x1, std::vector<Mat> x2, std::vector<Mat> &goodCombindX1, std::vector<Mat> &goodCombindX2, double &errorThr, int minFeatureChange, double minErrorChange, double lmErrorThr, double errorDecay, int &inliers, int minStableSolutions, int maxIterations, double maxError, double &stdDeviation, double &error) {
    double lastError = 0;
    int stableSolutions = 0;
    int iterations = 1;
    int featureChange = 0;
    stdDeviation = 0;
    inliers = 0;
    error = 0;
    double errThr = errorThr;
//    int nextIterInliers = 0;
//    Mat oldFlm;

    if (LOG_DEBUG) std::cout << "-- Running Levenberg Marquardt minimizing the standard deviation, min feature change: " << minFeatureChange << ", min rel. error change: " << minErrorChange << ", threshold for LM error: " << lmErrorThr << std::endl;


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

        goodCombindX1.clear();
        goodCombindX2.clear();

        homogMat(Flm);

//        oldFlm = Flm.clone();

        findGoodCombinedMatches(x1, x2, goodCombindX1, goodCombindX2, Flm, errThr);

        featureChange = featureChange - goodCombindX1.size();

        if (LOG_DEBUG) std::cout << "-- Refinement Iteration " << iterations << "/" << maxIterations << ", Refined feature count: " << goodCombindX1.size() << "/" << x1.size() << ", feature change: " << featureChange << ", error threshold: " << errThr << ", max Error: " << maxError << std::endl;
        //thr = 3.0;

        if(goodCombindX1.size() <= 50) {
            if (LOG_DEBUG) std::cout << "-- Less then 50 matches left, iteration finieshed" << std::endl;
            break;
        }

        GeneralFunctorStandardDeviation functor;
        functor.x1 = goodCombindX1;
        functor.x2 = goodCombindX2;
        functor.inlierThr = lmErrorThr;
        Eigen::NumericalDiff<GeneralFunctorStandardDeviation> numDiff(functor, 1.0e-6); //epsilon
        Eigen::LevenbergMarquardt<Eigen::NumericalDiff<GeneralFunctorStandardDeviation>,double> lm(numDiff);
        lm.parameters.ftol = 1.0e-15;
        lm.parameters.xtol = 1.0e-15;
        lm.parameters.maxfev = 40; // Max iterations
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

        errorFunctionCombinedMean(goodCombindX1, goodCombindX2, Flm, error, inliers, errThr, stdDeviation);

//        if(lmErrorThr > 0.8 && ((double)nextIterInliers)/inliers < 0.4) {
//            if(LOG_DEBUG) std::cout <<"-- Inliers droped from " << inliers << " to " << nextIterInliers << ", Iteration stopped" << std::endl;
//            Flm = oldFlm;
//            break;
//        }

//        inliers = nextIterInliers;

        double dError = (lastError - stdDeviation)/stdDeviation;

        if(((dError >= 0 && dError < minErrorChange) || abs(featureChange) <= minFeatureChange || stdDeviation <= maxError)) stableSolutions++;   //abs(lastFeatureCount - goodCombindX1.size()) < 4
        else stableSolutions = 0;

        if (LOG_DEBUG) std::cout <<"-- Error: " << error << ", standard dev.: " << stdDeviation << ", rel. error change: " << dError << ", inliers: " << inliers << ", stable solutions: " << stableSolutions << "/" << minStableSolutions << std::endl;

        featureChange = goodCombindX1.size();

        lastError = stdDeviation;

        if(errorDecay == 0.0) errThr = errorThr/(iterations);
        else errThr *= errorDecay;

        iterations++;

    } while(stableSolutions < minStableSolutions && iterations < maxIterations);

    errorThr = errThr;
}

