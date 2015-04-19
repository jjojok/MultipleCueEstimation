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
                if(LOG_DEBUG) std::cout << "Estimation: " << it->name << " = " << std::endl << it->getF() << std::endl;
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

        double error3;
        int cnt;
        if(F.data) {
            if(LOG_DEBUG) std::cout << "Refined F = " << std::endl << F << std::endl;
            if (compareWithGroundTruth) {
                //meanSquaredRSSTError = randomSampleSymmeticTransferError(Fgt, F, image_1_color, image_2_color, NUM_SAMPLES_F_COMARATION);
                meanSquaredRSSTError = -2;
                double error2 = squaredError(Fgt, F);
                if(LOG_DEBUG) std::cout << "Random sample epipolar error: " << meanSquaredRSSTError << ", Squated distance: " << error2 << std::endl;
                meanSampsonFDistanceGoodMatches(Fgt, F, x1combined, x2combined, error3, cnt);
            }
            if(VISUAL_DEBUG) {
                //rectify(x1, x2, it->getF(), image_1, image_2, "Rectified "+it->name);
                drawEpipolarLines(x1,x2, F, image_1, image_2, "Refined F");
            }
        }

        if (compareWithGroundTruth) {
            std::cout << "Ground truth = " << std::endl << Fgt << std::endl;
            meanSampsonFDistanceGoodMatches(Fgt, Fgt, x1combined, x2combined, error3, cnt);
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

    double squaredErrorThr = 2.00;

    std::vector<Mat> goodCombindX1;
    std::vector<Mat> goodCombindX2;

    std::vector<fundamentalMatrix*> bestFundMats;
    std::vector<fundamentalMatrix*> fundMats;
    std::vector<fundamentalMatrix*> fundMatsTmp;

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
            errorFunctionCombinedMeanSquared(x1Combined, x2Combined, estimationIter->getF(), estimationIter->meanSquaredCSTError, estimationIter->meanSquaredCSTErrorInliers, squaredErrorThr, estimationIter->meanSquaredCSTErrorStandardDeviation);
            errorFunctionCombinedMean(x1Combined, x2Combined, estimationIter->getF(), estimationIter->meanCSTError, estimationIter->meanCSTErrorInliers, std::sqrt(squaredErrorThr), estimationIter->meanCSTErrorStandardDeviation);
            if (LOG_DEBUG) std::cout << "Computing mean squared error of combined matches for " << estimationIter->name << ": " << estimationIter->meanSquaredCSTError << " Std. dev: " << estimationIter->meanSquaredCSTErrorStandardDeviation << ", inliers: " << estimationIter->meanSquaredCSTErrorInliers << std::endl;
            if (LOG_DEBUG) std::cout << "Computing mean error of combined matches for " << estimationIter->name << ": " << estimationIter->meanCSTError << " Std. dev: " << estimationIter->meanCSTErrorStandardDeviation << ", inliers: " << estimationIter->meanCSTErrorInliers << std::endl;
            if(bestMethod->meanSquaredCSTErrorInliers < estimationIter->meanSquaredCSTErrorInliers) {
                bestMethod = estimationIter;
            }

            fundamentalMatrix* fm = new fundamentalMatrix;
            fm->inlier = estimationIter->meanSquaredCSTErrorInliers;
            fm->inlierMeanSquaredErrror = 0;
            fm->inlierStdDeviation = 0;
            fm->selectedInlierCount = 0;
            fm->meanSquaredErrror = estimationIter->meanSquaredCSTError;
            fm->stdDeviation = estimationIter->meanSquaredCSTErrorStandardDeviation;
            fm->F = estimationIter->getF().clone();
            fm->name = estimationIter->name;
            fundMats.push_back(fm);
            if (LOG_DEBUG) std::cout << "Added to vector of fundamental matrices" << std::endl;

            numValues += estimationIter->getFeaturesImg1().size();
        }
    }

    if(numValues < 8) {
        if (LOG_DEBUG) std::cout << "-- Not enough features!" << std::endl;
        return Mat();
    }

    refinedF = bestMethod->getF().clone();

    if (LOG_DEBUG) std::cout << "-- Starting point for optimization: " << bestMethod->name << std::endl;

    int inliers = 0;
    double stdDeviation = 0;
    double error = 0;
    int  iterations = 1;
    double bestError = 999999999;
    double errorThr2 = 3.0;

    std::vector<Point2d> x1CombinedTmp;
    std::vector<Point2d> x2CombinedTmp;

    matToPoint(x1Combined, x1CombinedTmp);
    matToPoint(x2Combined, x2CombinedTmp);

    bestError = 0;
    double errorMeasure;
    int debugUsed;
    double debugErr;
    double smallestTrueErr = 1e50;

    int remove = ((int)std::ceil(x1Combined.size()*0.01));

    do {
        if(LOG_DEBUG) std::cout << "-- Iteration: " << iterations << ", refined number of matches: " << x1CombinedTmp.size() << std::endl;


        fundamentalMatrix* fm = new fundamentalMatrix;

        //fm->F = findFundamentalMat(x1CombinedTmp, x2CombinedTmp);
        fm->F = findFundamentalMat(x1CombinedTmp, x2CombinedTmp, noArray(), FM_RANSAC, squaredErrorThr, 0.99999);

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
            fm->selectedInlierCount = 0;
            char buffer[10];
            std::sprintf(buffer, "Iter_%i", iterations);
            fm->name = std::string(buffer);

            std::vector<correspSubsetError> errors;

            for(int i = 0; i < x1CombinedTmp.size(); i++) {
                correspSubsetError err;
                err.correspIdx = i;
                err.correspError = errorFunctionFPointsSquared(fm->F, matVector(x1CombinedTmp.at(i)), matVector(x2CombinedTmp.at(i)));
                errors.push_back(err);
            }

            std::sort(errors.begin(), errors.end(), compareCorrespErrors);

//            for(int i = 0; i < errors.size(); i++) {      //remove inliers d²< 0.1
//                if(errors.at(i).correspError < 0.0001) remove = i;
//                else break;
//            }

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
            if (LOG_DEBUG) std::cout << "-- Removed inliers: " << remove <<  " inlierMeanError: " << fm->inlierMeanSquaredErrror << ", inlierStdDev: " << fm->inlierStdDeviation << " inlier: " << fm->inlier << ", error measure: " << errorMeasure<< std::endl;

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

            fundMats.push_back(fm);

        }

        iterations++;

    } while(x1CombinedTmp.size() > 2*remove);

    std::sort(fundMats.begin(), fundMats.end(), compareFundMatSets);

    if(compareWithGroundTruth && LOG_DEBUG) std::cout << "-- smallest error to ground truth: " << smallestTrueErr << std::endl;

    std::cout << std::endl;

    double currentIterMeanErr = 0;
    double currentIterStdDev = 0;
    fundMatsTmp.clear();
    for(int i = 0; i < fundMats.size(); i++) {
        fundamentalMatrix* fm = fundMats.at(i);
        if(i > 2 && (fm->meanSquaredErrror > 10*(currentIterMeanErr/i) || (fm->meanSquaredErrror > 10*(currentIterStdDev/i)))) {
            if (LOG_DEBUG) {
                std::cout << "-- Current iteration max mean error: " << 10*(currentIterMeanErr/iterations) << ", max std. dev.: " << 10*(currentIterStdDev/iterations) << std::endl;
                std::cout << "-- Current iteration fundmat mean error: " << fm->meanSquaredErrror << ", std. dev.: " << fm->stdDeviation << std::endl;
                if(compareWithGroundTruth) {
                    meanSampsonFDistanceGoodMatches(Fgt, fm->F, x1Combined, x2Combined, debugErr, debugUsed);
                }
                std::cout << "-- Filtering fundmat " << fm->name << " (" << i << ")" << std::endl;
                //std::cout << std::endl;
            }
        } else {
            fundMatsTmp.push_back(fm);
            if(i < 40) {
                currentIterMeanErr += fm->meanSquaredErrror;
                currentIterStdDev += fm->stdDeviation;
            }
        }
    }
    fundMats = fundMatsTmp;

    if (LOG_DEBUG) std::cout << std::endl;

    double leastMeanErrormeasure = 0;
    double meanErrormeasure = 0;
    double meanMean = 0;
    double leastMeanMean = 0;
    double leastMeanStdDev = 0;
    double meanStdDev = 0;
    double errormeasures = fundMats.size();//fundMats.size()*0.4;
    int leastInliers = fundMats.at(errormeasures-1)->inlier;

    for(int i = 0; i < errormeasures; i++) {
        fundamentalMatrix* fm = fundMats.at(i);

        std::vector<double> squaredErrorVect = computeCombinedSquaredErrorVect(x1Combined, x2Combined, fm->F);
        std::sort(squaredErrorVect.begin(), squaredErrorVect.end());
        fm->leastInlierMeanSquaredErrror = 0;
        for(int j = 0; j < leastInliers; j++) {
            fm->leastInlierMeanSquaredErrror += squaredErrorVect.at(j);
        }
        fm->leastInlierMeanSquaredErrror/=leastInliers;

        fm->leastInlierStdDeviation = 0;
        for(int j = 0; j < leastInliers; j++) {
            fm->leastInlierStdDeviation += std::pow(fm->leastInlierMeanSquaredErrror - squaredErrorVect.at(j),2);
        }
        fm->leastInlierStdDeviation = std::sqrt(fm->leastInlierStdDeviation/leastInliers);

        leastMeanErrormeasure += fnorm(fm->leastInlierStdDeviation, fm->leastInlierMeanSquaredErrror);
        meanErrormeasure += fnorm(fm->stdDeviation, fm->meanSquaredErrror);
        leastMeanMean += fm->leastInlierMeanSquaredErrror;
        meanMean += fm->meanSquaredErrror;
        leastMeanStdDev += fm->leastInlierStdDeviation;
        meanStdDev += fm->stdDeviation;

        if (LOG_DEBUG) {
            std::cout << "-- FundMat " << i << ", name: "<< fm->name << ", inlier: " << fm->inlier << ", inlier error: " << fm->inlierMeanSquaredErrror << ", inlier std. dev.: " << fm->inlierStdDeviation << ", error: " << fm->meanSquaredErrror << ", std. dev.: " << fm->stdDeviation << std::endl;
            std::cout << "-- leastInlierMeanSquaredErrror: " << fm->leastInlierMeanSquaredErrror << ", leastInlierStdDeviation: " << fm->leastInlierStdDeviation << ", least error measure: " << fnorm(fm->leastInlierStdDeviation, fm->leastInlierMeanSquaredErrror) << ", inlier errormeasure: " << fnorm(fm->inlierMeanSquaredErrror, fm->inlierStdDeviation) << ", errormeasure: " << fnorm(fm->meanSquaredErrror, fm->stdDeviation) << std::endl;
            if(compareWithGroundTruth) {
                meanSampsonFDistanceGoodMatches(Fgt, fm->F, x1Combined, x2Combined, debugErr, debugUsed);
            }
            std::cout << std::endl;
        }
    }

    meanErrormeasure/=errormeasures;
    meanMean/=errormeasures;
    meanStdDev/=errormeasures;
    leastMeanErrormeasure/=errormeasures;
    leastMeanMean/=errormeasures;
    leastMeanStdDev/=errormeasures;

    if (LOG_DEBUG) std::cout << std::endl<< std::endl << "-- Mean Error measure: " << meanErrormeasure << ", mean mean: " << meanMean << ", mean std. dev.: " << meanStdDev << std::endl;
    if (LOG_DEBUG) std::cout << "-- Least mean Error measure: " << leastMeanErrormeasure << ", least mean mean: " << leastMeanMean << ", least mean std. dev.: " << leastMeanStdDev << std::endl;

    fundamentalMatrix* bestfm;

    for(int i = 0; i < errormeasures; i++) {
        fundamentalMatrix* fm = fundMats.at(i);

        //if(fnorm(fm->leastInlierMeanSquaredErrror, fm->leastInlierStdDeviation) < leastMeanErrormeasure && fnorm(fm->meanSquaredErrror, fm->stdDeviation) < meanErrormeasure) {
        //if(fm->meanSquaredErrror < 0.9*meanMean && fm->stdDeviation < 0.9*meanStdDev && fm->meanSquaredErrror > 0.5*meanMean && fm->stdDeviation > 0.5*meanStdDev) {       //Filter bad solutions an the ones that are "to good to be true"

        if(fm->leastInlierMeanSquaredErrror < leastMeanMean && fm->leastInlierStdDeviation < leastMeanStdDev && fm->meanSquaredErrror < meanMean && fm->stdDeviation < meanStdDev && fm->leastInlierMeanSquaredErrror > 0.1*leastMeanMean && fm->leastInlierStdDeviation > 0.1*leastMeanStdDev && fm->meanSquaredErrror > 0.1*meanMean && fm->stdDeviation > 0.1*meanStdDev) {
            bestFundMats.push_back(fm);
            //bestFm = fundMats.at(i);
            //break;
        } else {
            if (LOG_DEBUG) {
                std::cout << "-- Current iteration max mean error: " << meanMean << ", max std. dev.: " << meanStdDev << ", least mean error: " << leastMeanMean << ", least max std. dev.: " << leastMeanStdDev << std::endl;
                std::cout << "-- Current iteration fundmat mean error: " << fm->meanSquaredErrror << ", std. dev.: " << fm->stdDeviation << ", least mean error: " << fm->leastInlierMeanSquaredErrror << ", least std. dev.: " << fm->leastInlierStdDeviation << std::endl;
                if(compareWithGroundTruth) {
                    meanSampsonFDistanceGoodMatches(Fgt, fm->F, x1Combined, x2Combined, debugErr, debugUsed);
                }
                std::cout << "-- Filtering fundmat " << fm->name << " (" << i << ")" << std::endl;
                //std::cout << std::endl;
            }
        }
    }

    bestfm = bestFundMats.at(0);

    if(LOG_DEBUG) std::cout << "-- Curresnt best FundMat: " << bestfm->name << ", inlier: " << bestfm->inlier << ", inlier error: " << bestfm->inlierMeanSquaredErrror << ", inlier std. dev.: " << bestfm->inlierStdDeviation << ", error: " << bestfm->meanSquaredErrror << ", std. dev.: " << bestfm->stdDeviation << std::endl;
    if(compareWithGroundTruth) {
        meanSampsonFDistanceGoodMatches(Fgt, bestFundMats.at(0)->F, x1Combined, x2Combined, debugErr, debugUsed);
    }
//    drawEpipolarLines(x1, x1, bestFundMats.at(0)->F, image_1_color, image_2_color, "intermedeate best");
//    findGoodCombinedMatches(x1Combined, x2Combined, goodCombindX1, goodCombindX2, bestFundMats.at(0)->F, squaredErrorThr);
//    visualizePointMatches(image_1_color, image_2_color, goodCombindX1, goodCombindX2, 3, true, "intermedeate best");

    if (LOG_DEBUG) std::cout << "-- Computing inlier matrix... " << std::endl;
    Mat inlierMatrix = Mat::zeros(x1Combined.size(), bestFundMats.size(), CV_8UC1);

    for(int i = 0; i < x1Combined.size(); i++) {
        Mat x1 = x1Combined.at(i);
        Mat x2 = x2Combined.at(i);
        for(int j = 0; j < bestFundMats.size(); j++) {
            fundamentalMatrix* fm = bestFundMats.at(j);
            double d = errorFunctionFPointsSquared(fm->F,  x1, x2);
            if(d < squaredErrorThr) {  //leastMeanMean squaredErrorThr
                inlierMatrix.at<u_int8_t>(i, j) = 1;
            }
        }
    }

    double meanInlierCount = 0;

    for(int i = 0; i < x1Combined.size(); i++) {
        Mat row = inlierMatrix.row(i);
        int cnt = sum(row)[0];
        meanInlierCount += cnt;
    }
    meanInlierCount/=x1Combined.size();

    if (LOG_DEBUG) std::cout << "-- Computing mean inlier count per sample: " << meanInlierCount << std::endl;

    std::vector<Mat> x1CombinedSelection, x2CombinedSelection;
    std::vector<Point2d> x1CombinedSelectionP, x2CombinedSelectionP;

    for(int i = 0; i < x1Combined.size(); i++) {
        Mat row = inlierMatrix.row(i);
        int cnt = sum(row)[0];
        if(cnt > meanInlierCount) {
            x1CombinedSelection.push_back(x1Combined.at(i));
            x2CombinedSelection.push_back(x2Combined.at(i));
            for(int j = 0; j < row.cols; j++) {
                if(row.at<uint8_t>(0,j) == 1) {
                    bestFundMats.at(j)->selectedInlierCount++;
                }
            }
        }
    }

    if (LOG_DEBUG) std::cout << "-- Good selected points: " << x1CombinedSelection.size() << std::endl;

    if (LOG_DEBUG) std::cout << std::endl << std::endl << "-- Finding F with discribes the most points above mean inlier count..." << std::endl << std::endl << std::endl;

    std::sort(bestFundMats.begin(), bestFundMats.end(), compareFundMatSetsSelectedInliers);

    double meanSelectedInlierCount = 0;
    for(int i = 0; i < bestFundMats.size(); i++) {
        fundamentalMatrix* fm = bestFundMats.at(i);
        if (LOG_DEBUG) {
            std::cout << "-- FundMat " << i << ", name: "<< fm->name << ", inlier: " << fm->inlier << ", inlier error: " << fm->inlierMeanSquaredErrror << ", inlier std. dev.: " << fm->inlierStdDeviation << ", error: " << fm->meanSquaredErrror << ", std. dev.: " << fm->stdDeviation << std::endl;
            std::cout << "-- leastInlierMeanSquaredErrror: " << fm->leastInlierMeanSquaredErrror << ", leastInlierStdDeviation: " << fm->leastInlierStdDeviation << ", least error measure: " << fnorm(fm->leastInlierStdDeviation, fm->leastInlierMeanSquaredErrror) << ", inlier errormeasure: " << fnorm(fm->inlierMeanSquaredErrror, fm->inlierStdDeviation) << ", errormeasure: " << fnorm(fm->meanSquaredErrror, fm->stdDeviation) << std::endl;
            std::cout << "-- Selected inlier count: " << fm->selectedInlierCount << std::endl;
            if(compareWithGroundTruth) {
                meanSampsonFDistanceGoodMatches(Fgt, fm->F, x1Combined, x2Combined, debugErr, debugUsed);
            }
            std::cout << std::endl;
        }
        meanSelectedInlierCount+=fm->selectedInlierCount;

    }
    meanSelectedInlierCount/=bestFundMats.size();


    if (LOG_DEBUG) std::cout << std::endl << std::endl << "-- Filtering all solutions with selected inlier below mean selected inlier count: " << meanSelectedInlierCount << std::endl << std::endl;

    fundMatsTmp.clear();
    for(int i = 0; i < bestFundMats.size(); i++) {
        fundamentalMatrix* fm = bestFundMats.at(i);
        if(fm->selectedInlierCount > meanSelectedInlierCount) {
            if (LOG_DEBUG) {
                std::cout << "-- FundMat " << i << ", name: "<< fm->name << ", inlier: " << fm->inlier << ", inlier error: " << fm->inlierMeanSquaredErrror << ", inlier std. dev.: " << fm->inlierStdDeviation << ", error: " << fm->meanSquaredErrror << ", std. dev.: " << fm->stdDeviation << std::endl;
                std::cout << "-- leastInlierMeanSquaredErrror: " << fm->leastInlierMeanSquaredErrror << ", leastInlierStdDeviation: " << fm->leastInlierStdDeviation << ", least error measure: " << fnorm(fm->leastInlierStdDeviation, fm->leastInlierMeanSquaredErrror) << ", inlier errormeasure: " << fnorm(fm->inlierMeanSquaredErrror, fm->inlierStdDeviation) << ", errormeasure: " << fnorm(fm->meanSquaredErrror, fm->stdDeviation) << std::endl;
                std::cout << "-- Selected inlier count: " << fm->selectedInlierCount << std::endl;
                if(compareWithGroundTruth) {
                    meanSampsonFDistanceGoodMatches(Fgt, fm->F, x1Combined, x2Combined, debugErr, debugUsed);
                }
                std::cout << std::endl;
            }
            fundMatsTmp.push_back(fm);
        }
    }
    bestFundMats = fundMatsTmp;

    matToPoint(x1CombinedSelection, x1CombinedSelectionP);
    matToPoint(x2CombinedSelection, x2CombinedSelectionP);

    Mat FCombinedSelection = findFundamentalMat(x1CombinedSelectionP, x2CombinedSelectionP, noArray(), FM_RANSAC, squaredErrorThr, 0.9995);
    if(VISUAL_DEBUG) drawEpipolarLines(x1, x1, FCombinedSelection, image_1_color, image_2_color, "FCombinedSelection");
    if(VISUAL_DEBUG) findGoodCombinedMatches(x1CombinedSelection, x1CombinedSelection, goodCombindX1, goodCombindX2, FCombinedSelection, squaredErrorThr);
    if(VISUAL_DEBUG) visualizePointMatches(image_1_color, image_2_color, goodCombindX1, goodCombindX2, 3, true, "FCombinedSelection used points");
    fundamentalMatrix* fmCombinedSelection = new fundamentalMatrix;
    fmCombinedSelection->F = FCombinedSelection;
    fmCombinedSelection->name = "FCombinedSelection";
    fmCombinedSelection->inlierMeanSquaredErrror = 0;
    fmCombinedSelection->inlierStdDeviation = 0;
    fmCombinedSelection->leastInlierMeanSquaredErrror = 0;
    fmCombinedSelection->leastInlierStdDeviation = 0;
    fmCombinedSelection->selectedInlierCount = 0;
    errorFunctionCombinedMeanSquared(x1Combined, x2Combined, fmCombinedSelection->F, fmCombinedSelection->meanSquaredErrror, fmCombinedSelection->inlier, squaredErrorThr, fmCombinedSelection->stdDeviation);
    bestFundMats.push_back(fmCombinedSelection);
    if (LOG_DEBUG) std::cout << "-- Computed FCombinedSelection:" << std::endl;
    if (LOG_DEBUG) std::cout << "-- meanSquaredErrror: " << fmCombinedSelection->meanSquaredErrror << ", stdDeviation: " << fmCombinedSelection->stdDeviation << std::endl;
    if(compareWithGroundTruth) {
        meanSampsonFDistanceGoodMatches(Fgt, fmCombinedSelection->F, x1Combined, x2Combined, debugErr, debugUsed);
    }
    if (LOG_DEBUG) std::cout << std::endl;

    std::sort(bestFundMats.begin(), bestFundMats.end(), compareFundMatSetsError);

    if (LOG_DEBUG) {
        for(int i = 0; i < bestFundMats.size(); i++) {
            fundamentalMatrix* fm = bestFundMats.at(i);
            std::cout << "-- FundMat " << i << ", name: "<< fm->name << ", inlier: " << fm->inlier << ", inlier error: " << fm->inlierMeanSquaredErrror << ", inlier std. dev.: " << fm->inlierStdDeviation << ", error: " << fm->meanSquaredErrror << ", std. dev.: " << fm->stdDeviation << std::endl;
            std::cout << "-- leastInlierMeanSquaredErrror: " << fm->leastInlierMeanSquaredErrror << ", leastInlierStdDeviation: " << fm->leastInlierStdDeviation << ", least error measure: " << fnorm(fm->leastInlierStdDeviation, fm->leastInlierMeanSquaredErrror) << ", inlier errormeasure: " << fnorm(fm->inlierMeanSquaredErrror, fm->inlierStdDeviation) << ", errormeasure: " << fnorm(fm->meanSquaredErrror, fm->stdDeviation) << std::endl;
            std::cout << "-- Selected inlier count: " << fm->selectedInlierCount << std::endl;
            if(compareWithGroundTruth) {
                meanSampsonFDistanceGoodMatches(Fgt, fm->F, x1Combined, x2Combined, debugErr, debugUsed);
            }
            std::cout << std::endl;
        }
    }

    bestfm = bestFundMats.at(0);
    refinedF = bestFundMats.at(0)->F;

    if(LOG_DEBUG) std::cout << "-- Best error: " << bestfm->meanSquaredErrror << ", std. dev.: " << bestfm->stdDeviation << ", inliers: " << bestfm->inlier << std::endl;
    if(compareWithGroundTruth) meanSampsonFDistanceGoodMatches(Fgt, refinedF, x1Combined, x2Combined, debugErr, debugUsed);

    findGoodCombinedMatches(x1Combined, x2Combined, goodCombindX1, goodCombindX2, refinedF, squaredErrorThr);

    errorFunctionCombinedMeanSquared(goodCombindX1, goodCombindX1, refinedF, errorThr2, inliers, squaredErrorThr, stdDeviation);

    if(VISUAL_DEBUG) drawEpipolarLines(x1, x2, refinedF, image_1_color, image_2_color, "RefinedF bevore LM");
    if(VISUAL_DEBUG) visualizePointMatches(image_1_color, image_2_color, goodCombindX1, goodCombindX2, 3, true, "RefinedF bevore LM used points");

    //levenbergMarquardt(refinedF, x1Combined, x2Combined, goodCombindX1, goodCombindX2, squaredErrorThr, 10, 0.05, 0.0, 0.9, inliers, 3, 3000, 0.0, stdDeviation, error);

    if(VISUAL_DEBUG) visualizePointMatches(image_1_color, image_2_color, goodCombindX1, goodCombindX2, 3, true, "Refined F used matches");

    double generalError;
    int generalInliers;
    errorFunctionCombinedMeanSquared(x1Combined, x2Combined, refinedF, meanSquaredCombinedError, this->inlier, squaredErrorThr, stdDeviation);
    if (LOG_DEBUG) std::cout << "Computing mean squared error of combined matches for refined F: " << generalError << ", Std. dev: " << stdDeviation << ", inliers: " << generalInliers << std::endl;
    errorFunctionCombinedMean(x1Combined, x2Combined, refinedF, generalError, generalInliers, sqrt(squaredErrorThr), stdDeviation);
    if (LOG_DEBUG) std::cout << "Computing mean error of combined matches for refined F: " << generalError << ", Std. dev: " << stdDeviation << ", inliers: " << generalInliers << std::endl;

    debugCombinedMatches = x1Combined.size();
    if(compareWithGroundTruth) meanSampsonFDistanceGoodMatches(Fgt, refinedF, x1Combined, x2Combined, debugRefinedFGoodMatchedError, debugRefinedFGoodMatches);

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

int MultipleCueEstimation::getInlier() {
    return inlier;
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

        findGoodCombinedMatches(x1, x2, goodCombindX1, goodCombindX2, Flm, errorThr);

        featureChange = featureChange - goodCombindX1.size();

        if (LOG_DEBUG) std::cout << "-- Refinement Iteration " << iterations << "/" << maxIterations << ", Refined feature count: " << goodCombindX1.size() << "/" << x1.size() << ", feature change: " << featureChange << ", error threshold: " << errorThr << ", max Error: " << maxError << std::endl;

        GeneralFunctor functor;
        functor.x1 = goodCombindX1;
        functor.x2 = goodCombindX2;
        functor.inlierThr = lmErrorThr;
        Eigen::NumericalDiff<GeneralFunctor> numDiff(functor, 1.0e-6); //epsilon
        Eigen::LevenbergMarquardt<Eigen::NumericalDiff<GeneralFunctor>,double> lm(numDiff);
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

