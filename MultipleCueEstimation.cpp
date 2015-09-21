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

//        std::vector<Mat> x1combined, x2combined;

//        for(std::vector<FEstimationMethod>::iterator estimationIter = estimations.begin(); estimationIter != estimations.end(); ++estimationIter) {
//            for(int i = 0; i < estimationIter->getFeaturesImg1().size(); i++) {
//                x1combined.push_back(estimationIter->getFeaturesImg1().at(i));
//                x2combined.push_back(estimationIter->getFeaturesImg2().at(i));
//            }
//        }

        //double error = epipolarSADError(Fgt, x1, x2);

        combinePointCorrespondecies();
        debugCombinedMatches = x1Combined.size();
        F = refineF(estimations);

        double error, stdDev;
        int inlier;

        std::vector<Mat> x1goodPointsMat;
        std::vector<Mat> x2goodPointsMat;
        std::vector<Point2d> x1goodPoints;
        std::vector<Point2d> x2goodPoints;

        if(compareWithGroundTruth) {
            findGoodCombinedMatches(x1Combined, x2Combined, x1goodPointsMat, x2goodPointsMat, Fgt, 1.0);
            matToPoint(x1goodPointsMat, x1goodPoints);
            matToPoint(x2goodPointsMat, x2goodPoints);
        }

        for (std::vector<FEstimationMethod>::iterator it = estimations.begin() ; it != estimations.end(); ++it) {
            if(it->isSuccessful()) {
                errorFunctionCombinedMeanSquared(x1Combined, x2Combined, it->getF(), it->meanSquaredCSTError, it->inlier, 3.0, it->meanSquaredCSTErrorStandardDeviation);
                if (LOG_DEBUG) std::cout << "Mean squared error: " << it->meanSquaredCSTError << " Std. dev: " << it->meanSquaredCSTErrorStandardDeviation << ", inlier: " << it->inlier << std::endl;
                if (LOG_DEBUG) std::cout << "Mean squared selected point error: " << it->meanSquaredCSTErrorSelectedInlier << ", inlier: " << it->selectedInlier << std::endl;
                if(LOG_DEBUG) std::cout << "Estimation: " << it->name << " = " << std::endl << it->getF() << std::endl;
                if (compareWithGroundTruth) {
                    //it->meanSquaredRSSTError = randomSampleSymmeticTransferError(Fgt, it->getF(), image_1_color, image_2_color, NUM_SAMPLES_F_COMARATION);
                    it->meanSquaredRSSTError = -2;
                    double error2 = squaredError(Fgt, it->getF());
                    if(LOG_DEBUG) std::cout << "Random sample epipolar error: " << it->meanSquaredRSSTError << ", Squated distance: " << error2 << ", Mean squared symmetric tranfer error: " << it->getError() << std::endl;
                    meanSampsonFDistanceGoodMatches(Fgt, it->getF(), x1Combined, x2Combined, it->meanSampsonDistanceGoodPointMatches, it->goodPointMatchesCount);
                    if(VISUAL_DEBUG) drawEpipolarLines(x1goodPoints, x2goodPoints, it->getF(), image_1, image_2, it->name);
                } else {
                    if(VISUAL_DEBUG) {
                        //rectify(x1, x2, it->getF(), image_1, image_2, "Rectified "+it->name);
                        drawEpipolarLines(x1,x2, it->getF(), image_1, image_2, it->name);
                    }
                }
            }
        }

        double error3;
        int cnt;
        if(F.data) {
            if(LOG_DEBUG) std::cout << "Refined F = " << std::endl << F << std::endl;
            errorFunctionCombinedMeanSquared(x1Combined, x2Combined, F, meanSquaredCombinedError, this->inlier, 3.0, stdDev);
            if (LOG_DEBUG) std::cout << "Mean squared error: " << meanSquaredCombinedError << " Std. dev: " << stdDev << ", inlier: " << this->inlier << std::endl;
            if (compareWithGroundTruth) {
                //meanSquaredRSSTError = randomSampleSymmeticTransferError(Fgt, F, image_1_color, image_2_color, NUM_SAMPLES_F_COMARATION);
                meanSquaredRSSTError = -2;
                double error2 = squaredError(Fgt, F);
                if(LOG_DEBUG) std::cout << "Random sample epipolar error: " << meanSquaredRSSTError << ", Squated distance: " << error2 << std::endl;
                meanSampsonFDistanceGoodMatches(Fgt, F, x1Combined, x2Combined, debugRefinedFGoodMatchedError, debugRefinedFGoodMatches);
                if(VISUAL_DEBUG) drawEpipolarLines(x1goodPoints, x2goodPoints, F, image_1, image_2, "Refined F");
            } else {
                if(VISUAL_DEBUG) {
                    //rectify(x1, x2, it->getF(), image_1, image_2, "Rectified "+it->name);
                    drawEpipolarLines(x1,x2, F, image_1, image_2, "Refined F");
                }
            }
        }

        if (compareWithGroundTruth) {
            if (LOG_DEBUG) std::cout << "Ground truth = " << std::endl << Fgt << std::endl;
            meanSampsonFDistanceGoodMatches(Fgt, Fgt, x1Combined, x2Combined, error3, cnt);
            if(VISUAL_DEBUG) {
                //rectify(x1, x2, Fgt, image_1, image_2, "Rectified ground truth");
                drawEpipolarLines(x1goodPoints,x2goodPoints, Fgt, image_1, image_2, "F_groundtruth");
            }
            errorFunctionCombinedMeanSquared(x1Combined, x2Combined, Fgt, error, inlier, 3.0, stdDev);
            if (LOG_DEBUG) std::cout << "Mean squared error: " << error << " Std. dev: " << stdDev << ", inlier: " << inlier << std::endl;
//            Mat bestSolution = cv::findFundamentalMat(x1goodPoints, x2goodPoints, FM_RANSAC, 3.0, 0.999);
//            if (LOG_DEBUG) std::cout << "Best possible solution = " << std::endl << bestSolution << std::endl;
//            meanSampsonFDistanceGoodMatches(Fgt, bestSolution, x1Combined, x2Combined, error3, cnt);
//            if(VISUAL_DEBUG) {
//                //rectify(x1, x2, Fgt, image_1, image_2, "Rectified ground truth");
//                drawEpipolarLines(x1goodPoints,x2goodPoints, bestSolution, image_1, image_2, "Best possible solution");
//            }
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

    double squaredErrorThr = 3.00;

    double debugErr, debugStdDev;
    int debugUsed;

    std::vector<Mat> goodCombindX1;
    std::vector<Mat> goodCombindX2;

    //std::vector<fundamentalMatrix*> fundMats;
    std::vector<fundamentalMatrix*> results;

    if(estimations.size() == 0) {
        if (LOG_DEBUG) std::cout << "-- No Fundamental Matrix found!" << std::endl;
        return Mat();
    }

    std::vector<FEstimationMethod>::iterator bestMethod = estimations.begin();
    Mat refinedF = Mat::ones(3,3,CV_64FC1);
    int numValues = 0;
    int est = -1;

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
            fm->inlier = 0;
            fm->inlierMeanSquaredErrror = 0;
            fm->inlierStdDeviation = 0;
            fm->selectedInlierCount = 0;
            fm->meanSquaredErrror = 0;
            fm->stdDeviation = 0;//estimationIter->meanSquaredCSTErrorStandardDeviation;
            fm->F = estimationIter->getF().clone();
            fm->name = estimationIter->name;
            fm->id = est;
            //fm->containedInCluserCnt = 0;
            results.push_back(fm);
            if (LOG_DEBUG) std::cout << "Added to vector of fundamental matrices" << std::endl;

            est--;

            numValues += estimationIter->getFeaturesImg1().size();
        }
    }

    if(numValues < 8) {
        if (LOG_DEBUG) std::cout << "-- Not enough features!" << std::endl;
        return bestMethod->getF().clone();
    }

    std::vector<Mat> x1CombinedSelection, x2CombinedSelection, x1NotSelected, x2NotSelected;
    std::vector<Mat> x1FeatureSet, x2FeatureSet;

    for(int i = 0; i < x1Combined.size(); i++) {
        x1FeatureSet.push_back(x1Combined.at(i));
        x2FeatureSet.push_back(x2Combined.at(i));
    }

    int iteration = 0;
    int oldFeatures = 0;
    int featureChange = 0;

    int maxPoints = std::max(30, ((int)(x1Combined.size()*0.2)));

    do {

        iteration++;

        x1NotSelected.clear();
        x2NotSelected.clear();
        computeSelectedMatches(x1FeatureSet, x2FeatureSet, x1CombinedSelection, x2CombinedSelection, x1NotSelected, x2NotSelected, results, squaredErrorThr);
        if (LOG_DEBUG) std::cout << "-- Iteration: " << iteration << ", computing Fundamental Mat from selected matches, maxPoints: " << maxPoints << std::endl;

        x1FeatureSet = x1NotSelected;
        x2FeatureSet = x2NotSelected;

        featureChange = x1CombinedSelection.size() - oldFeatures;
        oldFeatures = x1CombinedSelection.size();

    } while(x1CombinedSelection.size() < maxPoints && featureChange > 0 && iteration < 100);

    for(int i = 0; i < results.size(); i++) {
        fundamentalMatrix* fm = results.at(i);
        errorFunctionCombinedMeanSquared(x1CombinedSelection, x2CombinedSelection, fm->F, fm->inlierMeanSquaredErrror, fm->selectedInlierCount, squaredErrorThr, fm->inlierStdDeviation);
    }

    //std::sort(results.begin(), results.end(), compareFundMatSetsInlinerError);
    std::sort(results.begin(), results.end(), compareFundMatSetsSelectedInliers);

    for(int i = 0; i < results.size(); i++) {
        fundamentalMatrix* fm = results.at(i);
        if (LOG_DEBUG) std::cout << "-- Computing mean squared error of combined matches (fm "<<fm->id<<","<<fm->name<<"): " << fm->meanSquaredErrror << " Std. dev: " << fm->stdDeviation << ", inliers: " << fm->inlier << std::endl;
        if (LOG_DEBUG) std::cout << "-- Computing mean squared error of selected matches: " << fm->inlierMeanSquaredErrror << " Std. dev: " << fm->inlierStdDeviation << ", inliers: " << fm->selectedInlierCount << std::endl;
        if(compareWithGroundTruth) {
            meanSampsonFDistanceGoodMatches(Fgt, fm->F, x1Combined, x2Combined, debugErr, debugUsed);
        }
    }

    fundamentalMatrix* bestfm = results.at(0);
    refinedF = bestfm->F;

    if(compareWithGroundTruth) {
        correctSelectedPoints = goodMatchesCount(Fgt, x1CombinedSelection, x2CombinedSelection, 10.0);
    }

    errorFunctionCombinedMeanSquared(x1CombinedSelection, x2CombinedSelection, bestfm->F, meanSquaredSelectedError, debugUsed, squaredErrorThr, debugStdDev);

    selectedPoints = x1CombinedSelection.size();
    selectedInlier = bestfm->selectedInlierCount;

    for(int i = 0; i < estimations.size(); i++) {
        if(estimations.at(i).isSuccessful()) {
            errorFunctionCombinedMeanSquared(x1CombinedSelection, x2CombinedSelection, estimations.at(i).getF(), estimations.at(i).meanSquaredCSTErrorSelectedInlier, estimations.at(i).selectedInlier, squaredErrorThr, debugStdDev);
            if(debugUsed > bestfm->selectedInlierCount) {
                if (LOG_DEBUG) std::cout << "-- Found better solution then refined: " << estimations.at(i).name << ", selected inlier: " << estimations.at(i).selectedInlier << std::endl;
                refinedF = estimations.at(i).getF();
                selectedInlier = estimations.at(i).selectedInlier;
                meanSquaredSelectedError = estimations.at(i).meanSquaredCSTErrorSelectedInlier;
            }
        }
    }

    if(LOG_DEBUG) {
        std::cout << "-- Best error: " << bestfm->meanSquaredErrror << ", std. dev.: " << bestfm->stdDeviation << ", inliers: " << bestfm->inlier << std::endl;
        if(compareWithGroundTruth) meanSampsonFDistanceGoodMatches(Fgt, refinedF, x1Combined, x2Combined, debugErr, debugUsed);
    }

    if(VISUAL_DEBUG) findGoodCombinedMatches(x1Combined, x2Combined, goodCombindX1, goodCombindX2, refinedF, squaredErrorThr);

    //errorFunctionCombinedMeanSquared(goodCombindX1, goodCombindX1, refinedF, debugErr, debugUsed, squaredErrorThr, debugStdDev);

//    if(VISUAL_DEBUG) drawEpipolarLines(x1, x2, refinedF, image_1_color, image_2_color, "RefinedF bevore LM");
//    if(VISUAL_DEBUG) visualizePointMatches(image_1_color, image_2_color, goodCombindX1, goodCombindX2, 3, true, "RefinedF bevore LM used points");

    //levenbergMarquardt(refinedF, x1Combined, x2Combined, goodCombindX1, goodCombindX2, squaredErrorThr, 10, 0.05, 0.0, 0.9, inliers, 3, 3000, 0.0, stdDeviation, error);

    if(VISUAL_DEBUG) visualizePointMatches(image_1_color, image_2_color, goodCombindX1, goodCombindX2, 3, true, "Refined F used matches");

    double generalError, stdDeviation;
    int generalInliers;
    errorFunctionCombinedMeanSquared(x1Combined, x2Combined, refinedF, meanSquaredCombinedError, this->inlier, squaredErrorThr, stdDeviation);
    if (LOG_DEBUG) std::cout << "Computing mean squared error of combined matches for refined F: " << meanSquaredCombinedError << ", Std. dev: " << stdDeviation << ", inlier: " << this->inlier << std::endl;
    errorFunctionCombinedMean(x1Combined, x2Combined, refinedF, generalError, generalInliers, sqrt(squaredErrorThr), stdDeviation);
    if (LOG_DEBUG) std::cout << "Computing mean error of combined matches for refined F: " << generalError << ", Std. dev: " << stdDeviation << ", inliers: " << generalInliers << std::endl;

    debugCombinedMatches = x1Combined.size();
    if(compareWithGroundTruth) meanSampsonFDistanceGoodMatches(Fgt, refinedF, x1Combined, x2Combined, debugRefinedFGoodMatchedError, debugRefinedFGoodMatches);

    return refinedF;
}

void MultipleCueEstimation::computeSelectedMatches(std::vector<Mat> x1Current, std::vector<Mat> x2Current, std::vector<Mat> &x1Selected, std::vector<Mat> &x2Selected, std::vector<Mat> &x1NotSelected, std::vector<Mat> &x2NotSelected, std::vector<fundamentalMatrix*> &fundMats, double squaredErrorThr) {
    int  iterations = 1;
    double bestError = 999999999;

    std::vector<Point2d> x1CombinedTmp;
    std::vector<Point2d> x2CombinedTmp;

    matToPoint(x1Current, x1CombinedTmp);
    matToPoint(x2Current, x2CombinedTmp);

    std::vector<fundamentalMatrix*> currentFundMats;

    bestError = 0;
    int debugUsed;
    double debugErr;


    int remove = ((int)std::ceil(x1Current.size()*0.01));
    int minPoints = std::max(30, (int)(x1Combined.size()*0.1));

    do {

        fundamentalMatrix* fm = new fundamentalMatrix;

        fm->F = findFundamentalMat(x1CombinedTmp, x2CombinedTmp, noArray(), FM_LMEDS, squaredErrorThr, 0.9995);

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
            fm->id = fundMats.size();
            char buffer[10];
            std::sprintf(buffer, "Iter_%i", (int)fundMats.size());
            fm->name = std::string(buffer);

            if(LOG_DEBUG) std::cout << "-- Iteration: " << iterations << ", name: " << fm->name << ", refined number of matches: " << x1CombinedTmp.size() << std::endl;

            std::vector<correspSubsetError> errors;

            for(int i = 0; i < x1CombinedTmp.size(); i++) {
                correspSubsetError err;
                err.correspIdx = i;
                err.correspError = errorFunctionFPointsSquared(fm->F, matVector(x1CombinedTmp.at(i)), matVector(x2CombinedTmp.at(i)));
                errors.push_back(err);
            }

            std::sort(errors.begin(), errors.end(), compareCorrespErrors);

            for(int i = 0; i < errors.size(); i++) {      //remove inliers d²< 0.1
                if(errors.at(i).correspError < 0.5) remove = i;
                else break;
            }

            if (LOG_DEBUG) std::cout << "-- Removed inliers: " << remove+1 << std::endl;

            errors.erase(errors.begin()+remove, errors.end());

            for(int i = 0; i < errors.size(); i++) {
                x1CombinedTmp.erase(x1CombinedTmp.begin()+errors.at(i).correspIdx);
                x2CombinedTmp.erase(x2CombinedTmp.begin()+errors.at(i).correspIdx);
            }

            errorFunctionCombinedMeanSquared(x1Combined, x2Combined, fm->F, fm->meanSquaredErrror, fm->inlier, squaredErrorThr, fm->stdDeviation);

//            if(compareWithGroundTruth) {
//                meanSampsonFDistanceGoodMatches(Fgt, fm->F, x1Combined, x2Combined, debugErr, debugUsed);
//            }

            fundMats.push_back(fm);
            currentFundMats.push_back(fm);

            iterations++;
        }
    } while(x1CombinedTmp.size() > minPoints);

    if (LOG_DEBUG) std::cout << "-- Computing inlier matrix... " << std::endl;
    Mat inlierMatrix = Mat::zeros(x1Current.size(), currentFundMats.size(), CV_8UC1);

    int maxInlierCount = 0;
    int inlierCount;

    for(int i = 0; i < x1Current.size(); i++) {
        Mat x1 = x1Current.at(i);
        Mat x2 = x2Current.at(i);
        inlierCount = 0;
        for(int j = 0; j < currentFundMats.size(); j++) {
            fundamentalMatrix* fm = currentFundMats.at(j);
            double d = errorFunctionFPointsSquared(fm->F, x1, x2);
            if(d < 10) {      //squaredErrorThr
                inlierMatrix.at<u_int8_t>(i, j) = 1;
                inlierCount++;
            }
        }
        int cnt = sum(inlierMatrix.row(i))[0];
        if(cnt > maxInlierCount) maxInlierCount = cnt;
    }

    if (LOG_DEBUG) std::cout << "-- Inlier Thr: " << maxInlierCount << std::endl;

    for(int i = 0; i < x1Current.size(); i++) {
        Mat row = inlierMatrix.row(i);
        int cnt = sum(row)[0];
//        if(cnt > meanInlierCount) {
        if(cnt == maxInlierCount) {
            x1Selected.push_back(x1Current.at(i));
            x2Selected.push_back(x2Current.at(i));
        } else {
            x1NotSelected.push_back(x1Current.at(i));
            x2NotSelected.push_back(x2Current.at(i));
        }
    }

    if (LOG_DEBUG) std::cout << "-- Good selected points: " << x1Selected.size() << std::endl;
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


//    do {

//        Eigen::VectorXd x(9);

//        x(0) = Flm.at<double>(0,0);
//        x(1) = Flm.at<double>(0,1);
//        x(2) = Flm.at<double>(0,2);

//        x(3) = Flm.at<double>(1,0);
//        x(4) = Flm.at<double>(1,1);
//        x(5) = Flm.at<double>(1,2);

//        x(6) = Flm.at<double>(2,0);
//        x(7) = Flm.at<double>(2,1);
//        x(8) = Flm.at<double>(2,2);

//        goodCombindX1.clear();
//        goodCombindX2.clear();

//        homogMat(Flm);

////        oldFlm = Flm.clone();

//        findGoodCombinedMatches(x1, x2, goodCombindX1, goodCombindX2, Flm, errThr);

//        featureChange = featureChange - goodCombindX1.size();

//        if (LOG_DEBUG) std::cout << "-- Refinement Iteration " << iterations << "/" << maxIterations << ", Refined feature count: " << goodCombindX1.size() << "/" << x1.size() << ", feature change: " << featureChange << ", error threshold: " << errThr << ", max Error: " << maxError << std::endl;
//        //thr = 3.0;

//        if(goodCombindX1.size() <= 50) {
//            if (LOG_DEBUG) std::cout << "-- Less then 50 matches left, iteration finieshed" << std::endl;
//            break;
//        }

//        GeneralFunctorStandardDeviation functor;
//        functor.x1 = goodCombindX1;
//        functor.x2 = goodCombindX2;
//        functor.inlierThr = lmErrorThr;
//        Eigen::NumericalDiff<GeneralFunctorStandardDeviation> numDiff(functor, 1.0e-6); //epsilon
//        Eigen::LevenbergMarquardt<Eigen::NumericalDiff<GeneralFunctorStandardDeviation>,double> lm(numDiff);
//        lm.parameters.ftol = 1.0e-15;
//        lm.parameters.xtol = 1.0e-15;
//        lm.parameters.maxfev = 40; // Max iterations
//        Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(x);

//        if (LOG_DEBUG) std::cout << "-- LMA Iterations: " << lm.nfev << ", Status: " << status << std::endl;

//        Flm.at<double>(0,0) = x(0);
//        Flm.at<double>(0,1) = x(1);
//        Flm.at<double>(0,2) = x(2);

//        Flm.at<double>(1,0) = x(3);
//        Flm.at<double>(1,1) = x(4);
//        Flm.at<double>(1,2) = x(5);

//        Flm.at<double>(2,0) = x(6);
//        Flm.at<double>(2,1) = x(7);
//        Flm.at<double>(2,2) = x(8);

//        enforceRankTwoConstraint(Flm);

//        //meanSquaredCSTError = computeCombinedMeanSquaredError(estimations, refinedF);

//        errorFunctionCombinedMean(goodCombindX1, goodCombindX2, Flm, error, inliers, errThr, stdDeviation);

////        if(lmErrorThr > 0.8 && ((double)nextIterInliers)/inliers < 0.4) {
////            if(LOG_DEBUG) std::cout <<"-- Inliers droped from " << inliers << " to " << nextIterInliers << ", Iteration stopped" << std::endl;
////            Flm = oldFlm;
////            break;
////        }

////        inliers = nextIterInliers;

//        double dError = (lastError - stdDeviation)/stdDeviation;

//        if(((dError >= 0 && dError < minErrorChange) || abs(featureChange) <= minFeatureChange || stdDeviation <= maxError)) stableSolutions++;   //abs(lastFeatureCount - goodCombindX1.size()) < 4
//        else stableSolutions = 0;

//        if (LOG_DEBUG) std::cout <<"-- Error: " << error << ", standard dev.: " << stdDeviation << ", rel. error change: " << dError << ", inliers: " << inliers << ", stable solutions: " << stableSolutions << "/" << minStableSolutions << std::endl;

//        featureChange = goodCombindX1.size();

//        lastError = stdDeviation;

//        if(errorDecay == 0.0) errThr = errorThr/(iterations);
//        else errThr *= errorDecay;

//        iterations++;

//    } while(stableSolutions < minStableSolutions && iterations < maxIterations);

    errorThr = errThr;
}

