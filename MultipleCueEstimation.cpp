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

    //Evaluation data
    combinedFeatures = -1;
    combinedFeaturesCorrect = -1;
    refinedFInlierCombined = -1;
    refinedFTrueInlierCombined = -1;

    refinedFSampsonDistCombined = -1;
    refinedFSampsonDistCorrect = -1;

    refinedFSampsonErrStdDevCombined = -1;

    groundTruthSampsonDistCorrect = -1;

    groundTruthSampsonStdDev = -1;
}

Mat MultipleCueEstimation::compute() {

    if (checkData()) {
        if(computations & F_FROM_POINTS) {      //Calculate F from poiint in general pos.
            FEstimationMethod* points = calcFfromPoints();
            estimations.push_back(*points);
        }
        if(computations & F_FROM_LINES_VIA_H) { //Calculate F from lines via homographies
            FEstimationMethod* lines = calcFfromHLines();
            estimations.push_back(*lines);
        }
        if(computations & F_FROM_POINTS_VIA_H) {//Calculate F from points via homographies
            FEstimationMethod* Hpoints = calcFfromHPoints();
            estimations.push_back(*Hpoints);
        }

        combinePointCorrespondecies();
        //combineAllPointCorrespondecies();
        F = refineF(estimations);

//        double error, stdDev;
//        int inlier;

        std::vector<Mat> x1goodPointsMat;
        std::vector<Mat> x2goodPointsMat;
        std::vector<Point2d> x1goodPoints;
        std::vector<Point2d> x2goodPoints;

        if(compareWithGroundTruth) {
            findGoodCombinedMatches(x1Combined, x2Combined, x1goodPointsMat, x2goodPointsMat, Fgt, INLIER_THRESHOLD);
            matToPoint(x1goodPointsMat, x1goodPoints);
            matToPoint(x2goodPointsMat, x2goodPoints);

            if(CREATE_DEBUG_IMG) visualizePointMatches(image_1_color, image_2_color, x1goodPointsMat, x2goodPointsMat, 20, 2, false, "True Combined Matches");
        }

        for (std::vector<FEstimationMethod>::iterator it = estimations.begin() ; it != estimations.end(); ++it) {
            if(it->isSuccessful()) {
                if(LOG_DEBUG) std::cout << "Estimation: " << it->name << " = " << std::endl << it->getF() << std::endl;
                if (LOG_DEBUG) std::cout << "Mean squared error: " << it->sampsonErrCombined << " Std. dev: " << it->sampsonErrStdDevCombined << ", inlier: " << it->inlierCountCombined  << ", true inlier: " << it->trueInlierCountCombined << std::endl;
                if (compareWithGroundTruth) {
                    errorFunctionCombinedMean(x1goodPointsMat, x2goodPointsMat, it->getF(), it->trueRootSampsonErr, it->trueInlierCountCombined, INLIER_THRESHOLD, it->trueRootSampsonErrStdDev);
                    errorFunctionCombinedMeanSquared(x1goodPointsMat, x2goodPointsMat, it->getF(), it->trueSampsonErr, it->trueInlierCountCombined, INLIER_THRESHOLD, it->trueSampsonErrStdDev);
                    if (LOG_DEBUG) std::cout << "Mean squared ground truth error: " << it->trueSampsonErr << " Std. dev: " << it->trueSampsonErrStdDev << ", true inlier: " << it->trueInlierCountCombined << std::endl;
                    if (LOG_DEBUG) std::cout << "Mean ground truth error: " << it->trueRootSampsonErr << " Std. dev: " << it->trueRootSampsonErrStdDev << ", true inlier: " << it->trueInlierCountCombined << std::endl;
                    //it->trueSampsonErr = meanSampsonFDistanceGoodMatches(Fgt, it->getF(), x1Combined, x2Combined);
                    if(CREATE_DEBUG_IMG) drawEpipolarLines(x1goodPoints, x2goodPoints, it->getF(), image_1_color, image_2_color, it->name);
                } else {
                    if(CREATE_DEBUG_IMG) {
                        //rectify(x1, x2, it->getF(), image_1, image_2, "Rectified "+it->name);
                        drawEpipolarLines(x1,x2, it->getF(), image_1_color, image_2_color, it->name);
                    }
                }
            }
        }

        if(F.data) {
            matToFile("F_result.csv", F);

            if(LOG_DEBUG) std::cout << "Refined F = " << std::endl << F << std::endl;
            errorFunctionCombinedMeanSquared(x1Combined, x2Combined, F, refinedFSampsonDistCombined, refinedFInlierCombined, INLIER_THRESHOLD, refinedFSampsonErrStdDevCombined);
            if (LOG_DEBUG) std::cout << "Mean squared error: " << refinedFSampsonDistCombined << " Std. dev: " << refinedFSampsonErrStdDevCombined << ", inlier: " << refinedFInlierCombined << ", true inlier: " << refinedFInlierCombined << std::endl;
            if (compareWithGroundTruth) {
                //refinedFSampsonDistCorrect = meanSampsonFDistanceGoodMatches(Fgt, F, x1Combined, x2Combined);
                errorFunctionCombinedMean(x1goodPointsMat, x2goodPointsMat, F, refinedFRootDistCorrect, refinedFTrueInlierCombined, INLIER_THRESHOLD, refinedFRootErrStdDevCorrect);
                errorFunctionCombinedMeanSquared(x1goodPointsMat, x2goodPointsMat, F, refinedFSampsonDistCorrect, refinedFTrueInlierCombined, INLIER_THRESHOLD, refinedFSampsonErrStdDevCorrect);
                if (LOG_DEBUG) std::cout << "Mean squared ground truth error: " << refinedFSampsonDistCorrect << " Std. dev: " << refinedFSampsonErrStdDevCorrect << ", true inlier: " << refinedFTrueInlierCombined << std::endl;
                if (LOG_DEBUG) std::cout << "Mean ground truth error: " << refinedFRootDistCorrect << " Std. dev: " << refinedFRootErrStdDevCorrect << ", true inlier: " << refinedFTrueInlierCombined << std::endl;
                if(CREATE_DEBUG_IMG) drawEpipolarLines(x1goodPoints, x2goodPoints, F, image_1_color, image_2_color, "Refined F");
            } else {
                if(CREATE_DEBUG_IMG) {
                    //rectify(x1, x2, it->getF(), image_1, image_2, "Rectified "+it->name);
                    drawEpipolarLines(x1,x2, F, image_1_color, image_2_color, "Refined F");
                }
            }
        }

        if (compareWithGroundTruth) {
            matToFile("F_gt.csv", Fgt);
            if (LOG_DEBUG) std::cout << "Ground truth = " << std::endl << Fgt << std::endl;
            errorFunctionCombinedMean(x1goodPointsMat, x2goodPointsMat, Fgt, groundTruthRootDistCorrect, combinedFeaturesCorrect, INLIER_THRESHOLD, groundTruthRootStdDev);
            errorFunctionCombinedMeanSquared(x1goodPointsMat, x2goodPointsMat, Fgt, groundTruthSampsonDistCorrect, combinedFeaturesCorrect, INLIER_THRESHOLD, groundTruthSampsonStdDev);
            //errorFunctionCombinedMeanSquared(x1Combined, x2Combined, Fgt, error, inlier, INLIER_THRESHOLD, stdDev);
            if (LOG_DEBUG) std::cout << "Mean squared error: " << groundTruthSampsonDistCorrect << " Std. dev: " << groundTruthSampsonStdDev << ", inlier: " << combinedFeaturesCorrect << std::endl;
            if (LOG_DEBUG) std::cout << "Mean error: " << groundTruthRootDistCorrect << " Std. dev: " << groundTruthRootStdDev << ", inlier: " << combinedFeaturesCorrect << std::endl;
            //meanSampsonFDistanceGoodMatches(Fgt, Fgt, x1Combined, x2Combined, groundTruthSampsonDistCombined, combinedFeaturesCorrect);
            if(CREATE_DEBUG_IMG) {
                //rectify(x1, x2, Fgt, image_1, image_2, "Rectified ground truth");
                drawEpipolarLines(x1goodPoints,x2goodPoints, Fgt, image_1_color, image_2_color, "F_groundtruth");
            }
        }

        if(LOG_DEBUG) std::cout << "done." << std::endl << "#######################################################" << std::endl << std::endl;
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

    if(CREATE_DEBUG_IMG) {
        //showImage("Image 1 original", image_1_color);
        //showImage("Image 2 original", image_2_color);
    }

    return 1;
}

FEstimationMethod* MultipleCueEstimation::calcFfromPoints() {
    FEstimatorPoints* estimatorPoints = new FEstimatorPoints(image_1, image_2, image_1_color, image_2_color, "F_points");
    if(compareWithGroundTruth) estimatorPoints->setGroundTruth(Fgt);
    estimatorPoints->compute();
    x1 = estimatorPoints->getUsedX1();
    x2 = estimatorPoints->getUsedX2();
    return estimatorPoints;
}

FEstimationMethod* MultipleCueEstimation::calcFfromHLines() {     // From Paper: "Robust line matching in image pairs of scenes with dominant planes", Sagúes C., Guerrero J.J.
    FEstimatorHLines* estomatorLines = new FEstimatorHLines(image_1, image_2, image_1_color, image_2_color, "F_Hlines");
    if(compareWithGroundTruth) estomatorLines->setGroundTruth(Fgt);
    estomatorLines->compute();
    return estomatorLines;
}

FEstimationMethod* MultipleCueEstimation::calcFfromHPoints() {    // From: two Homographies (one in each image) computed from points in general position
    FEstimatorHPoints* estomatorHPoints = new FEstimatorHPoints(image_1, image_2, image_1_color, image_2_color, "F_HPoints");
    if(compareWithGroundTruth) estomatorHPoints->setGroundTruth(Fgt);
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

    double squaredErrorThr = INLIER_THRESHOLD;

    if(estimations.size() == 0) {
        if (LOG_DEBUG) std::cout << "-- No Fundamental Matrix found!" << std::endl;
        return Mat();
    }

    std::vector<FEstimationMethod>::iterator bestMethod = estimations.begin();
    int numValues = 0;

    double smallestSampsonErr = std::numeric_limits<double>::max();
    double largestSampsonErr = 0;
    double smallestSampsonErrStdDev = std::numeric_limits<double>::max();
    double largestSampsonErrStdDev = 0;
    int largestInlier = 0;
    int smallestInlier = std::numeric_limits<int>::max();

    std::vector<Mat> estimationInlierX1, estimationInlierX2;

    for(std::vector<FEstimationMethod>::iterator estimationIter = estimations.begin(); estimationIter != estimations.end(); ++estimationIter) {
        if(estimationIter->isSuccessful()) {
            if (LOG_DEBUG) std::cout << estimationIter->name << std::endl;
            errorFunctionCombinedMeanSquared(x1Combined, x2Combined, estimationIter->getF(), estimationIter->sampsonErrCombined, estimationIter->inlierCountCombined, squaredErrorThr, estimationIter->sampsonErrStdDevCombined);
            //if (LOG_DEBUG) std::cout << "Computing mean error of combined matches for " << estimationIter->name << ": " << estimationIter->meanCSTError << " Std. dev: " << estimationIter->meanCSTErrorStandardDeviation << ", inliers: " << estimationIter->meanCSTErrorInliers << std::endl;
            meanSampsonFDistanceGoodMatches(Fgt, estimationIter->getF(), x1Combined, x2Combined, estimationIter->trueSampsonErr, combinedFeaturesCorrect);

            if(compareWithGroundTruth) {
                findGoodCombinedMatches(x1Combined, x2Combined, estimationInlierX1, estimationInlierX2, estimationIter->getF(), INLIER_THRESHOLD);
                estimationIter->trueInlierCountCombined = goodMatchesCount(Fgt, estimationInlierX1, estimationInlierX2, INLIER_THRESHOLD);
            }
            //if(bestMethod->sampsonErrCombined > estimationIter->sampsonErrCombined) {

            numValues += estimationIter->getFeaturesImg1().size();

            if(estimationIter->sampsonErrCombined > largestSampsonErr) largestSampsonErr = estimationIter->sampsonErrCombined;
            if(estimationIter->sampsonErrStdDevCombined > largestSampsonErrStdDev) largestSampsonErrStdDev = estimationIter->sampsonErrStdDevCombined;
            if(estimationIter->inlierCountCombined > largestInlier) largestInlier = estimationIter->inlierCountCombined;

            if(estimationIter->sampsonErrCombined < smallestSampsonErr) smallestSampsonErr = estimationIter->sampsonErrCombined;
            if(estimationIter->sampsonErrStdDevCombined < smallestSampsonErrStdDev) smallestSampsonErrStdDev = estimationIter->sampsonErrStdDevCombined;
            if(estimationIter->inlierCountCombined < smallestInlier) smallestInlier = estimationIter->inlierCountCombined;
        }
    }

    for(std::vector<FEstimationMethod>::iterator estimationIter = estimations.begin(); estimationIter != estimations.end(); ++estimationIter) {
        if(estimationIter->isSuccessful()) {

            estimationIter->quality = qualitiy(estimationIter->sampsonErrCombined, smallestSampsonErr, estimationIter->inlierCountCombined, largestInlier, estimationIter->sampsonErrStdDevCombined, smallestSampsonErrStdDev);

            //estimationIter->quality = qualitiy(estimationIter->sampsonErrCombined, estimationIter->inlierCountCombined, x1Combined.size(), estimationIter->sampsonErrStdDevCombined);
            if (LOG_DEBUG) std::cout << "Computing mean squared error of combined matches for " << estimationIter->name << ": " << estimationIter->sampsonErrCombined << " Std. dev: " << estimationIter->sampsonErrStdDevCombined << ", inlier: " << estimationIter->inlierCountCombined << " thereof true inlier: " << estimationIter->trueInlierCountCombined << ", quality: " << estimationIter->quality << std::endl;
            if (LOG_DEBUG) std::cout << "Ground truth error: " << meanSampsonFDistanceGoodMatches(Fgt, estimationIter->getF(), x1Combined, x2Combined) << std::endl;
            if(bestMethod->quality < estimationIter->quality) {
                bestMethod = estimationIter;
            }
        }
    }

    if(numValues < 8) {
        if (LOG_DEBUG) std::cout << "-- Not enough features!" << std::endl;
        return bestMethod->getF().clone();
    }

    //return  bestMethod->getF();

    Mat FSPLM = bestMethod->getF().clone();
    Mat result = bestMethod->getF().clone();
    std::vector<Mat> x1goodPointsSPLM;
    std::vector<Mat> x2goodPointsSPLM;
    int lastFeatureCnt = 0, featureCnt = 0;
    double lastErrorSPLM = 0, errorSPLM = 0, errorSPLMCombined = 0;

    SevenpointLevenbergMarquardtInit();
    double errorThrSPLM = (1.05 - bestMethod->quality)*INLIER_LM_THRESHOLD;
    int iter = 1;

    if (LOG_DEBUG) std::cout << "SPLM start: " << bestMethod->name << ", ground truth error: " << meanSampsonFDistanceGoodMatches(Fgt, FSPLM, x1Combined, x2Combined) << std::endl;

    findGoodCombinedMatches(x1Combined, x2Combined, x1goodPointsSPLM, x2goodPointsSPLM, FSPLM, errorThrSPLM);
    featureCnt = x1goodPointsSPLM.size();
    errorSPLM = std::numeric_limits<double>::max();

    //double quality = 0;

    //return FSPLM;
    do
    {
        SPLM(FSPLM, x1goodPointsSPLM, x2goodPointsSPLM);
        if (LOG_DEBUG) std::cout << "SPLM iter: " << iter << " error thr: " << errorThrSPLM/iter << " ground truth error: " << meanSampsonFDistanceGoodMatches(Fgt, FSPLM, x1Combined, x2Combined) << std::endl;
        findGoodCombinedMatches(x1Combined, x2Combined, x1goodPointsSPLM, x2goodPointsSPLM, FSPLM, errorThrSPLM/++iter);
        lastFeatureCnt = featureCnt;
        featureCnt = x1goodPointsSPLM.size();
        lastErrorSPLM = errorSPLM;
        //errorSPLM = sampsonDistanceFundamentalMat(FSPLM, x1Combined, x2Combined);
        errorSPLM = sampsonDistanceFundamentalMat(FSPLM, x1goodPointsSPLM, x2goodPointsSPLM);
        errorSPLMCombined = sampsonDistanceFundamentalMat(FSPLM, x1Combined, x2Combined);
        if (LOG_DEBUG) std::cout << "Features: " << featureCnt << ", Change: " << (lastFeatureCnt - featureCnt) << ", Error: " << errorSPLM << ", Error Change: " << (lastErrorSPLM - errorSPLM)/errorSPLM << ", Error combined: " << errorSPLMCombined << std::endl;
        if((lastErrorSPLM - errorSPLM)/errorSPLM < 0) break;
        result = FSPLM.clone();
    } while(abs(lastFeatureCnt - featureCnt) > 5 && (lastErrorSPLM - errorSPLM)/errorSPLM > 0.01 && errorSPLM > 0.3);

    if(compareWithGroundTruth) refinedFTrueInlierCombined = goodMatchesCount(Fgt, x1goodPointsSPLM, x2goodPointsSPLM, INLIER_THRESHOLD);

    if (CREATE_DEBUG_IMG) visualizePointMatches(image_1_color, image_2_color, x1goodPointsSPLM, x2goodPointsSPLM, 20, 2, false, "FSPLM used point matches");

    return result;
}

double MultipleCueEstimation::qualitiy(double sampsonErrCombined, double smallestSampsonErr, int inlierCountCombined, int largestInlier, double sampsonErrStdDevCombined, double smallestSampsonErrStdDev) {
    double qInlier = QI*inlierCountCombined/largestInlier;
    double qError = QE*smallestSampsonErr/sampsonErrCombined;
    double qStdDev = QS*smallestSampsonErrStdDev/sampsonErrStdDevCombined;
    return (qInlier + qError + qStdDev) / (QI+QE+QS);
}

bool MultipleCueEstimation::SPLM(Mat &F, std::vector<Mat> x1m, std::vector<Mat> x2m) {
    std::vector<double> x1, x2, y1, y2;
    std::vector<double>* Fvect = new std::vector<double>(9);

    int k = 0;
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++)
            Fvect->at(k++) = F.at<double>(i,j);
    }

    double f0 = 1.0;

    for(int i = 0; i < x1m.size(); i++) {
        x1.push_back(x1m.at(i).at<double>(0,0));
        x2.push_back(x2m.at(i).at<double>(0,0));
        y1.push_back(x1m.at(i).at<double>(1,0));
        y2.push_back(x2m.at(i).at<double>(1,0));
    }

    bool result = SevenpointLevenbergMarquardt(Fvect, x1, y1, x2, y2, f0, 50, 1e-10);  //0.5e-12
    if(!result) return false;

    k = 0;
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++)
            F.at<double>(i,j) = Fvect->at(k++);
    }
    F = F.t();

    homogMat(F);
    return true;
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
        combinedFeatures = x1Combined.size();
    }
}

void MultipleCueEstimation::combineAllPointCorrespondecies() {
    for(std::vector<FEstimationMethod>::iterator estimationIter = estimations.begin(); estimationIter != estimations.end(); ++estimationIter) {
        int cnt = 0;
        for(int i = 0; i < estimationIter->getCompleteFeaturesImg1().size(); i++) {
            Mat x11 = estimationIter->getCompleteFeaturesImg1().at(i);
            Mat x12 = estimationIter->getCompleteFeaturesImg2().at(i);
                x1Complete.push_back(x11);
                x2Complete.push_back(x12);
                cnt++;
//            }
        }
        if(LOG_DEBUG) std::cout << estimationIter->name << ": Added " << cnt << " Point correspondencies to complete feature vector" << std::endl;
    }
}
