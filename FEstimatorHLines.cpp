#include "FEstimatorHLines.h"
#include "LevenbergMarquardtLines.h"
#include "FeatureMatchers.h"

FEstimatorHLines::FEstimatorHLines(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name) {
    image_1 = img1.clone();
    image_2 = img2.clone();
    image_1_color = img1_c.clone();
    image_2_color = img2_c.clone();
    this->name = name;
    if(LOG_DEBUG) std::cout << "Estimating: " << name << std::endl;
    successful = false;
    computationType = F_FROM_LINES_VIA_H;

    compareWithGroundTruth = false;

    featureCountGood = -1;
    featureCountComplete = -1;
    inlierCountOwnGood = -1;
    inlierCountOwnComplete = -1;
    inlierCountCombined = -1;

    trueFeatureCountGood = -1;
    trueFeatureCountComplete = -1;
    trueInlierCountOwnGood = -1;
    trueInlierCountOwnComplete = -1;
    trueInlierCountCombined = -1;

    sampsonErrOwn = -1;
    sampsonErrComplete = -1;
    sampsonErrCombined = -1;
    trueSampsonErr = -1;

    sampsonErrStdDevCombined = -1;

    quality = -1;
}

FEstimatorHLines::~FEstimatorHLines() {

}

int FEstimatorHLines::extractMatches() {
    extractLineMatches(image_1, image_2, allMatchedLines);

    //Mat* T = normalizeLines(allMatchedLines, goodMatchedLines);
    //normT1 = T[0].clone();
    //normT2 = T[1].clone();

    for(std::vector<lineCorrespStruct>::const_iterator it = allMatchedLines.begin() ; it != allMatchedLines.end(); ++it) {
        if(it->isGoodMatch) {
            goodMatchedLines.push_back(getlineCorrespStruct(*it));
            goodMatchedLinesConst.push_back(getlineCorrespStruct(*it));
        }
        allMatchedLinesConst.push_back(getlineCorrespStruct(*it));
    }

    if(VISUAL_DEBUG) visualizeLineMatches(image_1, image_2, allMatchedLines, 3, false, "All line matches");
    if(VISUAL_DEBUG) visualizeLineMatches(image_1, image_2, goodMatchedLines, 2, true, "Good line matches");
}

bool FEstimatorHLines::compute() {

    extractMatches();
//    lineCorrespondencies.clear();       //TODO: Remove hard coded lines for entry 8+9
//    lineCorrespStruct lc1, lc2, lc3, lc4, lc5, lc6;
//    lc1 = getlineCorrespStruct(1092, 617, 1069, 1225, 1510, 608, 1514, 1216);
//    lc2 = getlineCorrespStruct(910, 1526, 1096, 1512, 1405, 1530, 1546, 1511);
//    lc3 = getlineCorrespStruct(1121, 897, 1209, 1149, 1542, 884, 1619, 1134);
//    lc4 = getlineCorrespStruct(1096, 1673, 1202, 1360, 1552, 1671, 1622, 1353);
//    lc5 = getlineCorrespStruct(2046, 962, 2256, 909, 2353, 855, 2540, 772);
//    lc6 = getlineCorrespStruct(2320, 1706, 2284, 1203, 2625, 1648, 2582, 1112);
//    lineCorrespondencies.push_back(lc1);
//    lineCorrespondencies.push_back(lc2);
//    lineCorrespondencies.push_back(lc3);
//    lineCorrespondencies.push_back(lc4);
//    lineCorrespondencies.push_back(lc5);
//    lineCorrespondencies.push_back(lc6);
//    if(VISUAL_DEBUG) visualizeMatches(lineCorrespondencies, 8, true, "Line matches");

    if(goodMatchedLines.size() < 2*NUM_LINE_CORRESP) {
        if(LOG_DEBUG) std::cout << "-- Estimation failed, not enough line correspondencies!" << std::endl;
        return false;
    }

    if(LOG_DEBUG) std::cout << "-- First estimation..." << std::endl;

//    std::vector<lineCorrespStruct> goodLineMatches;
//    for(std::vector<lineCorrespStruct>::const_iterator it = goodMatchedLines.begin() ; it != goodMatchedLines.end(); ++it) {
//        goodLineMatches.push_back(*it);
//    }

    lineSubsetStruct firstEstimation;
    lineSubsetStruct secondEstimation;


    std::vector<lineCorrespStruct> ransacInleirH1, ransacInleirH2;

    if(!findLineHomography(firstEstimation, goodMatchedLines, allMatchedLines, ransacInleirH1, RANSAC, RANSAC_CONFIDENCE, HOMOGRAPHY_OUTLIERS, INLIER_THRESHOLD)) {
        if(LOG_DEBUG) std::cout << "-- Estimation FAILED!" << std::endl;
        return false;
    }

    if(VISUAL_DEBUG) {
        visualizeHomography(firstEstimation.Hs, image_1, image_2, name+": H1");
        visualizeLineMatches(image_1_color, image_2_color, firstEstimation.lineCorrespondencies, 8, true, name+": H1 used Matches");
        //visualizeLineMatches(image_1_color, image_2_color, goodLineMatches, 8, true, name+": H1 good Matches");
        //visualizeProjectedLines(H1, 8, true, name+": H21 used lines projected to image 2");
    }

    cvWaitKey(0);

    filterUsedLineMatches(allMatchedLines, firstEstimation.lineCorrespondencies);
    int removed = filterUsedLineMatches(goodMatchedLines, firstEstimation.lineCorrespondencies);

    if(VISUAL_DEBUG) {
        //visualizeLineMatches(image_1_color, image_2_color, goodMatchedLines, 8, true, name+": Remaining matches for 2ed estimation");
    }

    double outliers = computeRelativeOutliers(HOMOGRAPHY_OUTLIERS, goodMatchedLines.size(), goodMatchedLines.size() + removed);

    int estCnt = 0;
    bool homographies_equal = true;
    Mat H;
    Mat e2;

    while(homographies_equal && goodMatchedLines.size() > NUM_LINE_CORRESP && MAX_H2_ESTIMATIONS > estCnt) {

        if(LOG_DEBUG) std::cout << "-- Second estimation " << estCnt << "/" << MAX_H2_ESTIMATIONS << "..." << std::endl;

        if(!findLineHomography(secondEstimation, goodMatchedLines, allMatchedLines, ransacInleirH2, RANSAC, RANSAC_CONFIDENCE, outliers, INLIER_THRESHOLD)) {
            if(LOG_DEBUG) std::cout << "-- Estimation FAILED!" << std::endl;
            return false;
        }

        H = firstEstimation.Hs*secondEstimation.Hs.inv(DECOMP_SVD); // H = (H1*H2⁻1)
        homogMat(H);

        homographies_equal = (!computeUniqeEigenvector(H, e2));
        if(homographies_equal) {
            if(LOG_DEBUG) std::cout << "-- Homographies equal, repeating estimation..." << std::endl << "-- H = " << std::endl << H << std::endl;
            filterUsedLineMatches(allMatchedLines, secondEstimation.lineCorrespondencies);
            removed = filterUsedLineMatches(goodMatchedLines, secondEstimation.lineCorrespondencies);
            outliers = computeRelativeOutliers(outliers, goodMatchedLines.size(), goodMatchedLines.size() + removed);
        }
        estCnt++;
    }

    if(homographies_equal) {     //Not able to find a secont homography
        if(LOG_DEBUG) std::cout << "-- Estimation failed!" << std::endl;
        return false;
    }

    std::vector<Mat> ransacInleirX1, ransacInleirX2;

    computePointCorrespondencies(firstEstimation.Hs, ransacInleirH1, ransacInleirX1, ransacInleirX2);
    computePointCorrespondencies(secondEstimation.Hs, ransacInleirH2, ransacInleirX1, ransacInleirX2);

    computePointCorrespondencies(firstEstimation.Hs, firstEstimation.lineCorrespondencies, featuresImg1, featuresImg2);
    computePointCorrespondencies(secondEstimation.Hs, secondEstimation.lineCorrespondencies, featuresImg1, featuresImg2);

    compfeaturesImg1 = featuresImg1;
    compfeaturesImg2 = featuresImg2;

    if(VISUAL_DEBUG) {
        visualizeHomography(secondEstimation.Hs, image_1, image_2, name+" H2");
        visualizeLineMatches(image_1_color, image_2_color, secondEstimation.lineCorrespondencies, 8, true, name+": H2 used Matches");
        //visualizeLineMatches(image_1_color, image_2_color, goodLineMatches, 8, true, name+": H2 good Matches");
        //visualizeProjectedLines(H2, 8, true, name+": H21_2 used lines projected to image 2");
        //visualizePointMatches(image_1_color, image_2_color, featuresImg1, featuresImg2, 3, true, name+": Line-generated Point correspondencies");
    }

    F = crossProductMatrix(e2)*firstEstimation.Hs;
    enforceRankTwoConstraint(F);

    featureCountGood = goodMatchedLinesConst.size();
    featureCountComplete = allMatchedLinesConst.size();
    inlierCountOwnGood = ransacInleirX1.size()/2.0;   //number of lines = 2 x number of points
    inlierCountOwnComplete = featuresImg1.size()/2.0;
    //inlierCountCombined = -1;

    if(compareWithGroundTruth) {
//        trueFeatureCountGood = -1;
//        trueFeatureCountComplete = -1;

        trueInlierCountOwnGood = goodMatchesCount(Fgt, ransacInleirX1, ransacInleirX2, INLIER_THRESHOLD)/2.0;
        trueInlierCountOwnComplete = goodMatchesCount(Fgt, featuresImg1, featuresImg2, INLIER_THRESHOLD)/2.0;

        //trueInlierCountCombined = -1;

    }


    sampsonErrOwn = sampsonDistanceFundamentalMat(F, featuresImg1, featuresImg2);
    //sampsonErrComplete = -1;
    //sampsonErrCombined = -1;
    //trueSampsonErr = -1;



//    double sampsonH1 = 0, sampsonH2 = 0;

//    Mat H_inv = firstEstimation.Hs.inv(DECOMP_SVD);
//    for(int i = 0; i < firstEstimation.lineCorrespondencies.size(); i++) {
//        lineCorrespStruct lc = firstEstimation.lineCorrespondencies.at(i);
//        sampsonH1 += sampsonDistanceHomography(firstEstimation.Hs, H_inv, lc.line1Start, lc.line1End, lc.line2Start, lc.line2End);
//    }
//    sampsonH1 /= firstEstimation.lineCorrespondencies.size();

//    H_inv = secondEstimation.Hs.inv(DECOMP_SVD);
//    for(int i = 0; i < secondEstimation.lineCorrespondencies.size(); i++) {
//        lineCorrespStruct lc = secondEstimation.lineCorrespondencies.at(i);
//        sampsonH2 += sampsonDistanceHomography(secondEstimation.Hs, H_inv, lc.line1Start, lc.line1End, lc.line2Start, lc.line2End);
//    }
//    sampsonH2 /= secondEstimation.lineCorrespondencies.size();

//    if(LOG_DEBUG) std::cout << "-- sampsonH1: " << sampsonH1 << ", sampsonH2: " << sampsonH2 << std::endl;

    successful = true;

    if(LOG_DEBUG) std::cout << "-- Added " << featuresImg1.size() << " point correspondencies to combined feature vector" << std::endl;

    if(LOG_DEBUG) std::cout << std::endl << std::endl;

    return true;
}

bool FEstimatorHLines::findLineHomography(lineSubsetStruct &bestSubset, std::vector<lineCorrespStruct> goodMatches, std::vector<lineCorrespStruct> allMatches, std::vector<lineCorrespStruct> &ransacInlier, int method, double confidence, double outliers, double threshold) {
    int N;
    //std::vector<lineCorrespStruct> lastIterLineMatches;
    std::vector<lineCorrespStruct> goodLineMatches;
    bestSubset.subsetError = 0;
    //int iteration = 0;

    if(LOG_DEBUG) std::cout << "-- findHomography: confidence = " << confidence << ", relative outliers = " << outliers << std::endl;

    goodLineMatches.clear();
    for(std::vector<lineCorrespStruct>::const_iterator it = goodMatches.begin() ; it != goodMatches.end(); ++it) {
        goodLineMatches.push_back(*it);
    }

    if(goodLineMatches.size() < NUM_LINE_CORRESP) {
        if(LOG_DEBUG) std::cout << "-- To few line matches left! ";
        if(LOG_DEBUG) std::cout << "Can't compute Homography." << std::endl;
        return false;
    }

    //outliers = computeRelativeOutliers(outliers, goodLineMatches.size(), goodLineMatches.size() + removedMatches);
    N = computeNumberOfEstimations(confidence, outliers, NUM_LINE_CORRESP);
    if(!estimateHomography(bestSubset, goodLineMatches, method, N, threshold)) {
        if(LOG_DEBUG) std::cout << "-- To few line or only parallel lines left! ";
        if(LOG_DEBUG) std::cout << "Can't compute Homography." << std::endl;
        return false;
    }

    ransacInlier.clear();
    for(std::vector<lineCorrespStruct>::const_iterator it = bestSubset.lineCorrespondencies.begin() ; it != bestSubset.lineCorrespondencies.end(); ++it) {
        ransacInlier.push_back(getlineCorrespStruct(*it));
    }


    return true;
//    visualizeHomography(bestSubset.Hs, image_1, image_2, "estimateHomography result");
//    visualizeLineMatches(image_1_color, image_2_color, bestSubset.lineCorrespondencies, 2, true, "estimateHomography line result");
//    cvWaitKey(0);

    //errorThr = errorThr;
    //errorThr = bestSubset.subsetError;

    int iterationLM = 0;
    double lastError = 0;
    int stableSolutions = 0;
    double dError = 0;
    int removedMatches = 0;

    lineSubsetStruct LMSubset;
    LMSubset.Hs = bestSubset.Hs.clone();

    LMSubset.subsetError = 0;
//        Mat H_T = bestSubset.Hs.t();
    Mat H_inv = LMSubset.Hs.inv(DECOMP_SVD);
    LMSubset.lineCorrespondencies.clear();
    for(int i = 0; i < allMatches.size(); i++) {
        lineCorrespStruct lc = allMatches.at(i);
        double error = sampsonDistanceHomography(LMSubset.Hs, H_inv, lc.line1Start, lc.line1End, lc.line2Start, lc.line2End);
        if(sqrt(error) <= threshold) {
            LMSubset.lineCorrespondencies.push_back(lc);
            lastError += error;
        }
    }
    lastError /= LMSubset.lineCorrespondencies.size();

    do {

        if(LMSubset.lineCorrespondencies.size() <= NUMERICAL_OPTIMIZATION_MIN_MATCHES) {
            if(LMSubset.lineCorrespondencies.size() < 4) return false;
            else break;
        }

        iterationLM++;

        if(LOG_DEBUG)  std::cout << "-- Numeric optimization iteration: " << iterationLM << "/" << NUMERICAL_OPTIMIZATION_MAX_ITERATIONS << ", error threshold for inliers: " << threshold << std::endl;

        removedMatches = LMSubset.lineCorrespondencies.size();

        levenbergMarquardt(LMSubset);

        LMSubset.subsetError = 0;
//        Mat H_T = bestSubset.Hs.t();
        H_inv = LMSubset.Hs.inv(DECOMP_SVD);
        LMSubset.lineCorrespondencies.clear();
        for(int i = 0; i < allMatches.size(); i++) {
            lineCorrespStruct lc = allMatches.at(i);
            double error = sampsonDistanceHomography(LMSubset.Hs, H_inv, lc.line1Start, lc.line1End, lc.line2Start, lc.line2End);
            if(sqrt(error) <= threshold) {
                LMSubset.lineCorrespondencies.push_back(lc);
                LMSubset.subsetError += error;
            }
        }
        LMSubset.subsetError /= LMSubset.lineCorrespondencies.size();

        removedMatches = removedMatches - LMSubset.lineCorrespondencies.size();

        dError = (lastError - LMSubset.subsetError)/LMSubset.subsetError;
        if(LOG_DEBUG) std::cout << "-- Mean squared error: " << LMSubset.subsetError << ", rel. Error change: "<< dError << ", num Matches: " << LMSubset.lineCorrespondencies.size() << ", removed: " << removedMatches << std::endl;

        if(dError < 0 || iterationLM == NUMERICAL_OPTIMIZATION_MAX_ITERATIONS) break;

        bestSubset.Hs = LMSubset.Hs.clone();

        lastError = LMSubset.subsetError;

        //if((dError >=0 && dError <= MAX_ERROR_CHANGE) || abs(removedMatches) <= MAX_FEATURE_CHANGE) stableSolutions++;
        if((dError >=0 && dError <= MAX_ERROR_CHANGE) || abs(removedMatches) <= MAX_FEATURE_CHANGE) stableSolutions++;
        else stableSolutions = 0;

        if(LOG_DEBUG) std::cout << "-- Stable solutions: " << stableSolutions << std::endl;

        //if(LMSubset.lineCorrespondencies.size() <= NUMERICAL_OPTIMIZATION_MIN_MATCHES) break;

    } while(stableSolutions < 3);

    H_inv = bestSubset.Hs.inv(DECOMP_SVD);
    bestSubset.subsetError = 0;
    bestSubset.lineCorrespondencies.clear();
    for(int i = 0; i < allMatches.size(); i++) {
        lineCorrespStruct lc = allMatches.at(i);
        double error = sampsonDistanceHomography(bestSubset.Hs, H_inv, lc.line1Start, lc.line1End, lc.line2Start, lc.line2End);
        if(sqrt(error) <= threshold) {
            bestSubset.lineCorrespondencies.push_back(lc);
            bestSubset.subsetError += error;
        }
    }
    bestSubset.subsetError /= bestSubset.lineCorrespondencies.size();

    if(LOG_DEBUG) std::cout << "-- Final number of used matches: " << bestSubset.lineCorrespondencies.size() << ", Mean squared error: " << bestSubset.subsetError << std::endl;

    //bestSubset.Hs = denormalize(bestSubset.Hs, normT1, normT2);

    return true;
}

double FEstimatorHLines::levenbergMarquardt(lineSubsetStruct &bestSubset) {
    Eigen::VectorXd x(9);

    x(0) = bestSubset.Hs.at<double>(0,0);
    x(1) = bestSubset.Hs.at<double>(0,1);
    x(2) = bestSubset.Hs.at<double>(0,2);

    x(3) = bestSubset.Hs.at<double>(1,0);
    x(4) = bestSubset.Hs.at<double>(1,1);
    x(5) = bestSubset.Hs.at<double>(1,2);

    x(6) = bestSubset.Hs.at<double>(2,0);
    x(7) = bestSubset.Hs.at<double>(2,1);
    x(8) = bestSubset.Hs.at<double>(2,2);

    LineFunctor functor;
    functor.estimator = this;
    functor.lines = &bestSubset;
    Eigen::NumericalDiff<LineFunctor> numDiff(functor, 1.0e-6); //epsilon
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<LineFunctor>,double> lm(numDiff);

    lm.parameters.ftol = 1.0e-10;
    lm.parameters.xtol = 1.0e-10;
    //lm.parameters.epsfcn = 1.0e-3;
    lm.parameters.maxfev = 40; // Max iterations
    Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(x);

    if (LOG_DEBUG) std::cout << "-- LMA Iterations: " << lm.nfev << ", Status: " << status << std::endl;

    bestSubset.Hs.at<double>(0,0) = x(0);
    bestSubset.Hs.at<double>(0,1) = x(1);
    bestSubset.Hs.at<double>(0,2) = x(2);

    bestSubset.Hs.at<double>(1,0) = x(3);
    bestSubset.Hs.at<double>(1,1) = x(4);
    bestSubset.Hs.at<double>(1,2) = x(5);

    bestSubset.Hs.at<double>(2,0) = x(6);
    bestSubset.Hs.at<double>(2,1) = x(7);
    bestSubset.Hs.at<double>(2,2) = x(8);

    homogMat(bestSubset.Hs);

    double error = 0;

////    Mat H_T = bestSubset.Hs.t();
//    Mat H_inv = bestSubset.Hs.inv(DECOMP_SVD);
//    for(std::vector<lineCorrespStruct>::const_iterator it = bestSubset.lineCorrespondencies.begin(); it != bestSubset.lineCorrespondencies.end(); ++it) {
//        //error += errorFunctionHLinesSqared_(H_invT, H_T, it->line1StartNormalized, it->line1EndNormalized, it->line2StartNormalized, it->line2EndNormalized);
//        error += sampsonDistanceHomography(bestSubset.Hs, H_inv, it->line1Start, it->line1End, it->line2Start, it->line2End);
//    }

//    if(bestSubset.lineCorrespondencies.size() > 0) bestSubset.subsetError = error/bestSubset.lineCorrespondencies.size();

    return error;
}

bool FEstimatorHLines::estimateHomography(lineSubsetStruct &result, std::vector<lineCorrespStruct> lineCorrespondencies, int method, int sets, double ransacThr) {
    int numOfPairs = lineCorrespondencies.size();
    std::vector<lineSubsetStruct> subsets;
    if(LOG_DEBUG) std::cout << "-- Computing "<< sets << " Homographies, using " << NUM_LINE_CORRESP << " point correspondencies each" << std::endl;
    //Compute H_21 from line correspondencies
    srand(time(NULL));  //Init random generator
    for(int i = 0; i < sets; i++) {
        lineSubsetStruct subset;

        for(int j = 0; j < NUM_LINE_CORRESP; j++) {
            int subsetIdx = 0;
            int search = 0;
            do {        //Generate NUM_CORRESP uniqe random indices for line pairs where not 3 are parallel
                subsetIdx = std::rand() % numOfPairs;
                search++;
                if(search == MAX_POINT_SEARCH) return false;    //only parallel lines remaining
            } while(!isUniqe(subset.lineCorrespondencies, lineCorrespondencies.at(subsetIdx)) || isParallel(subset.lineCorrespondencies, lineCorrespondencies.at(subsetIdx)));

            subset.lineCorrespondencies.push_back(getlineCorrespStruct(lineCorrespondencies.at(subsetIdx)));
        }
        computeHomography(subset);
        subsets.push_back(subset);
    }

    //if(method == RANSAC)
    result = calcRANSAC(subsets, ransacThr, lineCorrespondencies);
    if(result.lineCorrespondencies.size() >= 4) return true;
    //else result = calcLMedS(subsets, lineCorrespondencies);
    return false;
}

bool FEstimatorHLines::computeHomography(lineSubsetStruct &subset) {
    Mat* norm = normalizeLines(subset.lineCorrespondencies);
    Mat linEq = Mat::ones(2*subset.lineCorrespondencies.size(),9,CV_64FC1);
    fillHLinEq(linEq, subset.lineCorrespondencies);
    SVD svd;
    svd.solveZ(linEq, subset.Hs_normalized);
    subset.Hs_normalized = subset.Hs_normalized.reshape(1,3);
    homogMat(subset.Hs_normalized);
    subset.Hs = denormalize(subset.Hs_normalized, norm[0], norm[1]);
    homogMat(subset.Hs);
    //subset.Hs = subset.Hs_normalized;
    return true;
}

bool FEstimatorHLines::isParallel(std::vector<lineCorrespStruct> fixedCorresp, lineCorrespStruct lcNew) {
    if(fixedCorresp.size() < 3) return false;
    int parallelCout = 0;
    lineCorrespStruct lc;
    for(int i = 0; i < fixedCorresp.size(); i++) {
        lc = fixedCorresp.at(i);
        if(smallestRelAngle(lc.line1Angle, lcNew.line1Angle) < MAX_ANGLE_DIFF || smallestRelAngle(lc.line2Angle, lcNew.line2Angle) < MAX_ANGLE_DIFF) {
            parallelCout++;
        }
    }
    if(parallelCout >= 3) return true;
    return false;
}

int FEstimatorHLines::filterUsedLineMatches(std::vector<lineCorrespStruct> &matches, std::vector<lineCorrespStruct> usedMatches) {
    std::vector<lineCorrespStruct>::iterator it= matches.begin();
    int removed = 0;
    while (it!=matches.end()) {
        bool remove = false;
        for(std::vector<lineCorrespStruct>::const_iterator used = usedMatches.begin(); used != usedMatches.end(); ++used) {
            if(it->id == used->id) {
                removed++;
                remove = true;
                break;
            }
        }
        if(remove) matches.erase(it);
        else {
            it++;
        }
    }
    if(LOG_DEBUG) std::cout << "-- Refined number of matches: " << matches.size() <<  ", removed: " << removed << std::endl;
    return removed;
}

void FEstimatorHLines::visualizeProjectedLines(lineSubsetStruct subset, int lineWidth, bool drawConnections, std::string name) {
    Mat img;
    hconcat(image_1_color.clone(), image_2_color.clone(), img);
    for(std::vector<lineCorrespStruct>::iterator it = subset.lineCorrespondencies.begin() ; it != subset.lineCorrespondencies.end(); ++it) {
        Mat start2 = subset.Hs*it->line1Start;
        start2 /= start2.at<double>(2,0);
        Mat end2 = subset.Hs*it->line1End;
        end2 /= end2.at<double>(2,0);
        Scalar color = Scalar(rand()%255, rand()%255, rand()%255);
        cv::line(img, cvPoint2D32f(it->line1Start.at<double>(0,0), it->line1Start.at<double>(1,0)), cvPoint2D32f(it->line1End.at<double>(0,0), it->line1End.at<double>(1,0)), color, lineWidth);
        cv::line(img, cvPoint2D32f(start2.at<double>(0,0) + image_1_color.cols, start2.at<double>(1,0)), cvPoint2D32f(end2.at<double>(0,0) + image_1_color.cols, end2.at<double>(1,0)), color, lineWidth);
        if(drawConnections) {
            cv::line(img, cvPoint2D32f(it->line1Start.at<double>(0,0), it->line1Start.at<double>(1,0)), cvPoint2D32f(start2.at<double>(0,0) + image_1_color.cols, start2.at<double>(1,0)), color, lineWidth);
        }
    }
    showImage(name, img, WINDOW_NORMAL, 1600);
}

void FEstimatorHLines::fillHLinEq(Mat &linEq, std::vector<lineCorrespStruct> correspondencies) {
    lineCorrespStruct lc;
    for(int i = 0; i < correspondencies.size(); i++) {
        lc = correspondencies.at(i);
        double A = lc.line2StartNormalized.at<double>(1,0) - lc.line2EndNormalized.at<double>(1,0);
        double B = lc.line2EndNormalized.at<double>(0,0) - lc.line2StartNormalized.at<double>(0,0);
        double C = lc.line2StartNormalized.at<double>(0,0)*lc.line2EndNormalized.at<double>(1,0) - lc.line2EndNormalized.at<double>(0,0)*lc.line2StartNormalized.at<double>(1,0);
        int row = 2*i;
        fillHLinEqBase(linEq, lc.line1StartNormalized.at<double>(0,0), lc.line1StartNormalized.at<double>(1,0), A, B, C, row);
        fillHLinEqBase(linEq, lc.line1EndNormalized.at<double>(0,0), lc.line1EndNormalized.at<double>(1,0), A, B, C, row + 1);
    }
}

void FEstimatorHLines::fillHLinEqBase(Mat &linEq, double x, double y, double A, double B, double C, int row) {
    linEq.at<double>(row, 0) = A*x;
    linEq.at<double>(row, 1) = A*y;
    linEq.at<double>(row, 2) = A;
    linEq.at<double>(row, 3) = B*x;
    linEq.at<double>(row, 4) = B*y;
    linEq.at<double>(row, 5) = B;
    linEq.at<double>(row, 6) = C*x;
    linEq.at<double>(row, 7) = C*y;
    linEq.at<double>(row, 8) = C;
}

lineSubsetStruct FEstimatorHLines::calcRANSAC(std::vector<lineSubsetStruct> &subsets, double threshold, std::vector<lineCorrespStruct> lineCorrespondencies) {
    if(LOG_DEBUG) std::cout << "-- Computing RANSAC of " << subsets.size() << " Homographies" << std::endl;
    lineSubsetStruct bestSolution = *subsets.begin();
    bestSolution.qualityMeasure = 0;
    double error = 0;
    for(std::vector<lineSubsetStruct>::iterator subset = subsets.begin() ; subset != subsets.end(); ++subset) {
        subset->qualityMeasure = 0;       //count inlainers
        subset->subsetError = 0;
        //subset->lineCorrespondencies.clear();
        Mat H_inv = subset->Hs.inv(DECOMP_SVD);
        for(std::vector<lineCorrespStruct>::iterator it = lineCorrespondencies.begin() ; it != lineCorrespondencies.end(); ++it) {
            error = sampsonDistanceHomography(subset->Hs, H_inv, it->line1Start, it->line1End, it->line2Start, it->line2End);
            if(sqrt(error) <= threshold) {
                subset->subsetError += error;
                subset->qualityMeasure++;
                //subset->lineCorrespondencies.push_back(*it);
            }
        }
        subset->subsetError /= subset->qualityMeasure;
        if(subset->qualityMeasure > bestSolution.qualityMeasure)
            bestSolution = *subset;
    }

    if(bestSolution.qualityMeasure > 4) {

        Mat H_inv = bestSolution.Hs.inv(DECOMP_SVD);
        bestSolution.lineCorrespondencies.clear();
        for(std::vector<lineCorrespStruct>::iterator it = lineCorrespondencies.begin() ; it != lineCorrespondencies.end(); ++it) {
            error = sampsonDistanceHomography(bestSolution.Hs, H_inv, it->line1Start, it->line1End, it->line2Start, it->line2End);
            if(sqrt(error) <= threshold) {
                bestSolution.lineCorrespondencies.push_back(getlineCorrespStruct(*it));
            }
        }

        computeHomography(bestSolution);

    }
    if(LOG_DEBUG) std::cout << "-- RANSAC inlaiers: " << bestSolution.qualityMeasure << ", error: " << bestSolution.subsetError << std::endl;
    return bestSolution;
}

//lineSubsetStruct FEstimatorHLines::calcLMedS(std::vector<lineSubsetStruct> &subsets, std::vector<lineCorrespStruct> lineCorrespondencies) {
//    if(LOG_DEBUG) std::cout << "-- Computing LMedS of " << subsets.size() << " Homographies" << std::endl;
//    std::vector<lineSubsetStruct>::iterator it = subsets.begin();
//    lineSubsetStruct lMedSsubset = *it;
//    lMedSsubset.qualityMeasure = calcMedS(*it, lineCorrespondencies);
//    if(subsets.size() < 2) return lMedSsubset;
//    it++;
//    do {
//        it->qualityMeasure = calcMedS(*it, lineCorrespondencies);
//        if(it->qualityMeasure < lMedSsubset.qualityMeasure) {
//            lMedSsubset = *it;
//        }
//        it++;
//    } while(it != subsets.end());

//    lMedSsubset.subsetError = lMedSsubset.qualityMeasure;
//    if(LOG_DEBUG) std::cout << "-- LMEDS: " << lMedSsubset.qualityMeasure << std::endl;
//    return lMedSsubset;
//}


//double FEstimatorHLines::calcMedS(lineSubsetStruct &subset, std::vector<lineCorrespStruct> lineCorrespondencies) {
//    std::vector<double> errors;
//    double error;
//    for(std::vector<lineCorrespStruct>::iterator it = lineCorrespondencies.begin() ; it != lineCorrespondencies.end(); ++it) {
//        error = sampsonDistanceHomography(subset.Hs, it->line1Start, it->line1End, it->line2Start, it->line2End);
//        errors.push_back(error);
//        subset.subsetError += error;
//    }
//    subset.subsetError /= lineCorrespondencies.size();
//    std::sort(errors.begin(), errors.end());
//    return errors.at(errors.size()/2);
//}

Mat* FEstimatorHLines::normalizeLines(std::vector<lineCorrespStruct> &correspondencies , std::vector<lineCorrespStruct> &goodCorrespondencies) {

    //Normalization: Hartley, Zisserman, Multiple View Geometry in Computer Vision, p. 109

    Mat* normalizationMats = new Mat[2];
    double sum1x = 0, sum1y = 0, sum2x = 0, sum2y = 0, N = 0;
    double mean1x = 0, mean1y = 0, mean2x = 0, mean2y = 0, v1 = 0, v2 = 0, scale1 = 0, scale2 = 0;

    for (std::vector<lineCorrespStruct>::iterator it= correspondencies.begin(); it!=correspondencies.end(); ++it) {

        sum1x += it->line1Start.at<double>(0,0) + it->line1End.at<double>(0,0);
        sum2x += it->line2Start.at<double>(0,0) + it->line2End.at<double>(0,0);

        sum1y += it->line1Start.at<double>(1,0) + it->line1End.at<double>(1,0);
        sum2y += it->line2Start.at<double>(1,0) + it->line2End.at<double>(1,0);

    }

    normalizationMats[0] = Mat::eye(3,3, CV_64FC1);
    normalizationMats[1] = Mat::eye(3,3, CV_64FC1);
    N = 2*correspondencies.size();

    mean1x = sum1x/N;
    mean1y = sum1y/N;
    mean2x = sum2x/N;
    mean2y = sum2y/N;

    for (std::vector<lineCorrespStruct>::iterator it= correspondencies.begin(); it!=correspondencies.end(); ++it) {
        v1 += fnorm(it->line1Start.at<double>(0,0)-mean1x, it->line1Start.at<double>(1,0)-mean1y);
        v1 += fnorm(it->line1End.at<double>(0,0)-mean1x, it->line1End.at<double>(1,0)-mean1y);
        v2 += fnorm(it->line2Start.at<double>(0,0)-mean2x, it->line2Start.at<double>(1,0)-mean2y);
        v2 += fnorm(it->line2End.at<double>(0,0)-mean2x, it->line2End.at<double>(1,0)-mean2y);
    }

    v1 /= N;
    v2 /= N;

    scale1 = sqrt(2.0)/v1;
    scale2 = sqrt(2.0)/v2;

    normalizationMats[0].at<double>(0,0) = scale1;
    normalizationMats[0].at<double>(1,1) = scale1;
    normalizationMats[0].at<double>(0,2) = -scale1*mean1x;
    normalizationMats[0].at<double>(1,2) = -scale1*mean1y;

    normalizationMats[1].at<double>(0,0) = scale2;
    normalizationMats[1].at<double>(1,1) = scale2;
    normalizationMats[1].at<double>(0,2) = -scale2*mean2x;
    normalizationMats[1].at<double>(1,2) = -scale2*mean2y;

    //Carry out normalization:

//    std::cout << normalizationMats[0] << std::endl;
//    std::cout << normalizationMats[1] << std::endl;

    for (std::vector<lineCorrespStruct>::iterator it= correspondencies.begin(); it!=correspondencies.end(); ++it) {

        it->line1StartNormalized = normalizationMats[0]*it->line1Start;
        it->line2StartNormalized = normalizationMats[1]*it->line2Start;

        it->line1EndNormalized = normalizationMats[0]*it->line1End;
        it->line2EndNormalized = normalizationMats[1]*it->line2End;

        if(it->isGoodMatch) {
            goodCorrespondencies.push_back(getlineCorrespStruct(*it));
        }

    }

    return normalizationMats;
}

Mat* FEstimatorHLines::normalizeLines(std::vector<lineCorrespStruct> &correspondencies) {

    //Normalization: Hartley, Zisserman, Multiple View Geometry in Computer Vision, p. 109

    Mat* normalizationMats = new Mat[2];
    double sum1x = 0, sum1y = 0, sum2x = 0, sum2y = 0, N = 0;
    double mean1x = 0, mean1y = 0, mean2x = 0, mean2y = 0, v1 = 0, v2 = 0, scale1 = 0, scale2 = 0;

    for (std::vector<lineCorrespStruct>::iterator it= correspondencies.begin(); it!=correspondencies.end(); ++it) {

        sum1x += it->line1Start.at<double>(0,0) + it->line1End.at<double>(0,0);
        sum2x += it->line2Start.at<double>(0,0) + it->line2End.at<double>(0,0);

        sum1y += it->line1Start.at<double>(1,0) + it->line1End.at<double>(1,0);
        sum2y += it->line2Start.at<double>(1,0) + it->line2End.at<double>(1,0);

    }

    normalizationMats[0] = Mat::eye(3,3, CV_64FC1);
    normalizationMats[1] = Mat::eye(3,3, CV_64FC1);
    N = 2*correspondencies.size();

    mean1x = sum1x/N;
    mean1y = sum1y/N;
    mean2x = sum2x/N;
    mean2y = sum2y/N;

    for (std::vector<lineCorrespStruct>::iterator it= correspondencies.begin(); it!=correspondencies.end(); ++it) {
        v1 += fnorm(it->line1Start.at<double>(0,0)-mean1x, it->line1Start.at<double>(1,0)-mean1y);
        v1 += fnorm(it->line1End.at<double>(0,0)-mean1x, it->line1End.at<double>(1,0)-mean1y);
        v2 += fnorm(it->line2Start.at<double>(0,0)-mean2x, it->line2Start.at<double>(1,0)-mean2y);
        v2 += fnorm(it->line2End.at<double>(0,0)-mean2x, it->line2End.at<double>(1,0)-mean2y);
    }

    v1 /= N;
    v2 /= N;

    scale1 = sqrt(2.0)/v1;
    scale2 = sqrt(2.0)/v2;

    normalizationMats[0].at<double>(0,0) = scale1;
    normalizationMats[0].at<double>(1,1) = scale1;
    normalizationMats[0].at<double>(0,2) = -scale1*mean1x;
    normalizationMats[0].at<double>(1,2) = -scale1*mean1y;

    normalizationMats[1].at<double>(0,0) = scale2;
    normalizationMats[1].at<double>(1,1) = scale2;
    normalizationMats[1].at<double>(0,2) = -scale2*mean2x;
    normalizationMats[1].at<double>(1,2) = -scale2*mean2y;

    //Carry out normalization:

//    std::cout << normalizationMats[0] << std::endl;
//    std::cout << normalizationMats[1] << std::endl;

    for (std::vector<lineCorrespStruct>::iterator it= correspondencies.begin(); it!=correspondencies.end(); ++it) {

        it->line1StartNormalized = normalizationMats[0]*it->line1Start;
        it->line2StartNormalized = normalizationMats[1]*it->line2Start;

        it->line1EndNormalized = normalizationMats[0]*it->line1End;
        it->line2EndNormalized = normalizationMats[1]*it->line2End;
    }

    return normalizationMats;
}

void FEstimatorHLines::computePointCorrespondencies(Mat H, std::vector<lineCorrespStruct> lineMatches, std::vector<Mat> &pointMatchesX1, std::vector<Mat> &pointMatchesX2) {
    Mat H_inv = H.inv(DECOMP_SVD);
    for(std::vector<lineCorrespStruct>::const_iterator iter = lineMatches.begin(); iter != lineMatches.end(); ++iter) {
        Mat l2sProjection = H*iter->line1Start;
        Mat l2eProjection = H*iter->line1End;
        Mat l1sProjection = H_inv*iter->line2Start;
        Mat l1eProjection = H_inv*iter->line2End;

        homogMat(l1sProjection);
        homogMat(l1eProjection);
        homogMat(l2sProjection);
        homogMat(l2eProjection);

        pointMatchesX1.push_back(iter->line1Start);
        pointMatchesX1.push_back(iter->line1End);
        pointMatchesX2.push_back(iter->line2Start);
        pointMatchesX2.push_back(iter->line2End);

        pointMatchesX1.push_back(l1sProjection);
        pointMatchesX1.push_back(l1eProjection);
        pointMatchesX2.push_back(l2sProjection);
        pointMatchesX2.push_back(l2eProjection);

//        compfeaturesImg1.push_back(iter->line1Start);
//        compfeaturesImg1.push_back(iter->line1End);
//        compfeaturesImg2.push_back(iter->line2Start);
//        compfeaturesImg2.push_back(iter->line2End);

//        compfeaturesImg1.push_back(l1sProjection);
//        compfeaturesImg1.push_back(l1eProjection);
//        compfeaturesImg2.push_back(l2sProjection);
//        compfeaturesImg2.push_back(l2eProjection);
    }
}

void FEstimatorHLines::addAllPointCorrespondencies(Mat H, std::vector<lineCorrespStruct> goodLineMatches) {
    Mat H_inv = H.inv(DECOMP_SVD);
    for(std::vector<lineCorrespStruct>::const_iterator iter = goodLineMatches.begin(); iter != goodLineMatches.end(); ++iter) {
        Mat l2sProjection = H*iter->line1Start;
        Mat l2eProjection = H*iter->line1End;
        Mat l1sProjection = H_inv*iter->line2Start;
        Mat l1eProjection = H_inv*iter->line2End;

        homogMat(l1sProjection);
        homogMat(l1eProjection);
        homogMat(l2sProjection);
        homogMat(l2eProjection);

        compfeaturesImg1.push_back(iter->line1Start);
        compfeaturesImg1.push_back(iter->line1End);
        compfeaturesImg2.push_back(iter->line2Start);
        compfeaturesImg2.push_back(iter->line2End);

        compfeaturesImg1.push_back(l1sProjection);
        compfeaturesImg1.push_back(l1eProjection);
        compfeaturesImg2.push_back(l2sProjection);
        compfeaturesImg2.push_back(l2eProjection);
    }
}

//double FEstimatorHLines::squaredSymmeticTransferLineError(Mat H, lineCorrespStruct lc) {
//    Mat H_invT = H.inv(DECOMP_SVD).t();
//    Mat H_T = H.t();
//    return squaredSymmeticTransferLineError(H_invT, H_T, lc);
//}

//double FEstimatorHLines::errorFunctionHLinesSqaredAlgebraic_(Mat H_invT, Mat H_T, Mat l1s, Mat l1e, Mat l2s, Mat l2e) {
//    Mat A = (l1s.t()*H_T*crossProductMatrix(l2s)*l2e)*(l1s.t()*H_T*crossProductMatrix(l2s)*l2e) + (l1e.t()*H_T*crossProductMatrix(l2s)*l2e)*(l1e.t()*H_T*crossProductMatrix(l2s)*l2e);
//    Mat C = (l2s.t()*H_invT*crossProductMatrix(l1s)*l1e)*(l2s.t()*H_invT*crossProductMatrix(l1s)*l1e) + (l2e.t()*H_invT*crossProductMatrix(l1s)*l1e)*(l2e.t()*H_invT*crossProductMatrix(l1s)*l1e);
//    return std::pow(A.at<double>(0,0),2) + std::pow(C.at<double>(0,0),2);
//}

//double FEstimatorHLines::errorFunctionHLinesSqared_(Mat H_invT, Mat H_T, Mat l1s, Mat l1e, Mat l2s, Mat l2e) {
//    return errorFunctionHLinesSqared(H_T, l1s, l1e, l2s, l2e) + errorFunctionHLinesSqared(H_invT, l2s, l2e, l1s, l1e);
//}

//double FEstimatorHLines::errorFunctionHLines_(Mat H_invT, Mat H_T, Mat l1s, Mat l1e, Mat l2s, Mat l2e) {
//    return fabs(errorFunctionHLines(H_T, l1s, l1e, l2s, l2e)) + fabs(errorFunctionHLines(H_invT, l2s, l2e, l1s, l1e));
//    //return errorFunctionHLinesSqared_(H_invT, H_T, l1s, l1e, l2s, l2e);
//}

int FEstimatorHLines::filterBadLineMatches(lineSubsetStruct subset, std::vector<lineCorrespStruct> &lineCorresp, double threshold) {
    int removed = 0;
    std::vector<lineCorrespStruct>::iterator it= lineCorresp.begin();
    Mat H_inv = subset.Hs.inv(DECOMP_SVD);
    while (it!=lineCorresp.end()) {
        if(sqrt(sampsonDistanceHomography(subset.Hs, H_inv, it->line1Start, it->line1End, it->line2Start, it->line2End)) > threshold) {
            removed++;
            lineCorresp.erase(it);
        } else it++;
    }
    if(LOG_DEBUG) std::cout << "-- Refined number of matches: " << lineCorresp.size() <<  ", removed: " << removed << ", threshold: " << threshold << std::endl;
    return removed;
}

bool FEstimatorHLines::isUniqe(std::vector<lineCorrespStruct> existingCorresp, lineCorrespStruct newCorresp) {
    if(existingCorresp.size() == 0) return true;
    for(std::vector<lineCorrespStruct>::const_iterator iter = existingCorresp.begin(); iter != existingCorresp.end(); ++iter) {
        if(iter->id == newCorresp.id) return false;
    }
    return true;
}
