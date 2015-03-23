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
}

FEstimatorHLines::~FEstimatorHLines() {

}

int FEstimatorHLines::extractMatches() {
    extractLineMatches(image_1, image_2, matchedLines);
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

    if(matchedLines.size() < 2*NUM_CORRESP) {
        if(LOG_DEBUG) std::cout << "-- Estimation failed, not enough line correspondencies!" << std::endl;
        return false;
    }

    if(LOG_DEBUG) std::cout << "-- First estimation..." << std::endl;

    std::vector<lineCorrespStruct> goodLineMatches;
    for(std::vector<lineCorrespStruct>::const_iterator it = matchedLines.begin() ; it != matchedLines.end(); ++it) {
        goodLineMatches.push_back(*it);
    }

    lineSubsetStruct H1;

    if(!findHomography(goodLineMatches, LMEDS, CONFIDENCE, LINE_OUTLIERS, H1)) {
        if(LOG_DEBUG) std::cout << "-- Estimation FAILED!" << std::endl;
        return false;
    }

    for(std::vector<lineCorrespStruct>::const_iterator it = goodLineMatches.begin() ; it != goodLineMatches.end(); ++it) {
        lineCorrespStruct lc = *it;
        featuresImg1.push_back(lc.line1Start.clone());
        featuresImg1.push_back(lc.line1End.clone());
        featuresImg2.push_back(lc.line2Start.clone());
        featuresImg2.push_back(lc.line2End.clone());
    }

    if(VISUAL_DEBUG) {
        visualizeHomography(H1.Hs, image_1, image_2, "H21");
        visualizeMatches(image_1_color, image_2_color, H1.lineCorrespondencies, 8, true, "H21 used Matches");
        visualizeProjectedLines(H1, 8, true, "H21 used lines projected to image 2");
    }

    if(LOG_DEBUG) std::cout << "-- Second estimation..." << std::endl;

    double outlierReduction = filterUsedLineMatches(matchedLines, goodLineMatches);

    lineSubsetStruct H2;
    int estCnt = 0;
    bool H_close_to_unitiy = true;
    Mat H;
    float outliers = LINE_OUTLIERS * outlierReduction;

    while(H_close_to_unitiy && matchedLines.size() > NUM_CORRESP && MAX_H2_ESTIMATIONS > estCnt) {

        goodLineMatches.clear();
        for(std::vector<lineCorrespStruct>::const_iterator it = matchedLines.begin() ; it != matchedLines.end(); ++it) {
            goodLineMatches.push_back(*it);
        }
        if(!findHomography(goodLineMatches, RANSAC, CONFIDENCE, outliers, H2)) {
            if(LOG_DEBUG) std::cout << "-- Estimation FAILED!" << std::endl;
            return false;
        }

        H = H1.Hs*H2.Hs.inv(DECOMP_SVD); // H = (H1*H2⁻1)

        H_close_to_unitiy = isUnity(H);
        if(H_close_to_unitiy) {
            if(LOG_DEBUG) std::cout << "-- H close to unity, repeating estimation..." << std::endl << "-- H = " << std::endl << H << std::endl;
            outliers = outliers * filterUsedLineMatches(matchedLines, goodLineMatches);
        }
        estCnt++;
    }

    if(H_close_to_unitiy) {     //Not able to find a secont homographie
        if(LOG_DEBUG) std::cout << "-- Estimation failed!" << std::endl;
        return false;
    }

    for(std::vector<lineCorrespStruct>::const_iterator it = goodLineMatches.begin() ; it != goodLineMatches.end(); ++it) {
        lineCorrespStruct lc = *it;
        featuresImg1.push_back(lc.line1Start.clone());
        featuresImg1.push_back(lc.line1End.clone());
        featuresImg2.push_back(lc.line2Start.clone());
        featuresImg2.push_back(lc.line2End.clone());
    }

    if(VISUAL_DEBUG) {
        visualizeHomography(H2.Hs, image_1, image_2, "H21_2");
        visualizeMatches(image_1_color, image_2_color, H2.lineCorrespondencies, 8, true, "H21_2 used Matches");
        visualizeProjectedLines(H2, 8, true, "H21_2 used lines projected to image 2");
    }

    Mat e = computeUniqeEigenvector(H);

    F = crossProductMatrix(e)*H1.Hs;
    enforceRankTwoConstraint(F);

    meanSymmetricTranferError = 0;
    Mat H_t = H1.Hs.t();
    Mat H_invT = H1.Hs.inv(DECOMP_SVD).t();
    for(std::vector<lineCorrespStruct>::iterator it = H1.lineCorrespondencies.begin() ; it != H1.lineCorrespondencies.end(); ++it) {
        meanSymmetricTranferError += squaredSymmeticTransferError(H_invT, H_t, *it);
    }
    H_t = H2.Hs.t();
    H_invT = H2.Hs.inv(DECOMP_SVD).t();
    for(std::vector<lineCorrespStruct>::iterator it = H2.lineCorrespondencies.begin() ; it != H2.lineCorrespondencies.end(); ++it) {
        meanSymmetricTranferError += squaredSymmeticTransferError(H_invT, H_t, *it);
    }
    meanSymmetricTranferError /= (H1.lineCorrespondencies.size() + H2.lineCorrespondencies.size());

    successful = true;

    if(LOG_DEBUG) std::cout << std::endl << std::endl;

    return true;
}

bool FEstimatorHLines::findHomography(std::vector<lineCorrespStruct> &goodLineMatches, int method, double confidence, double outliers, lineSubsetStruct &result) {
    double lastError = 0;
    int N;
    std::vector<lineCorrespStruct> lastIterLineMatches;
    lineSubsetStruct bestSubset;
    bestSubset.meanSquaredSymmeticTransferError = 0;
    int iteration = 0;
    int stableSolutions = 0;
    double dError = 0;

    if(LOG_DEBUG) std::cout << "-- findHomography: confidence = " << confidence << ", relative outliers = " << outliers << std::endl;

    N = std::log(1.0 - confidence)/std::log(1.0 - std::pow(1.0 - outliers, NUM_CORRESP)); //See Hartley, Zisserman p119
    bestSubset = estimateHomography(goodLineMatches, method, N);

    do {

        iteration++;

        if(LOG_DEBUG) std::cout << "-- Iteration: " << iteration <<"/" << MAX_REFINEMENT_ITERATIONS << ", Used number of matches: " << goodLineMatches.size() << std::endl;

        if(goodLineMatches.size() < NUM_CORRESP) {
            if(LOG_DEBUG) std::cout << "-- To few line matches left! Using second best solution..." << std::endl;
            goodLineMatches = lastIterLineMatches;
            break;
        }

        lastError = bestSubset.meanSquaredSymmeticTransferError;

        levenbergMarquardt(bestSubset);

        //outliers = goodLineMatches.size();

        lastIterLineMatches.clear();
        for(std::vector<lineCorrespStruct>::const_iterator it = goodLineMatches.begin() ; it != goodLineMatches.end(); ++it) {
            lastIterLineMatches.push_back(*it);
        }

        goodLineMatches.clear();
        for(std::vector<lineCorrespStruct>::const_iterator it = matchedLines.begin() ; it != matchedLines.end(); ++it) {
            if(squaredSymmeticTransferError(bestSubset.Hs, *it) < MAX_TRANSFER_DIST) goodLineMatches.push_back(*it);
        }

        //outliers = goodLineMatches.size() / outliers;

        if(iteration == MAX_REFINEMENT_ITERATIONS) return false;

        dError = (lastError - bestSubset.meanSquaredSymmeticTransferError)/bestSubset.meanSquaredSymmeticTransferError;
        if(LOG_DEBUG) std::cout << "-- Mean squared symmetric transfer error: " << bestSubset.meanSquaredSymmeticTransferError << ", rel. Error change: "<< dError << std::endl;

        if(fabs(dError) <= MAX_ERROR_CHANGE) stableSolutions++;
        else stableSolutions = 0;

        if(LOG_DEBUG) std::cout << "-- Stable solutions: " << stableSolutions << std::endl;

    } while(stableSolutions < 3 && bestSubset.meanSquaredSymmeticTransferError > 0.001);

    bestSubset.lineCorrespondencies = goodLineMatches;
    computeHomography(bestSubset);
    levenbergMarquardt(bestSubset);

    if(LOG_DEBUG) std::cout << "-- Final number of used matches: " << bestSubset.lineCorrespondencies.size() << std::endl;
    result = bestSubset;

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

    //std::cout << "x input: " << std::endl << x << std::endl;

    LineFunctor functor;
    functor.estimator = this;
    functor.lines = &bestSubset;
    Eigen::NumericalDiff<LineFunctor> numDiff(functor, 1.0e-6); //epsilon
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<LineFunctor>,double> lm(numDiff);

    //FunctorNumericalLineDiff functor(1.0e-3);
    //Eigen::LevenbergMarquardt<FunctorNumericalLineDiff> lm(functor);

    lm.parameters.ftol = 1.0e-15;
    lm.parameters.xtol = 1.0e-15;
    //lm.parameters.epsfcn = 1.0e-3;
    lm.parameters.maxfev = 4000; // Max iterations
    Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(x);
    //std::cout << "LMA status: " << status << std::endl;

    //std::cout << "maxfev: " << lm.parameters.maxfev << std::endl;
    if (LOG_DEBUG) std::cout << "-- LMA Iterations: " << lm.nfev << ", Status: " << status << std::endl;

    //std::cout << "LMA info: " << lm.info() << std::endl;

    //std::cout << "x that minimizes the function: " << std::endl << x << std::endl;

    //std::cout << "Hs: " << std::endl << bestSubset.Hs << std::endl;

    bestSubset.Hs.at<double>(0,0) = x(0);
    bestSubset.Hs.at<double>(0,1) = x(1);
    bestSubset.Hs.at<double>(0,2) = x(2);

    bestSubset.Hs.at<double>(1,0) = x(3);
    bestSubset.Hs.at<double>(1,1) = x(4);
    bestSubset.Hs.at<double>(1,2) = x(5);

    bestSubset.Hs.at<double>(2,0) = x(6);
    bestSubset.Hs.at<double>(2,1) = x(7);

    double error = 0;

    for(std::vector<lineCorrespStruct>::const_iterator iter = bestSubset.lineCorrespondencies.begin(); iter != bestSubset.lineCorrespondencies.end(); ++iter) {
        error += squaredSymmeticTransferError(bestSubset.Hs, *iter);
    }

    bestSubset.meanSquaredSymmeticTransferError = error/bestSubset.lineCorrespondencies.size();

    return error;
}

lineSubsetStruct FEstimatorHLines::estimateHomography(std::vector<lineCorrespStruct> lineCorrespondencies, int method, int sets) {
    int numOfPairs = lineCorrespondencies.size();
    int numOfPairSubsets = sets;//NUM_LINE_PAIR_SUBSETS_FACTOR*numOfPairs;
    std::vector<lineSubsetStruct> subsets;
    if(LOG_DEBUG) std::cout << "-- Computing "<< numOfPairSubsets << " Homographies" << std::endl;
    //Compute H_21 from NUM_CORRESP line correspondencies
    srand(time(NULL));  //Init random generator
    for(int i = 0; i < numOfPairSubsets; i++) {
        std::vector<int> subsetsIdx;
        lineSubsetStruct subset;

        for(int j = 0; j < NUM_CORRESP; j++) {
            int subsetIdx = 0;

            do {        //Generate NUM_CORRESP uniqe random indices for line pairs where not 3 are parallel
                subsetIdx = std::rand() % numOfPairs;
            } while(!isUniqe(subsetsIdx, subsetIdx) || !hasGeneralPosition(subsetsIdx, subsetIdx, lineCorrespondencies));

            subsetsIdx.push_back(subsetIdx);
            subset.lineCorrespondencies.push_back(getlineCorrespStruct(lineCorrespondencies.at(subsetIdx)));
        }
        computeHomography(subset);
        subsets.push_back(subset);
    }

    if(method == RANSAC) return calcRANSAC(subsets, 0.6, lineCorrespondencies);
    else return calcLMedS(subsets, lineCorrespondencies);
}

bool FEstimatorHLines::computeHomography(lineSubsetStruct &subset) {
    Mat linEq = Mat::ones(2*subset.lineCorrespondencies.size(),9,CV_64FC1);
    Mat* T = normalizeLines(subset.lineCorrespondencies);
    fillHLinEq(linEq, subset.lineCorrespondencies);
    Mat A = linEq.colRange(0, linEq.cols-1);
    Mat x = -linEq.col(linEq.cols-1);
    solve(A, x, subset.Hs_normalized, DECOMP_SVD);
    subset.Hs_normalized.resize(9);
    subset.Hs_normalized = subset.Hs_normalized.reshape(1,3);
    subset.Hs_normalized.at<double>(2,2) = 1.0;
    subset.Hs = denormalize(subset.Hs_normalized, T[0], T[1]);
    return true;
}

bool FEstimatorHLines::hasGeneralPosition(std::vector<int> subsetsIdx, int newIdx, std::vector<lineCorrespStruct> lineCorrespondencies) {
    if(subsetsIdx.size() < 3) return true;
    int parallelCout = 0;
    lineCorrespStruct lc;
    lineCorrespStruct lcNew = lineCorrespondencies.at(newIdx);
    for(int i = 0; i < subsetsIdx.size(); i++) {
        lc = lineCorrespondencies.at(subsetsIdx.at(i));
        if(smallestRelAngle(lc.line1Angle, lcNew.line1Angle) < MAX_ANGLE_DIFF || smallestRelAngle(lc.line2Angle, lcNew.line2Angle) < MAX_ANGLE_DIFF) parallelCout++;
    }
    if(parallelCout >= 3) return false;
    return true;
}

bool FEstimatorHLines::isUniqe(std::vector<int> subsetsIdx, int newIdx) {
    if(subsetsIdx.size() == 0) return true;
    for(std::vector<int>::const_iterator iter = subsetsIdx.begin(); iter != subsetsIdx.end(); ++iter) {
        if(*iter == newIdx) return false;
    }
    return true;
}

double FEstimatorHLines::filterUsedLineMatches(std::vector<lineCorrespStruct> &matches, std::vector<lineCorrespStruct> usedMatches) {
    std::vector<lineCorrespStruct>::iterator it= matches.begin();
    double reduction = 1.0 - ((double)usedMatches.size())/((double)matches.size());
    while (it!=matches.end()) {
        bool remove = false;
        for(std::vector<lineCorrespStruct>::const_iterator used = usedMatches.begin(); used != usedMatches.end(); ++used) {
            if(it->id == used->id) {
                remove = true;
                break;
            }
        }
        if(remove) matches.erase(it);
        else {
            it++;
        }
    }
    if(LOG_DEBUG) std::cout << "-- Refined number of matches: " << matches.size() <<  ", removed: " << usedMatches.size() << " Ratio: " << reduction << std::endl;
    return reduction;
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
    for(std::vector<lineSubsetStruct>::iterator it = subsets.begin() ; it != subsets.end(); ++it) {
        Mat H_T = it->Hs.t();
        Mat H_invT = it->Hs.inv(DECOMP_SVD).t();
        it->qualityMeasure = 0;       //count inlainers
        for(int i = 0; i < lineCorrespondencies.size(); i++) {
            error = squaredSymmeticTransferError(H_invT, H_T, lineCorrespondencies.at(i));
            if(error <= threshold) {
                it->meanSquaredSymmeticTransferError += error;
                it->qualityMeasure++;
            }
        }
        it->meanSquaredSymmeticTransferError /= it->qualityMeasure;
        if(it->qualityMeasure > bestSolution.qualityMeasure) bestSolution = *it;
    }
    if(LOG_DEBUG) std::cout << "-- RANSAC inlaiers: " << bestSolution.qualityMeasure << std::endl;
    return bestSolution;
}

lineSubsetStruct FEstimatorHLines::calcLMedS(std::vector<lineSubsetStruct> &subsets, std::vector<lineCorrespStruct> lineCorrespondencies) {
    if(LOG_DEBUG) std::cout << "-- Computing LMedS of " << subsets.size() << " Homographies" << std::endl;
    std::vector<lineSubsetStruct>::iterator it = subsets.begin();
    lineSubsetStruct lMedSsubset = *it;
    lMedSsubset.qualityMeasure = calcMedS(*it, lineCorrespondencies);
    if(subsets.size() < 2) return lMedSsubset;
    it++;
    do {
        it->qualityMeasure = calcMedS(*it, lineCorrespondencies);
        //std::cout << meds << std::endl;
        if(it->qualityMeasure < lMedSsubset.qualityMeasure) {
            lMedSsubset = *it;
        }
        it++;
    } while(it != subsets.end());

    if(LOG_DEBUG) std::cout << "-- LMEDS: " << lMedSsubset.qualityMeasure << std::endl;
    return lMedSsubset;
}


double FEstimatorHLines::calcMedS(lineSubsetStruct &subset, std::vector<lineCorrespStruct> lineCorrespondencies) {
    Mat H_invT = subset.Hs.inv(DECOMP_SVD).t();
    Mat H_T = subset.Hs.t();
    std::vector<double> errors;
    double error;
    for(std::vector<lineCorrespStruct>::iterator it = lineCorrespondencies.begin() ; it != lineCorrespondencies.end(); ++it) {
        error = squaredSymmeticTransferError(H_invT, H_T, *it);
        errors.push_back(error);
        subset.meanSquaredSymmeticTransferError += error;
    }
    subset.meanSquaredSymmeticTransferError /= lineCorrespondencies.size();
    std::sort(errors.begin(), errors.end());
    return errors.at(errors.size()/2);
}

double FEstimatorHLines::squaredSymmeticTransferError(Mat H, lineCorrespStruct lc) {
    Mat H_invT = H.inv(DECOMP_SVD).t();
    Mat H_T = H.t();
    return squaredSymmeticTransferError(H_invT, H_T, lc);
}

double FEstimatorHLines::squaredSymmeticTransferError(Mat H_invT, Mat H_T, lineCorrespStruct lc) {
    Mat A = H_T*crossProductMatrix(lc.line2Start)*lc.line2End;
    Mat start1 = lc.line1Start.t()*A;
    Mat end1 = lc.line1End.t()*A;
    Mat B = H_invT*crossProductMatrix(lc.line1Start)*lc.line1End;
    Mat start2 = lc.line2Start.t()*B;
    Mat end2 = lc.line2End.t()*B;
    Mat result = (start1*start1 + end1*end1)/(A.at<double>(0,0)*A.at<double>(0,0) + A.at<double>(1,0)*A.at<double>(1,0)) + (start2*start2 + end2*end2)/(B.at<double>(0,0)*B.at<double>(0,0) + B.at<double>(1,0)*B.at<double>(1,0));
    return result.at<double>(0,0);
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

    for (std::vector<lineCorrespStruct>::iterator it= correspondencies.begin(); it!=correspondencies.end(); ++it) {

        it->line1StartNormalized = normalizationMats[0]*it->line1Start;
        it->line2StartNormalized = normalizationMats[1]*it->line2Start;

        it->line1EndNormalized = normalizationMats[0]*it->line1End;
        it->line2EndNormalized = normalizationMats[1]*it->line2End;

    }

    return normalizationMats;
}

bool compareLineCorrespErrors(lineCorrespSubsetError ls1, lineCorrespSubsetError ls2) {
    return ls1.lineCorrespError < ls2.lineCorrespError;
}
