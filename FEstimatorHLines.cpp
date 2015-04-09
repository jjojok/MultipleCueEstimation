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
    extractLineMatches(image_1, image_2, allMatchedLines);

    Mat* T = normalizeLines(allMatchedLines, goodMatchedLines);
    normT1 = T[0].clone();
    normT2 = T[1].clone();

    //if(VISUAL_DEBUG) visualizeMatches(image_1, image_2, allMatchedLines, 2, true, "All line matches");
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

    if(!findLineHomography(firstEstimation, goodMatchedLines, allMatchedLines, RANSAC, CONFIDENCE, HOMOGRAPHY_OUTLIERS, HOMOGRAPHY_RANSAC_THRESHOLD)) {
        if(LOG_DEBUG) std::cout << "-- Estimation FAILED!" << std::endl;
        return false;
    }

    addPointCorrespondencies(firstEstimation.Hs, firstEstimation.lineCorrespondencies);

    if(VISUAL_DEBUG) {
        visualizeHomography(firstEstimation.Hs, image_1, image_2, name+": H1");
        visualizeLineMatches(image_1_color, image_2_color, firstEstimation.lineCorrespondencies, 8, true, name+": H1 used Matches");
        //visualizeLineMatches(image_1_color, image_2_color, goodLineMatches, 8, true, name+": H1 good Matches");
        //visualizeProjectedLines(H1, 8, true, name+": H21 used lines projected to image 2");
    }

    filterUsedLineMatches(allMatchedLines, firstEstimation.lineCorrespondencies);
    int removed = filterUsedLineMatches(goodMatchedLines, firstEstimation.lineCorrespondencies);

    if(VISUAL_DEBUG) {
        visualizeLineMatches(image_1_color, image_2_color, goodMatchedLines, 8, true, name+": Remaining matches for 2ed estimation");
    }

    double outliers = computeRelativeOutliers(HOMOGRAPHY_OUTLIERS, goodMatchedLines.size(), goodMatchedLines.size() + removed);

    int estCnt = 0;
    bool homographies_equal = true;
    Mat H;
    Mat e2;

    while(homographies_equal && goodMatchedLines.size() > NUM_LINE_CORRESP && MAX_H2_ESTIMATIONS > estCnt) {

        if(LOG_DEBUG) std::cout << "-- Second estimation " << estCnt << "/" << MAX_H2_ESTIMATIONS << "..." << std::endl;

        if(!findLineHomography(secondEstimation, goodMatchedLines, allMatchedLines, RANSAC, CONFIDENCE, outliers, HOMOGRAPHY_RANSAC_THRESHOLD)) {
            if(LOG_DEBUG) std::cout << "-- Estimation FAILED!" << std::endl;
            return false;
        }

        H = firstEstimation.Hs*secondEstimation.Hs.inv(DECOMP_SVD); // H = (H1*H2⁻1)
        H /= H.at<double>(2,2);

        homographies_equal = (!computeUniqeEigenvector(H, e2) || isUnity(H));
        if(homographies_equal) {
            if(LOG_DEBUG) std::cout << "-- Homographies equal, repeating estimation..." << std::endl << "-- H = " << std::endl << H << std::endl;
            filterUsedLineMatches(allMatchedLines, secondEstimation.lineCorrespondencies);
            removed = filterUsedLineMatches(goodMatchedLines, secondEstimation.lineCorrespondencies);
            outliers = computeRelativeOutliers(outliers, goodMatchedLines.size(), goodMatchedLines.size() + removed);
        }
        estCnt++;
    }

    if(homographies_equal) {     //Not able to find a secont homographie
        if(LOG_DEBUG) std::cout << "-- Estimation failed!" << std::endl;
        return false;
    }

    addPointCorrespondencies(secondEstimation.Hs, secondEstimation.lineCorrespondencies);

    if(VISUAL_DEBUG) {
        visualizeHomography(secondEstimation.Hs, image_1, image_2, name+" H2");
        visualizeLineMatches(image_1_color, image_2_color, secondEstimation.lineCorrespondencies, 8, true, name+": H2 used Matches");
        //visualizeLineMatches(image_1_color, image_2_color, goodLineMatches, 8, true, name+": H2 good Matches");
        //visualizeProjectedLines(H2, 8, true, name+": H21_2 used lines projected to image 2");
        //visualizePointMatches(image_1_color, image_2_color, featuresImg1, featuresImg2, 3, true, name+": Line-generated Point correspondencies");
    }

    F = crossProductMatrix(e2)*firstEstimation.Hs;

    enforceRankTwoConstraint(F);

    error = sampsonFDistance(F, featuresImg1, featuresImg2);

    successful = true;

    if(LOG_DEBUG) std::cout << "-- Added " << featuresImg1.size() << " point correspondencies to combined feature vector" << std::endl;

    if(LOG_DEBUG) std::cout << std::endl << std::endl;

    return true;
}

bool FEstimatorHLines::findLineHomography(lineSubsetStruct &bestSubset, std::vector<lineCorrespStruct> goodMatches, std::vector<lineCorrespStruct> allMatches, int method, double confidence, double outliers, double threshold) {
    double lastError = 0;
    int N;
    //std::vector<lineCorrespStruct> lastIterLineMatches;
    std::vector<lineCorrespStruct> goodLineMatches;
    bestSubset.subsetError = 0;
    //int iteration = 0;
    int stableSolutions = 0;
    double dError = 0;
    int removedMatches = 0;

    if(LOG_DEBUG) std::cout << "-- findHomography: confidence = " << confidence << ", relative outliers = " << outliers << std::endl;

    goodLineMatches.clear();
    for(std::vector<lineCorrespStruct>::const_iterator it = goodMatches.begin() ; it != goodMatches.end(); ++it) {
        goodLineMatches.push_back(*it);
    }

    double errorThr = normalizeThr(normT1, normT2, threshold);

    if(goodLineMatches.size() < NUM_LINE_CORRESP) {
        if(LOG_DEBUG) std::cout << "-- To few line matches left! ";
        if(LOG_DEBUG) std::cout << "Can't compute Homography." << std::endl;
        return false;
    }

    outliers = computeRelativeOutliers(outliers, goodLineMatches.size(), goodLineMatches.size() + removedMatches);
    N = computeNumberOfEstimations(confidence, outliers, NUM_LINE_CORRESP);
    if(!estimateHomography(bestSubset, goodLineMatches, method, N, errorThr)) {
        if(LOG_DEBUG) std::cout << "-- Only colinear points left! ";
        if(LOG_DEBUG) std::cout << "Can't compute Homography." << std::endl;
        return false;
    }

    errorThr = bestSubset.subsetError;

    int iterationLM = 0;

    do {

        iterationLM++;

        if(LOG_DEBUG)  std::cout << "-- Numeric optimization iteration: " << iterationLM << "/" << NUMERICAL_OPTIMIZATION_MAX_ITERATIONS << ", error threshold for inliers: " << sqrt(errorThr)/iterationLM << std::endl;

        Mat H_T = bestSubset.Hs.t();
        Mat H_invT = bestSubset.Hs.inv(DECOMP_SVD).t();
        bestSubset.lineCorrespondencies.clear();
        for(int i = 0; i < allMatches.size(); i++) {
            lineCorrespStruct lc = allMatches.at(i);
            if(errorFunctionHLines_(H_invT, H_T, lc.line1StartNormalized, lc.line1EndNormalized, lc.line2StartNormalized, lc.line2EndNormalized) < sqrt(errorThr)/iterationLM) {        //errorThr
                bestSubset.lineCorrespondencies.push_back(lc);
            }
        }

        removedMatches = removedMatches - bestSubset.lineCorrespondencies.size();

        levenbergMarquardt(bestSubset);

        if(iterationLM == NUMERICAL_OPTIMIZATION_MAX_ITERATIONS) break;

        dError = (lastError - bestSubset.subsetError)/bestSubset.subsetError;
        if(LOG_DEBUG) std::cout << "-- Mean squared error: " << bestSubset.subsetError << ", rel. Error change: "<< dError << ", num Matches: " << bestSubset.lineCorrespondencies.size() << std::endl;

        lastError = bestSubset.subsetError;

        if((dError >=0 && dError <= MAX_ERROR_CHANGE) || removedMatches == 0) stableSolutions++;
        else stableSolutions = 0;

        if(LOG_DEBUG) std::cout << "-- Stable solutions: " << stableSolutions << std::endl;

        if(bestSubset.lineCorrespondencies.size() <= NUMERICAL_OPTIMIZATION_MIN_MATCHES) break;

    } while(stableSolutions < 3);

    if(LOG_DEBUG) std::cout << "-- Final number of used matches: " << bestSubset.lineCorrespondencies.size() << ", Mean squared error: " << bestSubset.subsetError << std::endl;

    bestSubset.Hs = denormalize(bestSubset.Hs, normT1, normT2);

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

    lm.parameters.ftol = 1.0e-15;
    lm.parameters.xtol = 1.0e-15;
    //lm.parameters.epsfcn = 1.0e-3;
    lm.parameters.maxfev = 4000; // Max iterations
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

    Mat H_T = bestSubset.Hs.t();
    Mat H_invT = bestSubset.Hs.inv(DECOMP_SVD).t();
    for(std::vector<lineCorrespStruct>::const_iterator it = bestSubset.lineCorrespondencies.begin(); it != bestSubset.lineCorrespondencies.end(); ++it) {
        error += errorFunctionHLinesSqared_(H_invT, H_T, it->line1StartNormalized, it->line1EndNormalized, it->line2StartNormalized, it->line2EndNormalized);
    }

    if(bestSubset.lineCorrespondencies.size() > 0) bestSubset.subsetError = error/bestSubset.lineCorrespondencies.size();

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
                if(search == MAX_POINT_SEARCH) return false;    //No non colinear points remaining
            } while(!isUniqe(subset.lineCorrespondencies, lineCorrespondencies.at(subsetIdx)) || isParallel(subset.lineCorrespondencies, lineCorrespondencies.at(subsetIdx)));

            subset.lineCorrespondencies.push_back(getlineCorrespStruct(lineCorrespondencies.at(subsetIdx)));
        }
        computeHomography(subset);
        subsets.push_back(subset);
    }

    if(method == RANSAC) result = calcRANSAC(subsets, ransacThr, lineCorrespondencies);
    else result = calcLMedS(subsets, lineCorrespondencies);
    return true;
}

bool FEstimatorHLines::computeHomography(lineSubsetStruct &subset) {
    Mat linEq = Mat::ones(2*subset.lineCorrespondencies.size(),9,CV_64FC1);
    fillHLinEq(linEq, subset.lineCorrespondencies);
    SVD svd;
    svd.solveZ(linEq, subset.Hs_normalized);
    subset.Hs_normalized = subset.Hs_normalized.reshape(1,3);
    homogMat(subset.Hs_normalized);
    subset.Hs = subset.Hs_normalized;
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
    std::srand(std::time(0));
    int removed = 0;
    while (it!=matches.end()) {
        bool remove = false;
        for(std::vector<lineCorrespStruct>::const_iterator used = usedMatches.begin(); used != usedMatches.end(); ++used) {
            bool keep = false;//std::rand()%3 - 1;
            if(it->id == used->id && !keep) {
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
        Mat H_T = subset->Hs.t();
        Mat H_invT = subset->Hs.inv(DECOMP_SVD).t();
        subset->qualityMeasure = 0;       //count inlainers
        for(std::vector<lineCorrespStruct>::iterator it = lineCorrespondencies.begin() ; it != lineCorrespondencies.end(); ++it) {
            error = errorFunctionHLinesSqared_(H_invT, H_T, it->line1StartNormalized, it->line1EndNormalized, it->line2StartNormalized, it->line2EndNormalized);
            if(error <= threshold) {
                subset->subsetError += error;
                subset->qualityMeasure++;
            }
        }
        subset->subsetError /= subset->qualityMeasure;
        if(subset->qualityMeasure > bestSolution.qualityMeasure) bestSolution = *subset;
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
        if(it->qualityMeasure < lMedSsubset.qualityMeasure) {
            lMedSsubset = *it;
        }
        it++;
    } while(it != subsets.end());

    lMedSsubset.subsetError = lMedSsubset.qualityMeasure;
    if(LOG_DEBUG) std::cout << "-- LMEDS: " << lMedSsubset.qualityMeasure << std::endl;
    return lMedSsubset;
}


double FEstimatorHLines::calcMedS(lineSubsetStruct &subset, std::vector<lineCorrespStruct> lineCorrespondencies) {
    Mat H_invT = subset.Hs.inv(DECOMP_SVD).t();
    Mat H_T = subset.Hs.t();
    std::vector<double> errors;
    double error;
    for(std::vector<lineCorrespStruct>::iterator it = lineCorrespondencies.begin() ; it != lineCorrespondencies.end(); ++it) {
        error = errorFunctionHLinesSqared_(H_invT, H_T, it->line1StartNormalized, it->line1EndNormalized, it->line2StartNormalized, it->line2EndNormalized);
        errors.push_back(error);
        subset.subsetError += error;
    }
    subset.subsetError /= lineCorrespondencies.size();
    std::sort(errors.begin(), errors.end());
    return errors.at(errors.size()/2);
}

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

bool compareLineCorrespErrors(lineCorrespSubsetError ls1, lineCorrespSubsetError ls2) {
    return ls1.lineCorrespError < ls2.lineCorrespError;
}

void FEstimatorHLines::addPointCorrespondencies(Mat H, std::vector<lineCorrespStruct> goodLineMatches) {
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

        featuresImg1.push_back(iter->line1Start);
        featuresImg1.push_back(iter->line1End);
        featuresImg2.push_back(iter->line2Start);
        featuresImg2.push_back(iter->line2End);

        featuresImg1.push_back(l1sProjection);
        featuresImg1.push_back(l1eProjection);
        featuresImg2.push_back(l2sProjection);
        featuresImg2.push_back(l2eProjection);
    }
}

//double FEstimatorHLines::squaredSymmeticTransferLineError(Mat H, lineCorrespStruct lc) {
//    Mat H_invT = H.inv(DECOMP_SVD).t();
//    Mat H_T = H.t();
//    return squaredSymmeticTransferLineError(H_invT, H_T, lc);
//}

double FEstimatorHLines::errorFunctionHLinesSqaredAlgebraic_(Mat H_invT, Mat H_T, Mat l1s, Mat l1e, Mat l2s, Mat l2e) {
    Mat A = (l1s.t()*H_T*crossProductMatrix(l2s)*l2e)*(l1s.t()*H_T*crossProductMatrix(l2s)*l2e) + (l1e.t()*H_T*crossProductMatrix(l2s)*l2e)*(l1e.t()*H_T*crossProductMatrix(l2s)*l2e);
    Mat C = (l2s.t()*H_invT*crossProductMatrix(l1s)*l1e)*(l2s.t()*H_invT*crossProductMatrix(l1s)*l1e) + (l2e.t()*H_invT*crossProductMatrix(l1s)*l1e)*(l2e.t()*H_invT*crossProductMatrix(l1s)*l1e);
    return std::pow(A.at<double>(0,0),2) + std::pow(C.at<double>(0,0),2);
}

double FEstimatorHLines::errorFunctionHLinesSqared_(Mat H_invT, Mat H_T, Mat l1s, Mat l1e, Mat l2s, Mat l2e) {
    return errorFunctionHLinesSqared(H_T, l1s, l1e, l2s, l2e) + errorFunctionHLinesSqared(H_invT, l2s, l2e, l1s, l1e);
}

double FEstimatorHLines::errorFunctionHLines_(Mat H_invT, Mat H_T, Mat l1s, Mat l1e, Mat l2s, Mat l2e) {
    return fabs(errorFunctionHLines(H_T, l1s, l1e, l2s, l2e)) + fabs(errorFunctionHLines(H_invT, l2s, l2e, l1s, l1e));
    //return errorFunctionHLinesSqared_(H_invT, H_T, l1s, l1e, l2s, l2e);
}

int FEstimatorHLines::filterBadLineMatches(lineSubsetStruct subset, std::vector<lineCorrespStruct> &lineCorresp, double threshold) {
    Mat H_invT = subset.Hs.inv(DECOMP_SVD).t();
    Mat H_T = subset.Hs.t();
    int removed = 0;
    std::vector<lineCorrespStruct>::iterator it= lineCorresp.begin();
    while (it!=lineCorresp.end()) {
        if(errorFunctionHLines_(H_invT, H_T, it->line1StartNormalized, it->line1EndNormalized, it->line2StartNormalized, it->line2EndNormalized) > threshold) {
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
