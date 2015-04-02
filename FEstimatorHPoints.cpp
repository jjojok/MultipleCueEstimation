#include "FEstimatorHPoints.h"

FEstimatorHPoints::FEstimatorHPoints(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name) {
    image_1 = img1.clone();
    image_2 = img2.clone();
    image_1_color = img1_c.clone();
    image_2_color = img2_c.clone();
    this->name = name;
    if(LOG_DEBUG) std::cout << "Estimating: " << name << std::endl;
    successful = false;

    computationType = F_FROM_POINTS_VIA_H;

    normT1 = Mat::eye(3,3,CV_64FC1);
    normT2 = Mat::eye(3,3,CV_64FC1);
}

int FEstimatorHPoints::extractMatches() {
    std::vector<Point2d> x1, x2;
    extractPointMatches(image_1, image_2, x1, x2);

    for(int i = 0; i < x1.size(); i++) {
        pointCorrespStruct *pc = new pointCorrespStruct;
        pc->id = i;
        pc->x1 = x1.at(i);
        pc->x2 = x2.at(i);
        pointCorrespondencies.push_back(*pc);
    }

    Mat* T = normalizePoints(pointCorrespondencies);
    normT1 = T[0].clone();
    normT2 = T[1].clone();

    visualizePointMatches(image_1_color, image_2_color, x1, x2, 3, true, "Point matches F_HPoints");
}

bool FEstimatorHPoints::compute() {

    extractMatches();

    if(LOG_DEBUG) std::cout << "-- First Estimation..."<< std::endl;

    //Mat H1 = findHomography(x1, x2, LMEDS, RANSAC_THREDHOLD, mask, 2000, RANSAC_CONFIDENCE);
    pointSubsetStruct firstEstimation;
    pointSubsetStruct secondEstimation;

    std::vector<Point2d> x1_used;
    std::vector<Point2d> x2_used;


    if(!findPointHomography(firstEstimation, LMEDS, CONFIDENCE, HOMOGRAPHY_OUTLIERS)) {
        if(LOG_DEBUG) std::cout << "-- Estimation FAILED!" << std::endl;
        return false;
    }

    for(std::vector<pointCorrespStruct>::const_iterator pointIter = firstEstimation.pointCorrespondencies.begin(); pointIter != firstEstimation.pointCorrespondencies.end(); ++pointIter) {
        x1_used.push_back(pointIter->x1);
        x2_used.push_back(pointIter->x2);
        featuresImg1.push_back(matVector(pointIter->x1));
        featuresImg2.push_back(matVector(pointIter->x2));
    }

    if(VISUAL_DEBUG) {
        visualizePointMatches(image_1_color, image_2_color, x1_used, x2_used, 3, true, "Point Homography 1 matches");
        visualizeHomography(firstEstimation.Hs, image_1_color, image_2_color, "Point homography 1");
    }

    if(LOG_DEBUG) std::cout << "-- Used matches: " << x1_used.size() << std::endl;

    int estCnt = 0;
    bool homographies_equal = true;
    int removed = filterUsedPointMatches(pointCorrespondencies, firstEstimation.pointCorrespondencies);
    double outliers = computeRelativeOutliers(HOMOGRAPHY_OUTLIERS, pointCorrespondencies.size(), pointCorrespondencies.size() + removed);
    Mat H;
    Mat e2;

    while(homographies_equal) {

        if(LOG_DEBUG) std::cout << "-- Second Estimation..."<< std::endl;

        estCnt++;

        if(!findPointHomography(secondEstimation, RANSAC, CONFIDENCE, outliers)) {
            if(LOG_DEBUG) std::cout << "-- Estimation FAILED!" << std::endl;
            return false;
        }

        H = firstEstimation.Hs*secondEstimation.Hs.inv(DECOMP_SVD); // H = (H1*H2⁻1)

        homographies_equal = (!computeUniqeEigenvector(H, e2) || isUnity(H));
        if(homographies_equal) {
            if(LOG_DEBUG) std::cout << "-- Homographies equal, repeating estimation..." << std::endl << "-- H = " << std::endl << H << std::endl;
            outliers = computeRelativeOutliers(outliers, secondEstimation.pointCorrespondencies.size(), pointCorrespondencies.size());
            filterUsedPointMatches(pointCorrespondencies, secondEstimation.pointCorrespondencies);
        }

        if(MAX_H2_ESTIMATIONS < estCnt) {   //Not able to find a second homographie
            if(LOG_DEBUG) std::cout << "-- Estimation failed!" << std::endl;
            return false;
        }
    }

    std::vector<Point2d> x1_used_temp;    //debug only
    std::vector<Point2d> x2_used_temp;

    for(std::vector<pointCorrespStruct>::const_iterator pointIter = secondEstimation.pointCorrespondencies.begin(); pointIter != secondEstimation.pointCorrespondencies.end(); ++pointIter) {
        x1_used.push_back(pointIter->x1);
        x2_used.push_back(pointIter->x2);
        x1_used_temp.push_back(pointIter->x1);
        x2_used_temp.push_back(pointIter->x2);
        featuresImg1.push_back(matVector(pointIter->x1));
        featuresImg2.push_back(matVector(pointIter->x2));
    }

    if(VISUAL_DEBUG) {
        visualizePointMatches(image_1_color, image_2_color, x1_used_temp, x2_used_temp, 3, true, "Point Homography 2 matches");
        visualizeHomography(secondEstimation.Hs, image_1_color, image_2_color, "Point homography 2");
    }

    F = crossProductMatrix(e2)*firstEstimation.Hs;
    enforceRankTwoConstraint(F);

    if(LOG_DEBUG) std::cout << "-- Used matches: " << x1_used_temp.size() << std::endl;

    successful = true;

    error = sampsonFDistance(F, x1_used, x2_used);

    //if(VISUAL_DEBUG) visualizeMatches(image_1_color, image_2_color, x1_used, x2_used, 3, true, "Point Homography used point matches");

    if(LOG_DEBUG) std::cout << std::endl << std::endl;

    return successful;
}

bool FEstimatorHPoints::findPointHomography(pointSubsetStruct &bestSubset, int method, double confidence, double outliers) {
    double lastError = 0;
    int N;
    std::vector<pointCorrespStruct> goodMatches;
    std::vector<pointCorrespStruct> lastIterLineMatches;
    bestSubset.subsetError = 0;
    int iteration = 0;
    int stableSolutions = 0;
    double dError = 0;
    int removedMatches = 0;

    if(LOG_DEBUG) std::cout << "-- findPointHomography: confidence = " << confidence << ", relative outliers = " << outliers << std::endl;

    goodMatches.clear();
    for(std::vector<pointCorrespStruct>::const_iterator it = pointCorrespondencies.begin() ; it != pointCorrespondencies.end(); ++it) {
        goodMatches.push_back(*it);
    }

    do {

        iteration++;

        if(LOG_DEBUG) std::cout << "-- Iteration: " << iteration <<"/" << MAX_NUMERICAL_OPTIMIZATION_ITERATIONS << ", Used number of matches: " << goodMatches.size() << std::endl;

        if(goodMatches.size() < NUM_POINT_CORRESP) {
            if(LOG_DEBUG) std::cout << "-- To few line matches left! ";
            if(iteration == 1) {
                if(LOG_DEBUG) std::cout << "Can't compute Homography." << std::endl;
                return false;
            }
            if(LOG_DEBUG) std::cout << "Using second best solution..." << std::endl;
            goodMatches = lastIterLineMatches;
            break;
        }

        lastError = bestSubset.subsetError;

        outliers = computeRelativeOutliers(outliers, goodMatches.size(), goodMatches.size() + removedMatches);
        N = computeNumberOfEstimations(confidence, outliers, NUM_POINT_CORRESP);
        if(!estimateHomography(bestSubset, goodMatches, method, N)) {
            if(LOG_DEBUG) std::cout << "-- Only colinear points left! ";
            if(iteration == 1) {
                if(LOG_DEBUG) std::cout << "Can't compute Homography." << std::endl;
                return false;
            }
            if(LOG_DEBUG) std::cout << "Using second best solution..." << std::endl;
            goodMatches = lastIterLineMatches;
            break;
        }

        levenbergMarquardt(bestSubset);

        lastIterLineMatches.clear();
        for(int i = 0; i < goodMatches.size(); i++) {
            lastIterLineMatches.push_back(goodMatches.at(i));
        }

        removedMatches = filterBadPointMatches(bestSubset, goodMatches, MAX_TRANSFER_DIST/(iteration));

        if(iteration == MAX_NUMERICAL_OPTIMIZATION_ITERATIONS) return false;

        dError = (lastError - bestSubset.subsetError)/bestSubset.subsetError;
        if(LOG_DEBUG) std::cout << "-- Mean squared error: " << bestSubset.subsetError << ", rel. Error change: "<< dError << std::endl;

        if(fabs(dError) <= MAX_ERROR_CHANGE || removedMatches == 0) stableSolutions++;
        else stableSolutions = 0;

        if(LOG_DEBUG) std::cout << "-- Stable solutions: " << stableSolutions << std::endl;

    } while(stableSolutions < 3);

    //bestSubset.pointCorrespondencies.clear();
    for(int i = 0; i < lastIterLineMatches.size(); i++) {
        //if(!isColinear(bestSubset.pointCorrespondencies, lastIterLineMatches.at(i))) {
        if(isUniqe(bestSubset.pointCorrespondencies, lastIterLineMatches.at(i))) {
            bestSubset.pointCorrespondencies.push_back(lastIterLineMatches.at(i));
        }
    }
    //computeHomography(bestSubset);
    levenbergMarquardt(bestSubset);

    bestSubset.Hs = denormalize(bestSubset.Hs, normT1, normT2);

    if(LOG_DEBUG) std::cout << "-- Final number of used matches: " << bestSubset.pointCorrespondencies.size() << ", Mean squared error: " << bestSubset.subsetError << std::endl;

    return true;
}

bool FEstimatorHPoints::estimateHomography(pointSubsetStruct &result, std::vector<pointCorrespStruct> pointCorresp, int method, int sets) {
    int numOfPairs = pointCorresp.size();
    std::vector<pointSubsetStruct> subsets;
    if(LOG_DEBUG) std::cout << "-- Computing "<< sets << " Homographies, using " << NUM_POINT_CORRESP << " point correspondencies each" << std::endl;
    //Compute H_21 from NUM_CORRESP line correspondencies
    std::srand(time(NULL));  //Init random generator
    for(int i = 0; i < sets; i++) {
        pointSubsetStruct subset;
        for(int j = 0; j < NUM_POINT_CORRESP; j++) {
            int subsetIdx = 0;
            int search = 0;
            do {        //Generate uniqe random indices for line pairs where not 3 are colinear
                subsetIdx = std::rand() % numOfPairs;
                search++;
                if(search == MAX_POINT_SEARCH) return false;    //No non colinear points remaining
            } while(!isUniqe(subset.pointCorrespondencies, pointCorresp.at(subsetIdx)) || isColinear(subset.pointCorrespondencies, pointCorresp.at(subsetIdx)));

            subset.pointCorrespondencies.push_back(getPointCorrespStruct(pointCorresp.at(subsetIdx)));
        }

        computeHomography(subset);

        if(subset.Hs.data) {
            subsets.push_back(subset);
        } else i--;
    }

    if(method == RANSAC) result = calcRANSAC(subsets, 0.6, pointCorresp);
    else result = calcLMedS(subsets, pointCorresp);
    return true;
}

void FEstimatorHPoints::computeHomography(pointSubsetStruct &subset) {     //See hartley, Ziss p89

    Mat A = Mat::zeros(subset.pointCorrespondencies.size()*3, 9, CV_64FC1);

    for(int i = 0; i < subset.pointCorrespondencies.size(); i++) {      //Stack point correspondencies
        pointCorrespStruct pc = subset.pointCorrespondencies.at(i);

        //0^t

        Mat term = -pc.x2norm.at<double>(2,0)*pc.x1norm;
        A.at<double>(3*i, 3) = term.at<double>(0,0);
        A.at<double>(3*i, 4) = term.at<double>(1,0);
        A.at<double>(3*i, 5) = term.at<double>(2,0);

        term = pc.x2norm.at<double>(1,0)*pc.x1norm;
        A.at<double>(3*i, 6) = term.at<double>(0,0);
        A.at<double>(3*i, 7) = term.at<double>(1,0);
        A.at<double>(3*i, 8) = term.at<double>(2,0);

        term = pc.x2norm.at<double>(2,0)*pc.x1norm;
        A.at<double>(3*i+1, 0) = term.at<double>(0,0);
        A.at<double>(3*i+1, 1) = term.at<double>(1,0);
        A.at<double>(3*i+1, 2) = term.at<double>(2,0);

        //0^t

        term = -pc.x2norm.at<double>(0,0)*pc.x1norm;
        A.at<double>(3*i+1, 6) = term.at<double>(0,0);
        A.at<double>(3*i+1, 7) = term.at<double>(1,0);
        A.at<double>(3*i+1, 8) = term.at<double>(2,0);

        term = -pc.x2norm.at<double>(1,0)*pc.x1norm;
        A.at<double>(3*i+2, 0) = term.at<double>(0,0);
        A.at<double>(3*i+2, 1) = term.at<double>(1,0);
        A.at<double>(3*i+2, 2) = term.at<double>(2,0);

        term = pc.x2norm.at<double>(0,0)*pc.x1norm;
        A.at<double>(3*i+2, 3) = term.at<double>(0,0);
        A.at<double>(3*i+2, 4) = term.at<double>(1,0);
        A.at<double>(3*i+2, 5) = term.at<double>(2,0);

        //0^t
    }

    SVD svd;
    svd.solveZ(A, subset.Hs_normalized);
    subset.Hs_normalized = subset.Hs_normalized.reshape(1,3);
    homogMat(subset.Hs_normalized);
    subset.Hs = subset.Hs_normalized.clone();

}

pointSubsetStruct FEstimatorHPoints::calcRANSAC(std::vector<pointSubsetStruct> &subsets, double threshold, std::vector<pointCorrespStruct> pointCorresp) {
    if(LOG_DEBUG) std::cout << "-- Computing RANSAC of " << subsets.size() << " Homographies" << std::endl;
    pointSubsetStruct bestSolution = *subsets.begin();
    bestSolution.qualityMeasure = 0;
    double error = 0;
    for(std::vector<pointSubsetStruct>::iterator it = subsets.begin() ; it != subsets.end(); ++it) {
        Mat H_inv = it->Hs.inv(DECOMP_SVD);
        it->qualityMeasure = 0;       //count inlainers
        for(std::vector<pointCorrespStruct>::const_iterator pointIter = pointCorresp.begin(); pointIter != pointCorresp.end(); ++pointIter) {
            error = squaredPointError(it->Hs, H_inv, *pointIter);
            if(error <= threshold) {
                it->subsetError += error;
                it->qualityMeasure++;
            }
        }
        it->subsetError /= it->qualityMeasure;
        if(it->qualityMeasure > bestSolution.qualityMeasure) bestSolution = *it;
    }
    if(LOG_DEBUG) std::cout << "-- RANSAC inlaiers: " << bestSolution.qualityMeasure << std::endl;
    return bestSolution;
}

pointSubsetStruct FEstimatorHPoints::calcLMedS(std::vector<pointSubsetStruct> &subsets, std::vector<pointCorrespStruct> pointCorresp) {
    if(LOG_DEBUG) std::cout << "-- Computing LMedS of " << subsets.size() << " Homographies" << std::endl;
    std::vector<pointSubsetStruct>::iterator it = subsets.begin();
    pointSubsetStruct lMedSsubset = *it;
    lMedSsubset.qualityMeasure = calcMedS(*it, pointCorresp);
    if(subsets.size() < 2) return lMedSsubset;
    it++;
    do {
        it->qualityMeasure = calcMedS(*it, pointCorresp);
        //std::cout << meds << std::endl;
        if(it->qualityMeasure < lMedSsubset.qualityMeasure) {
            lMedSsubset = *it;
        }
        it++;
    } while(it != subsets.end());

    if(LOG_DEBUG) std::cout << "-- LMEDS: " << lMedSsubset.qualityMeasure << std::endl;
    return lMedSsubset;
}

double FEstimatorHPoints::calcMedS(pointSubsetStruct &subset, std::vector<pointCorrespStruct> pointCorresp) {
    Mat H_inv = subset.Hs.inv(DECOMP_SVD);
    std::vector<double> errors;
    double error = 0;
    for(std::vector<pointCorrespStruct>::const_iterator pointIter = pointCorresp.begin(); pointIter != pointCorresp.end(); ++pointIter) {
        error = squaredPointError(subset.Hs, H_inv, *pointIter);
        errors.push_back(error);
        subset.subsetError += error;
    }
    subset.subsetError /= pointCorresp.size();
    std::sort(errors.begin(), errors.end());
    return errors.at(errors.size()/2);
}

double FEstimatorHPoints::meanSquaredPointError(Mat H, std::vector<pointCorrespStruct> pointCorresp) {
    Mat H_inv = H.inv(DECOMP_SVD);
    double error = 0;
    for(std::vector<pointCorrespStruct>::const_iterator pointIter = pointCorresp.begin(); pointIter != pointCorresp.end(); ++pointIter) {
        error += squaredPointError(H, H_inv, *pointIter);
    }
    return error/pointCorresp.size();
}

double FEstimatorHPoints::squaredPointError(Mat H, Mat H_inv, pointCorrespStruct pointCorresp) {
    return errorFunctionHPointsSqared(H, H_inv, pointCorresp.x1norm, pointCorresp.x2norm);
}

int FEstimatorHPoints::filterUsedPointMatches(std::vector<pointCorrespStruct> &pointCorresp, std::vector<pointCorrespStruct> usedPointCorresp) {
    std::vector<pointCorrespStruct>::iterator it= pointCorresp.begin();
    std::srand(std::time(0));
    int removed = 0;
    while (it!=pointCorresp.end()) {
        bool remove = false;
        for(std::vector<pointCorrespStruct>::const_iterator used = usedPointCorresp.begin(); used != usedPointCorresp.end(); ++used) {
            bool keep = false;//std::rand()%4 - 1;  //Delete every third correpsondency
            if(it->id == used->id && !keep) {
                removed++;
                remove = true;
                break;
            }
        }
        if(remove) pointCorresp.erase(it);
        else {
            it++;
        }
    }
    if(LOG_DEBUG) std::cout << "-- Refined number of matches: " << pointCorresp.size() <<  ", removed: " << removed << std::endl;
    return removed;
}

int FEstimatorHPoints::filterBadPointMatches(pointSubsetStruct subset, std::vector<pointCorrespStruct> &pointCorresp, double threshold) {
    Mat Hs_inv = subset.Hs.inv(DECOMP_SVD);
    int removed = 0;
    std::vector<pointCorrespStruct>::iterator it= pointCorresp.begin();
    while (it!=pointCorresp.end()) {
        if(squaredPointError(subset.Hs, Hs_inv, *it) > threshold) {
            removed++;
            pointCorresp.erase(it);
        } else it++;
    }
    if(LOG_DEBUG) std::cout << "-- Refined number of matches: " << pointCorresp.size() <<  ", removed: " << removed << ", threshold: " << threshold << std::endl;
    return removed;
}

bool FEstimatorHPoints::isColinear(std::vector<pointCorrespStruct> fixedCorresp, pointCorrespStruct pcNew) {
    if(fixedCorresp.size() < 3) return false;
    int colinearCout = 0;
    pointCorrespStruct pc1;
    pointCorrespStruct pc2;
    for(int i = 0; i < fixedCorresp.size(); i++) {
        pc1 = fixedCorresp.at(i);
        for(int j = i+1; j < fixedCorresp.size(); j++) {
            pc2 = fixedCorresp.at(j);
            double a1 = (pc1.x1.y - pc2.x1.y)/(pc1.x1.x - pc2.x1.x);
            double a2 = (pc1.x2.y - pc2.x2.y)/(pc1.x2.x - pc2.x2.x);
            double b1 = pc1.x1.y - a1*pc1.x1.x;
            double b2 = pc1.x2.y - a2*pc1.x2.x;
            if(fabs(a1*pcNew.x1.x + b1 - pcNew.x1.y) < MAX_COLINEAR_DIST && fabs(a2*pcNew.x2.x + b2 - pcNew.x2.y) < MAX_COLINEAR_DIST) colinearCout++;
            if(colinearCout >= 3) return true;
        }
    }

    return false;
}

double FEstimatorHPoints::levenbergMarquardt(pointSubsetStruct &bestSubset) {
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

    PointFunctor functor;
    functor.points = &bestSubset;
    Eigen::NumericalDiff<PointFunctor> numDiff(functor, 1.0e-6); //epsilon
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<PointFunctor>,double> lm(numDiff);

    lm.parameters.ftol = 1.0e-15;
    lm.parameters.xtol = 1.0e-15;
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

    bestSubset.subsetError = meanSquaredPointError(bestSubset.Hs, bestSubset.pointCorrespondencies);

    return error;
}

Mat* FEstimatorHPoints::normalizePoints(std::vector<pointCorrespStruct> &correspondencies) {

    //Normalization: Hartley, Zisserman, Multiple View Geometry in Computer Vision, p. 109

    Mat* normalizationMats = new Mat[2];
    double sum1x = 0, sum1y = 0, sum2x = 0, sum2y = 0, N = 0;
    double mean1x = 0, mean1y = 0, mean2x = 0, mean2y = 0, v1 = 0, v2 = 0, scale1 = 0, scale2 = 0;

    for (std::vector<pointCorrespStruct>::iterator it= correspondencies.begin(); it!=correspondencies.end(); ++it) {

        sum1x += it->x1.x;
        sum2x += it->x2.x;
        sum1y += it->x1.y;
        sum2y += it->x2.y;

    }

    normalizationMats[0] = Mat::eye(3,3, CV_64FC1);
    normalizationMats[1] = Mat::eye(3,3, CV_64FC1);
    N = correspondencies.size();

    mean1x = sum1x/N;
    mean1y = sum1y/N;
    mean2x = sum2x/N;
    mean2y = sum2y/N;

    for (std::vector<pointCorrespStruct>::iterator it= correspondencies.begin(); it!=correspondencies.end(); ++it) {
        v1 += fnorm(it->x1.x-mean1x, it->x1.y-mean1y);
        v2 += fnorm(it->x2.x-mean2x, it->x2.y-mean2y);
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

    //if(LOG_DEBUG) std::cout << "-- Normalization: " << std::endl <<"-- T1 = " << std::endl << normalizationMats[0] << std::endl << "-- T2 = " << std::endl << normalizationMats[1] << std::endl;

    //Carry out normalization:

    for (std::vector<pointCorrespStruct>::iterator it= correspondencies.begin(); it!=correspondencies.end(); ++it) {

        it->x1norm = normalizationMats[0]*matVector(it->x1);
        it->x2norm = normalizationMats[1]*matVector(it->x2);

    }

    return normalizationMats;
}

bool FEstimatorHPoints::isUniqe(std::vector<pointCorrespStruct> existingCorresp, pointCorrespStruct newCorresp) {
    if(existingCorresp.size() == 0) return true;
    for(std::vector<pointCorrespStruct>::const_iterator iter = existingCorresp.begin(); iter != existingCorresp.end(); ++iter) {
        if(iter->id == newCorresp.id) return false;
    }
    return true;
}
