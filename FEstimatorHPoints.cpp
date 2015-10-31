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

int FEstimatorHPoints::extractMatches() {
    extractPointMatches(image_1, image_2, allMatchedPoints);

    for(std::vector<pointCorrespStruct>::const_iterator it = allMatchedPoints.begin() ; it != allMatchedPoints.end(); ++it) {
        if(it->isGoodMatch) {
            goodMatchedPoints.push_back(getPointCorrespStruct(*it));
            goodMatchedPointsConst.push_back(getPointCorrespStruct(*it));
        }
        allMatchedPointsConst.push_back(getPointCorrespStruct(*it));
    }

    if(LOG_DEBUG) std::cout << "-- Number of good matches: " << goodMatchedPoints.size() << std::endl;

    if(CREATE_DEBUG_IMG) visualizePointMatches(image_1_color, image_2_color, goodMatchedPoints, 20 , 2, false, name+": good point matches");
}

bool FEstimatorHPoints::compute() {

    extractMatches();

    if(LOG_DEBUG) std::cout << "-- First Estimation..."<< std::endl;

    pointSubsetStruct firstEstimation;
    pointSubsetStruct secondEstimation;

    std::vector<pointCorrespStruct> ransacInleirH1, ransacInleirH2;

    if(!findPointHomography(firstEstimation, goodMatchedPoints, allMatchedPoints,ransacInleirH1, RANSAC, RANSAC_CONFIDENCE, HOMOGRAPHY_OUTLIERS, INLIER_THRESHOLD)) {
        if(LOG_DEBUG) std::cout << "-- Estimation FAILED!" << std::endl;
        return false;
    }

    if(CREATE_DEBUG_IMG) {
        visualizePointMatches(image_1_color, image_2_color, firstEstimation.pointCorrespondencies, 20, 2, false, name+": H1 used Matches");
        visualizeHomography(firstEstimation.Hs, image_1_color, image_2_color, name+": H1");
    }

    bool homographies_equal = true;
    int removed = filterUsedPointMatches(goodMatchedPoints, firstEstimation.removePointCorrespondencies);
    filterUsedPointMatches(allMatchedPoints, firstEstimation.removePointCorrespondencies);
    double outliers = computeRelativeOutliers(HOMOGRAPHY_OUTLIERS, goodMatchedPoints.size(), goodMatchedPoints.size() + removed);
    Mat H;
    Mat e2;
    int estCnt = 1;

    while(homographies_equal && MAX_H2_ESTIMATIONS > estCnt) {

        if(LOG_DEBUG) std::cout << "-- Second estimation " << estCnt << "/" << MAX_H2_ESTIMATIONS << "..." << std::endl;

        if(!findPointHomography(secondEstimation, goodMatchedPoints, allMatchedPoints, ransacInleirH2, RANSAC, RANSAC_CONFIDENCE, outliers, INLIER_THRESHOLD)) {
            if(LOG_DEBUG) std::cout << "-- Estimation FAILED!" << std::endl;
            return false;
        }

        H = firstEstimation.Hs*secondEstimation.Hs.inv(DECOMP_SVD); // H = (H1*H2⁻1)
        homogMat(H);

        homographies_equal = (!computeUniqeEigenvector(H, e2));
        if(homographies_equal) {
            if(LOG_DEBUG) std::cout << "-- Homographies equal, repeating estimation..." << std::endl << "-- H = " << std::endl << H << std::endl;
            removed = filterUsedPointMatches(goodMatchedPoints, secondEstimation.removePointCorrespondencies);
            filterUsedPointMatches(allMatchedPoints, secondEstimation.removePointCorrespondencies);
            outliers = computeRelativeOutliers(outliers, goodMatchedPoints.size(), goodMatchedPoints.size() + removed);
        }

        estCnt++;
    }

    if(homographies_equal) {     //Not able to find a second homography
        if(LOG_DEBUG) std::cout << "-- Estimation failed!" << std::endl;
        return false;
    }

    if(CREATE_DEBUG_IMG) {
        visualizePointMatches(image_1_color, image_2_color, secondEstimation.pointCorrespondencies, 20, 2, false, name+": H2 used Matches");
        visualizeHomography(secondEstimation.Hs, image_1_color, image_2_color, name+": H2");
    }

    if(LOG_DEBUG) std::cout << "-- Used matches: " << firstEstimation.pointCorrespondencies.size() << std::endl;


    for(std::vector<pointCorrespStruct>::const_iterator pointIter = firstEstimation.pointCorrespondencies.begin(); pointIter != firstEstimation.pointCorrespondencies.end(); ++pointIter) {
        featuresImg1.push_back(matVector(pointIter->x1));
        featuresImg2.push_back(matVector(pointIter->x2));
    }

    for(std::vector<pointCorrespStruct>::const_iterator pointIter = secondEstimation.pointCorrespondencies.begin(); pointIter != secondEstimation.pointCorrespondencies.end(); ++pointIter) {
        featuresImg1.push_back(matVector(pointIter->x1));
        featuresImg2.push_back(matVector(pointIter->x2));
    }

    if(LOG_DEBUG) std::cout << "-- Added " << featuresImg1.size() << " point correspondencies to combined feature vector" << std::endl;

    F = crossProductMatrix(e2)*firstEstimation.Hs;
    enforceRankTwoConstraint(F);

    matToFile("H1_points.csv", firstEstimation.Hs);
    matToFile("H2_points.csv", secondEstimation.Hs);
    matToFile("F_H_points.csv", F);

    if(LOG_DEBUG) std::cout << "-- Used matches: " << secondEstimation.pointCorrespondencies.size() << std::endl;

    featureCountGood = goodMatchedPointsConst.size();
    featureCountComplete = allMatchedPointsConst.size();
    inlierCountOwnGood = ransacInleirH1.size() + ransacInleirH2.size();
    inlierCountOwnComplete = featuresImg1.size() + featuresImg2.size();
    //inlierCountCombined = -1;

    std::vector<Mat> goodPointsX1, goodPointsX2, allPointsX1, allPointsX2, goodInlierX1, goodInlierX2;

    for(std::vector<pointCorrespStruct>::const_iterator pointIter = goodMatchedPointsConst.begin(); pointIter != goodMatchedPointsConst.end(); ++pointIter) {
        goodPointsX1.push_back(matVector(pointIter->x1));
        goodPointsX2.push_back(matVector(pointIter->x2));
    }

    for(std::vector<pointCorrespStruct>::const_iterator pointIter = allMatchedPointsConst.begin(); pointIter != allMatchedPointsConst.end(); ++pointIter) {
        allPointsX1.push_back(matVector(pointIter->x1));
        allPointsX2.push_back(matVector(pointIter->x2));
    }

    for(std::vector<pointCorrespStruct>::const_iterator pointIter = ransacInleirH1.begin(); pointIter != ransacInleirH1.end(); ++pointIter) {
        goodInlierX1.push_back(matVector(pointIter->x1));
        goodInlierX2.push_back(matVector(pointIter->x2));
    }

    for(std::vector<pointCorrespStruct>::const_iterator pointIter = ransacInleirH2.begin(); pointIter != ransacInleirH2.end(); ++pointIter) {
        goodInlierX1.push_back(matVector(pointIter->x1));
        goodInlierX2.push_back(matVector(pointIter->x2));
    }

    if(compareWithGroundTruth) {

        trueFeatureCountGood = goodMatchesCount(Fgt, goodPointsX1, goodPointsX2, INLIER_THRESHOLD);
        trueFeatureCountComplete = goodMatchesCount(Fgt, allPointsX1, allPointsX2, INLIER_THRESHOLD);
        trueInlierCountOwnGood = goodMatchesCount(Fgt, goodInlierX1, goodInlierX2, INLIER_THRESHOLD);
        trueInlierCountOwnComplete = goodMatchesCount(Fgt, featuresImg1, featuresImg2, INLIER_THRESHOLD);
    }

    sampsonErrOwn = sampsonDistanceFundamentalMat(F, featuresImg1, featuresImg2);
    sampsonErrComplete = sampsonDistanceFundamentalMat(F, allPointsX1, allPointsX2);

    successful = true;

    if(LOG_DEBUG) std::cout << std::endl << std::endl;

    return successful;
}

bool FEstimatorHPoints::findPointHomography(pointSubsetStruct &bestSubset, std::vector<pointCorrespStruct> goodMatches, std::vector<pointCorrespStruct> allMatches, std::vector<pointCorrespStruct> &ransacInlier, int method, double confidence, double outliers, double threshold) {
    int N;
    std::vector<pointCorrespStruct> goodPointMatches;
    bestSubset.subsetError = 0;

    if(LOG_DEBUG) std::cout << "-- findPointHomography: confidence = " << confidence << ", relative outliers = " << outliers << std::endl;

    goodPointMatches.clear();
    for(std::vector<pointCorrespStruct>::const_iterator it = goodMatches.begin() ; it != goodMatches.end(); ++it) {
        goodPointMatches.push_back(*it);
    }

    if(goodPointMatches.size() < NUM_POINT_CORRESP) {
        if(LOG_DEBUG) std::cout << "-- To few point matches left! ";
        if(LOG_DEBUG) std::cout << "Can't compute Homography." << std::endl;
        return false;
    }

    N = computeNumberOfEstimations(confidence, outliers, NUM_POINT_CORRESP);
    if(!estimateHomography(bestSubset, goodPointMatches, method, N, threshold)) {
        if(LOG_DEBUG) std::cout << "-- Not enough or only colinear points left! ";
        if(LOG_DEBUG) std::cout << "Can't compute Homography." << std::endl;
        return false;
    }

    ransacInlier.clear();
    for(std::vector<pointCorrespStruct>::const_iterator it = bestSubset.pointCorrespondencies.begin() ; it != bestSubset.pointCorrespondencies.end(); ++it) {
        ransacInlier.push_back(getPointCorrespStruct(*it));
    }

    int iterationLM = 0;
    double lastError = 0;
    double dError = 0;
    int removedMatches = 0;

    pointSubsetStruct LMSubset;
    LMSubset.Hs = bestSubset.Hs.clone();
    Mat H_inv = LMSubset.Hs.inv(DECOMP_SVD);
    LMSubset.pointCorrespondencies.clear();
    for(std::vector<pointCorrespStruct>::iterator pointIter = allMatches.begin(); pointIter != allMatches.end(); ++pointIter) {
        double e = sampsonDistanceHomography_(LMSubset.Hs, H_inv, *pointIter);
        if(sqrt(e) <= threshold) {        //errorThr
            LMSubset.pointCorrespondencies.push_back(*pointIter);
            lastError += e;
        }
    }
    lastError /= LMSubset.pointCorrespondencies.size();

    double errTher = threshold;
    double bestSubsetErrThr = threshold;

    do {

        if(LMSubset.pointCorrespondencies.size() <= NUMERICAL_OPTIMIZATION_MIN_MATCHES) {
            if(LMSubset.pointCorrespondencies.size() < 4) return false;
            else break;
        }

        iterationLM++;

        if(LOG_DEBUG)  std::cout << "-- Numeric optimization iteration: " << iterationLM << "/" << NUMERICAL_OPTIMIZATION_MAX_ITERATIONS << ", error threshold for inliers: " << errTher << std::endl;

        levenbergMarquardt(LMSubset);

        errTher = threshold - 0.2*iterationLM;
        //errTher = 1.0;
        if(errTher < 2.0) errTher = 2.0;

        removedMatches = LMSubset.pointCorrespondencies.size();

        bool onlyColinearCorresp = true;
        LMSubset.subsetError = 0;
        Mat H_inv = LMSubset.Hs.inv(DECOMP_SVD);
        LMSubset.pointCorrespondencies.clear();
        for(std::vector<pointCorrespStruct>::iterator pointIter = allMatches.begin(); pointIter != allMatches.end(); ++pointIter) {
            double e = sampsonDistanceHomography_(LMSubset.Hs, H_inv, *pointIter);
            if(sqrt(e) <= errTher) {        //errorThr
                if(onlyColinearCorresp) onlyColinearCorresp = isColinear(LMSubset.pointCorrespondencies, *pointIter);
                LMSubset.pointCorrespondencies.push_back(*pointIter);
                LMSubset.subsetError += e;
            }
        }
        LMSubset.subsetError /= LMSubset.pointCorrespondencies.size();

        removedMatches = removedMatches - LMSubset.pointCorrespondencies.size();

        dError = (lastError - LMSubset.subsetError)/LMSubset.subsetError;
        if(LOG_DEBUG) std::cout << "-- Mean squared error: " << LMSubset.subsetError << ", rel. Error change: "<< dError << ", num Matches: " << LMSubset.pointCorrespondencies.size() << std::endl;

        if(dError < 0 || iterationLM == NUMERICAL_OPTIMIZATION_MAX_ITERATIONS || onlyColinearCorresp) break;

        bestSubset.Hs = LMSubset.Hs.clone();
        bestSubsetErrThr = errTher;

        lastError = LMSubset.subsetError;

    } while((dError > MAX_ERROR_CHANGE) && abs(removedMatches) > 0);

    bestSubset.subsetError = 0;
    bestSubset.pointCorrespondencies.clear();
    H_inv = bestSubset.Hs.inv(DECOMP_SVD);
    for(std::vector<pointCorrespStruct>::iterator pointIter = allMatches.begin(); pointIter != allMatches.end(); ++pointIter) {
        double error = sampsonDistanceHomography_(bestSubset.Hs, H_inv, *pointIter);
        if(sqrt(error) <= bestSubsetErrThr) {
            bestSubset.pointCorrespondencies.push_back(*pointIter);
            bestSubset.subsetError += error;
        }
        if(sqrt(error) <= 2*threshold) {
            bestSubset.removePointCorrespondencies.push_back(*pointIter);
        }
    }
    bestSubset.subsetError /= bestSubset.pointCorrespondencies.size();

    if(LOG_DEBUG) std::cout << "-- Final number of used matches: " << bestSubset.pointCorrespondencies.size() << ", Mean squared error: " << bestSubset.subsetError << std::endl;

    return true;
}

bool FEstimatorHPoints::estimateHomography(pointSubsetStruct &result, std::vector<pointCorrespStruct> pointCorresp, int method, int sets, double errorThr) {
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

    //if(method == RANSAC)
    result = calcRANSAC(subsets, errorThr, pointCorresp);
    if(result.pointCorrespondencies.size() >= 4) return true;
    return false;
}

void FEstimatorHPoints::computeHomography(pointSubsetStruct &subset) {     //See hartley, Ziss p89

    Mat* norm = normalizePoints(subset.pointCorrespondencies);

    Mat A = Mat::zeros(subset.pointCorrespondencies.size()*2, 9, CV_64FC1);

    for(int i = 0; i < subset.pointCorrespondencies.size(); i++) {      //Stack point correspondencies
        pointCorrespStruct pc = subset.pointCorrespondencies.at(i);

        //0^t

        Mat term = -pc.x2norm.at<double>(2,0)*pc.x1norm;
        A.at<double>(2*i, 3) = term.at<double>(0,0);
        A.at<double>(2*i, 4) = term.at<double>(1,0);
        A.at<double>(2*i, 5) = term.at<double>(2,0);

        term = pc.x2norm.at<double>(1,0)*pc.x1norm;
        A.at<double>(2*i, 6) = term.at<double>(0,0);
        A.at<double>(2*i, 7) = term.at<double>(1,0);
        A.at<double>(2*i, 8) = term.at<double>(2,0);

        term = pc.x2norm.at<double>(2,0)*pc.x1norm;
        A.at<double>(2*i+1, 0) = term.at<double>(0,0);
        A.at<double>(2*i+1, 1) = term.at<double>(1,0);
        A.at<double>(2*i+1, 2) = term.at<double>(2,0);

        //0^t

        term = -pc.x2norm.at<double>(0,0)*pc.x1norm;
        A.at<double>(2*i+1, 6) = term.at<double>(0,0);
        A.at<double>(2*i+1, 7) = term.at<double>(1,0);
        A.at<double>(2*i+1, 8) = term.at<double>(2,0);
    }

    SVD svd;
    svd.solveZ(A, subset.Hs_normalized);
    subset.Hs_normalized = subset.Hs_normalized.reshape(1,3);
    homogMat(subset.Hs_normalized);
    subset.Hs = denormalize(subset.Hs_normalized, norm[0], norm[1]);
    homogMat(subset.Hs);
}

pointSubsetStruct FEstimatorHPoints::calcRANSAC(std::vector<pointSubsetStruct> &subsets, double threshold, std::vector<pointCorrespStruct> pointCorresp) {
    if(LOG_DEBUG) std::cout << "-- Computing RANSAC of " << subsets.size() << " Homographies" << std::endl;
    pointSubsetStruct bestSolution = *subsets.begin();
    bestSolution.qualityMeasure = 0;
    bestSolution.subsetError = 0;
    double error = 0;
    for(std::vector<pointSubsetStruct>::iterator it = subsets.begin() ; it != subsets.end(); ++it) {
        it->qualityMeasure = 0;       //count inlainers
        it->subsetError = 0;
        Mat H_inv = it->Hs.inv(DECOMP_SVD);
        for(std::vector<pointCorrespStruct>::iterator pointIter = pointCorresp.begin(); pointIter != pointCorresp.end(); ++pointIter) {
            error = sampsonDistanceHomography_(it->Hs, H_inv, *pointIter);
            if(sqrt(error) <= threshold) {
                it->subsetError += error;
                it->qualityMeasure++;
            }
        }
        it->subsetError /= it->qualityMeasure;
        if(it->qualityMeasure > bestSolution.qualityMeasure)
            bestSolution = *it;
    }

    //return bestSolution;

    if(bestSolution.qualityMeasure > 4) {

        Mat H_inv = bestSolution.Hs.inv(DECOMP_SVD);
        bestSolution.pointCorrespondencies.clear();
        for(std::vector<pointCorrespStruct>::iterator pointIter = pointCorresp.begin(); pointIter != pointCorresp.end(); ++pointIter) {
            error = sampsonDistanceHomography_(bestSolution.Hs, H_inv, *pointIter);
            if(sqrt(error) <= threshold) {
                bestSolution.pointCorrespondencies.push_back(getPointCorrespStruct(*pointIter));
            }
        }

        computeHomography(bestSolution);

    }

    if(LOG_DEBUG) std::cout << "-- RANSAC inlaiers: " << bestSolution.pointCorrespondencies.size() << ", error: " << bestSolution.subsetError << std::endl;
    return bestSolution;
}

double FEstimatorHPoints::sampsonDistanceHomography_(Mat H, std::vector<pointCorrespStruct> pointCorresp) {
    double error = 0;
    Mat H_inv = H.inv(DECOMP_SVD);
    for(std::vector<pointCorrespStruct>::const_iterator pointIter = pointCorresp.begin(); pointIter != pointCorresp.end(); ++pointIter) {
        //error += sampsonDistanceHomography_(H, *pointIter);
        error += sampsonDistanceHomography_(H, H_inv, *pointIter);
    }
    if(pointCorresp.size() == 0) return 0;
    return error/pointCorresp.size();
}

double FEstimatorHPoints::sampsonDistanceHomography_(Mat H, Mat H_inv, pointCorrespStruct pointCorresp) {
    return sampsonDistanceHomography(H, H_inv, matVector(pointCorresp.x1), matVector(pointCorresp.x2));
}

int FEstimatorHPoints::filterUsedPointMatches(std::vector<pointCorrespStruct> &pointCorresp, std::vector<pointCorrespStruct> usedPointCorresp) {
    std::vector<pointCorrespStruct>::iterator it= pointCorresp.begin();
    int removed = 0;
    while (it!=pointCorresp.end()) {
        bool remove = false;
        for(std::vector<pointCorrespStruct>::const_iterator used = usedPointCorresp.begin(); used != usedPointCorresp.end(); ++used) {
            if(it->id == used->id) {
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
    int removed = 0;
    Mat H_inv = subset.Hs.inv(DECOMP_SVD);
    std::vector<pointCorrespStruct>::iterator it= pointCorresp.begin();
    while (it!=pointCorresp.end()) {
        if(sqrt(sampsonDistanceHomography_(subset.Hs, H_inv, *it)) > threshold) {
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

    lm.parameters.ftol = 1.0e-10;
    lm.parameters.xtol = 1.0e-10;
    lm.parameters.maxfev = MAX_LM_ITER; // Max iterations
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

    return bestSubset.subsetError;
}

Mat* FEstimatorHPoints::normalizePoints(std::vector<pointCorrespStruct> &correspondencies) {

    //Normalization:

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
