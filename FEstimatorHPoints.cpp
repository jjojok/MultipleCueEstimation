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
    extractPointMatches(image_1, image_2, x1, x2);
}

bool FEstimatorHPoints::compute() {

    extractMatches();

    if(LOG_DEBUG) std::cout << "-- First Estimation..."<< std::endl;

    std::vector<int> mask;
    //Mat H1 = findHomography(x1, x2, LMEDS, RANSAC_THREDHOLD, mask, 2000, RANSAC_CONFIDENCE);
    pointSubsetStruct firstEstimation;
    if(!findPointHomography(x1, x2, LMEDS, CONFIDENCE, LINE_OUTLIERS, firstEstimation)) {
        if(LOG_DEBUG) std::cout << "-- Estimation FAILED!" << std::endl;
        return false;
    }

    if(!H1.data) {
        if(LOG_DEBUG) std::cout << "-- Estimation failed!  (Unable to compute H1)" << std::endl;
        return false;
    }

    if(VISUAL_DEBUG) {
        visualizeMatches(image_1_color, image_2_color, x1_used, x2_used, 3, true, "Point Homography 1 matches");
        visualizeHomography(H1, image_1_color, image_2_color, "Point homography 1");
    }

    for(int i = mask.size()-1; i >= 0 ; i--) {
        if(mask.at(i)) {
            x1_used.push_back(x1.at(i));
            x2_used.push_back(x2.at(i));
            featuresImg1.push_back(matVector(x1.at(i)));
            featuresImg2.push_back(matVector(x2.at(i)));
            x1.erase(x1.begin()+i);
            x2.erase(x2.begin()+i);
        }
    }

    if(LOG_DEBUG) std::cout << "-- Used matches: " << x1_used.size() << std::endl;

    int estCnt = 0;
    bool homographies_equal = true;
    Mat H, H2;
    Mat e2;

    std::vector<Point2d> goodX1, goodX2;

    while(homographies_equal) {

        if(LOG_DEBUG) std::cout << "-- First Estimation..."<< std::endl;

        estCnt++;

        goodX1.clear();
        goodX2.clear();
        mask.clear();
        for(int i = 0; i < x1.size(); i++) {
            goodX1.push_back(x1.at(i));
            goodX2.push_back(x2.at(i));
        }

        if(LOG_DEBUG) std::cout << "-- Refined number of matches: " << goodX1.size() << std::endl;

        H2 = findHomography(goodX1, goodX2, CV_RANSAC, RANSAC_THREDHOLD, mask, 2000, RANSAC_CONFIDENCE);

        if(!H2.data) {
            if(LOG_DEBUG) std::cout << "-- Estimation failed! (Unable to compute H2)" << std::endl;
            return false;
        }

        H = H1*H2.inv(DECOMP_SVD); // H = (H1*H2⁻1)

        homographies_equal = (computeUniqeEigenvector(H, e2) && isUnity(H));
        if(homographies_equal) {
            if(LOG_DEBUG) std::cout << "-- Homographies equal, repeating estimation..." << std::endl << "-- H = " << std::endl << H << std::endl;
            for(int i = mask.size()-1; i >= 0 ; i--) {
                if(mask.at(i)) {
                    x1.erase(x1.begin()+i);
                    x2.erase(x2.begin()+i);
                }
            }
        }

        if(MAX_H2_ESTIMATIONS < estCnt) {   //Not able to find a second homographie
            if(LOG_DEBUG) std::cout << "-- Estimation failed!" << std::endl;
            return false;
        }
    }

    std::vector<Point2d> x1_temp;
    std::vector<Point2d> x2_temp;

    for(int i = 0; i < mask.size(); i++) {
        if(mask.at(i)) {
            x1_used.push_back(x1.at(i));
            x2_used.push_back(x2.at(i));
            x1_temp.push_back(x1.at(i));
            x2_temp.push_back(x2.at(i));
            featuresImg1.push_back(matVector(x1.at(i)));
            featuresImg2.push_back(matVector(x2.at(i)));
        }
    }

    if(VISUAL_DEBUG) {
        visualizeMatches(image_1_color, image_2_color, x1_temp, x2_temp, 3, true, "Point Homography 2 matches");
        visualizeHomography(H2, image_1_color, image_2_color, "Point homography 2");
    }

    F = crossProductMatrix(e2)*H1;
    enforceRankTwoConstraint(F);

    if(LOG_DEBUG) std::cout << "-- Used matches: " << x1_used.size() << std::endl;

    if(x1_used.size() >= 16) successful = true;

    meanSymmetricTranferError = meanSquaredSymmeticTransferError(F, x1_used, x2_used);

    if(VISUAL_DEBUG) visualizeMatches(image_1_color, image_2_color, x1_used, x2_used, 3, true, "Point Homography used point matches");

    if(LOG_DEBUG) std::cout << std::endl << std::endl;

    return successful;
}

bool FEstimatorHPoints::findPointHomography(std::vector<Point2d> &goodx1, std::vector<Point2d> &goodx2, int method, double confidence, double outliers, pointSubsetStruct &result) {
    double lastError = 0;
    int N;
    std::vector<Point2d> lastIterLineMatchesX1;
    std::vector<Point2d> lastIterLineMatchesX2;
    pointSubsetStruct bestSubset;
    bestSubset.meanSquaredSymmeticTransferError = 0;
    int iteration = 0;
    int stableSolutions = 0;
    double dError = 0;

    if(LOG_DEBUG) std::cout << "-- findPointHomography: confidence = " << confidence << ", relative outliers = " << outliers << std::endl;

    N = std::log(1.0 - confidence)/std::log(1.0 - std::pow(1.0 - outliers, NUM_LINE_CORRESP)); //See Hartley, Zisserman p119
    bestSubset = estimateHomography(goodx1, goodx2, method, N);

    do {

        iteration++;

        if(LOG_DEBUG) std::cout << "-- Iteration: " << iteration <<"/" << MAX_REFINEMENT_ITERATIONS << ", Used number of matches: " << goodLineMatches.size() << std::endl;

        if(goodLineMatches.size() < NUM_LINE_CORRESP) {
            if(LOG_DEBUG) std::cout << "-- To few line matches left! Using second best solution..." << std::endl;
            goodLineMatches = lastIterLineMatches;
            break;
        }

        lastError = bestSubset.meanSquaredSymmeticTransferError;

        //levenbergMarquardt(bestSubset);

        //outliers = goodLineMatches.size();

        lastIterLineMatchesX1.clear();
        lastIterLineMatchesX2.clear();
        for(int i = 0; i < goodx1.size(); i++) {
            lastIterLineMatchesX1.push_back(goodx1.at(i));
            lastIterLineMatchesX2.push_back(goodx2.at(i));
        }

        goodx1.clear();
        goodx2.clear();
        for(int i = 0; i < x1.size(); i++) {
            if(squaredSymmeticTransferPointError(bestSubset.Hs, x1.at(i), x2.at(i)) < MAX_TRANSFER_DIST) {
                goodx1.push_back(x1.at(i));
                goodx2.push_back(x2.at(i));
            }
        }

        //outliers = goodLineMatches.size() / outliers;

        if(iteration == MAX_REFINEMENT_ITERATIONS) return false;

        dError = (lastError - bestSubset.meanSquaredSymmeticTransferError)/bestSubset.meanSquaredSymmeticTransferError;
        if(LOG_DEBUG) std::cout << "-- Mean squared symmetric transfer error: " << bestSubset.meanSquaredSymmeticTransferError << ", rel. Error change: "<< dError << std::endl;

        if(fabs(dError) <= MAX_ERROR_CHANGE) stableSolutions++;
        else stableSolutions = 0;

        if(LOG_DEBUG) std::cout << "-- Stable solutions: " << stableSolutions << std::endl;

    } while(stableSolutions < 3 && bestSubset.meanSquaredSymmeticTransferError > 0.001);

    bestSubset.x1 = goodx1;
    bestSubset.x2 = goodx2;
    bestSubset.Hs = findHomography(bestSubset.x1, bestSubset.x2, 0);
    //levenbergMarquardt(bestSubset);

    if(LOG_DEBUG) std::cout << "-- Final number of used matches: " << bestSubset.x1.size() << std::endl;
    result = bestSubset;

    return true;
}

lineSubsetStruct FEstimatorHPoints::estimateHomography(std::vector<Point2d> x1, std::vector<Point2d> x2, int method, int sets) {
    int numOfPairs = x1.size();
    int numOfPairSubsets = sets;//NUM_LINE_PAIR_SUBSETS_FACTOR*numOfPairs;
    std::vector<pointSubsetStruct> subsets;
    if(LOG_DEBUG) std::cout << "-- Computing "<< numOfPairSubsets << " Homographies" << std::endl;
    //Compute H_21 from NUM_CORRESP line correspondencies
    srand(time(NULL));  //Init random generator
    for(int i = 0; i < numOfPairSubsets; i++) {
        std::vector<int> subsetsIdx;
        pointSubsetStruct subset;

        for(int j = 0; j < NUM_POINT_CORRESP; j++) {
            int subsetIdx = 0;

            do {        //Generate NUM_CORRESP uniqe random indices for line pairs where not 3 are parallel
                subsetIdx = std::rand() % numOfPairs;
            } while(!isUniqe(subsetsIdx, subsetIdx) || !hasGeneralPosition(subsetsIdx, subsetIdx, lineCorrespondencies));

            subsetsIdx.push_back(subsetIdx);
            subset.x1.push_back(getlineCorrespStruct(x1.at(subsetIdx)));
            subset.x2.push_back(getlineCorrespStruct(x2.at(subsetIdx)));
        }
        subset.Hs = findHomography(x1, x2, 0);

        if(subset.Hs.data) {
            subsets.push_back(subset);
        } else i--;

    }

    if(method == RANSAC) return calcRANSAC(subsets, 0.6, lineCorrespondencies);
    else return calcLMedS(subsets, lineCorrespondencies);
}

pointSubsetStruct FEstimatorHPoints::calcRANSAC(std::vector<pointSubsetStruct> &subsets, double threshold, std::vector<Point2d> x1, std::vector<Point2d> x2) {
    if(LOG_DEBUG) std::cout << "-- Computing RANSAC of " << subsets.size() << " Homographies" << std::endl;
    lineSubsetStruct bestSolution = *subsets.begin();
    bestSolution.qualityMeasure = 0;
    double error = 0;
    for(std::vector<lineSubsetStruct>::iterator it = subsets.begin() ; it != subsets.end(); ++it) {
        Mat H_T = it->Hs.t();
        Mat H_invT = it->Hs.inv(DECOMP_SVD).t();
        it->qualityMeasure = 0;       //count inlainers
        for(int i = 0; i < x1.size(); i++) {
            error = squaredTransferPointError(H_T, matVector(x1.at(i)), matVector(x2.at(i)));
            error += squaredTransferPointError(H_invT, matVector(x2.at(i)), matVector(x1.at(i)));
            error/=2.0;
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

pointSubsetStruct FEstimatorHPoints::calcLMedS(std::vector<lineSubsetStruct> &subsets, std::vector<Point2d> x1, std::vector<Point2d> x2) {
    if(LOG_DEBUG) std::cout << "-- Computing LMedS of " << subsets.size() << " Homographies" << std::endl;
    std::vector<pointSubsetStruct>::iterator it = subsets.begin();
    pointSubsetStruct lMedSsubset = *it;
    lMedSsubset.qualityMeasure = calcMedS(*it, x1, x2);
    if(subsets.size() < 2) return lMedSsubset;
    it++;
    do {
        it->qualityMeasure = calcMedS(*it, x1, x2);
        //std::cout << meds << std::endl;
        if(it->qualityMeasure < lMedSsubset.qualityMeasure) {
            lMedSsubset = *it;
        }
        it++;
    } while(it != subsets.end());

    if(LOG_DEBUG) std::cout << "-- LMEDS: " << lMedSsubset.qualityMeasure << std::endl;
    return lMedSsubset;
}

double FEstimatorHPoints::calcMedS(pointSubsetStruct &subset, std::vector<Point2d> x1, std::vector<Point2d> x2) {
    Mat H_invT = subset.Hs.inv(DECOMP_SVD).t();
    Mat H_T = subset.Hs.t();
    std::vector<double> errors;
    double error;
    for(int i = 0; i < x1.size(); i++) { {
        error = squaredTransferPointError(H_T, matVector(x1.at(i)), matVector(x2.at(i)));
        error += squaredTransferPointError(H_invT, matVector(x2.at(i)), matVector(x1.at(i)));
        error/=2.0;
        errors.push_back(error);
        subset.meanSquaredSymmeticTransferError += error;
    }
    subset.meanSquaredSymmeticTransferError /= x1.size();
    std::sort(errors.begin(), errors.end());
    return errors.at(errors.size()/2);
}

double FEstimatorHPoints::squaredSymmeticTransferPointError(Mat H, Point2d x1, Point2d x2) {
    Mat H_invT = H.inv(DECOMP_SVD).t();
    Mat H_T = H.t();
    return (squaredTransferPointError(H_T, x1, x2) + squaredTransferPointError(H_invT, x2, x1))/2.0;
}
