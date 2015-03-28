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
}

bool FEstimatorHPoints::compute() {

    extractMatches();

    if(LOG_DEBUG) std::cout << "-- First Estimation..."<< std::endl;

    //Mat H1 = findHomography(x1, x2, LMEDS, RANSAC_THREDHOLD, mask, 2000, RANSAC_CONFIDENCE);
    pointSubsetStruct firstEstimation;
    pointSubsetStruct secondEstimation;

    std::vector<Point2d> x1_used;
    std::vector<Point2d> x2_used;
    std::vector<pointCorrespStruct> goodMatchesH1;
    std::vector<pointCorrespStruct> goodMatchesH2;
//    for(int i = 0; i < x1.size(); i++) {
//        goodx1.push_back(x1.at(i));
//        goodx2.push_back(x2.at(i));
//    }


    if(!findPointHomography(goodMatchesH1, LMEDS, 0.5, HOMOGRAPHIE_OUTLIERS, firstEstimation)) {
        if(LOG_DEBUG) std::cout << "-- Estimation FAILED!" << std::endl;
        return false;
    }

    for(std::vector<pointCorrespStruct>::const_iterator pointIter = goodMatchesH1.begin(); pointIter != goodMatchesH1.end(); ++pointIter) {
        x1_used.push_back(pointIter->x1);
        x2_used.push_back(pointIter->x2);
        featuresImg1.push_back(matVector(pointIter->x1));
        featuresImg2.push_back(matVector(pointIter->x2));
    }

    if(VISUAL_DEBUG) {
        visualizeMatches(image_1_color, image_2_color, x1_used, x2_used, 3, true, "Point Homography 1 matches");
        visualizeHomography(firstEstimation.Hs, image_1_color, image_2_color, "Point homography 1");
    }

    cv::waitKey(0);

    if(LOG_DEBUG) std::cout << "-- Used matches: " << x1_used.size() << std::endl;

    int estCnt = 0;
    bool homographies_equal = true;
    double outliers = computeRelativeOutliers(HOMOGRAPHIE_OUTLIERS, goodMatchesH1.size(), pointCorrespondencies.size());
    filterUsedPointMatches(pointCorrespondencies, goodMatchesH1);
    Mat H;
    Mat e2;

    while(homographies_equal) {

        if(LOG_DEBUG) std::cout << "-- Second Estimation..."<< std::endl;

        estCnt++;

        if(LOG_DEBUG) std::cout << "-- Refined number of matches: " << pointCorrespondencies.size() << std::endl;

        if(!findPointHomography(goodMatchesH2, RANSAC, 0.5, outliers, secondEstimation)) {
            if(LOG_DEBUG) std::cout << "-- Estimation FAILED!" << std::endl;
            return false;
        }

        H = secondEstimation.Hs*secondEstimation.Hs.inv(DECOMP_SVD); // H = (H1*H2⁻1)

        homographies_equal = (computeUniqeEigenvector(H, e2) && isUnity(H));
        if(homographies_equal) {
            if(LOG_DEBUG) std::cout << "-- Homographies equal, repeating estimation..." << std::endl << "-- H = " << std::endl << H << std::endl;
            outliers = computeRelativeOutliers(outliers, goodMatchesH2.size(), pointCorrespondencies.size());
            filterUsedPointMatches(pointCorrespondencies, goodMatchesH2);
        }

        if(MAX_H2_ESTIMATIONS < estCnt) {   //Not able to find a second homographie
            if(LOG_DEBUG) std::cout << "-- Estimation failed!" << std::endl;
            return false;
        }
    }

    std::vector<Point2d> x1_used_temp;    //debug only
    std::vector<Point2d> x2_used_temp;

    for(std::vector<pointCorrespStruct>::const_iterator pointIter = goodMatchesH2.begin(); pointIter != goodMatchesH2.end(); ++pointIter) {
        x1_used.push_back(pointIter->x1);
        x2_used.push_back(pointIter->x2);
        x1_used_temp.push_back(pointIter->x1);
        x2_used_temp.push_back(pointIter->x2);
        featuresImg1.push_back(matVector(pointIter->x1));
        featuresImg2.push_back(matVector(pointIter->x2));
    }

    if(VISUAL_DEBUG) {
        visualizeMatches(image_1_color, image_2_color, x1_used_temp, x2_used_temp, 3, true, "Point Homography 2 matches");
        visualizeHomography(secondEstimation.Hs, image_1_color, image_2_color, "Point homography 2");
    }

    F = crossProductMatrix(e2)*firstEstimation.Hs;
    enforceRankTwoConstraint(F);

    if(LOG_DEBUG) std::cout << "-- Used matches: " << x1_used_temp.size() << std::endl;

    successful = true;

    error = computeSampsonFDistance(F, x1_used, x2_used);

    //if(VISUAL_DEBUG) visualizeMatches(image_1_color, image_2_color, x1_used, x2_used, 3, true, "Point Homography used point matches");

    if(LOG_DEBUG) std::cout << std::endl << std::endl;

    return successful;
}

bool FEstimatorHPoints::findPointHomography(std::vector<pointCorrespStruct> &goodMatches, int method, double confidence, double outliers, pointSubsetStruct &bestSubset) {
    double lastError = 0;
    int N;
    std::vector<pointCorrespStruct> lastIterLineMatches;
    bestSubset.meanSquaredSymmeticTransferError = 0;
    int iteration = 0;
    int stableSolutions = 0;
    double dError = 0;

    if(LOG_DEBUG) std::cout << "-- findPointHomography: confidence = " << confidence << ", relative outliers = " << outliers << std::endl;

    goodMatches.clear();
    for(std::vector<pointCorrespStruct>::const_iterator it = pointCorrespondencies.begin() ; it != pointCorrespondencies.end(); ++it) {
        goodMatches.push_back(*it);
    }

    do {

        iteration++;

        if(LOG_DEBUG) std::cout << "-- Iteration: " << iteration <<"/" << MAX_REFINEMENT_ITERATIONS << ", Used number of matches: " << goodMatches.size() << std::endl;

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

        lastError = bestSubset.meanSquaredSymmeticTransferError;

        //levenbergMarquardt(bestSubset);

        outliers = computeRelativeOutliers(HOMOGRAPHIE_OUTLIERS, goodMatches.size(), pointCorrespondencies.size());
        N = computeNumberOfEstimations(confidence, outliers, NUM_POINT_CORRESP);
        bestSubset = estimateHomography(goodMatches, method, N);

        lastIterLineMatches.clear();
        for(int i = 0; i < goodMatches.size(); i++) {
            lastIterLineMatches.push_back(goodMatches.at(i));
        }

        goodMatches.clear();
        for(int i = 0; i < pointCorrespondencies.size(); i++) {
            if(sampsonDistance(bestSubset.Hs, pointCorrespondencies.at(i)) < 0.5) {
                goodMatches.push_back(pointCorrespondencies.at(i));
            }
        }

        if(iteration == MAX_REFINEMENT_ITERATIONS) return false;

        dError = (lastError - bestSubset.meanSquaredSymmeticTransferError)/bestSubset.meanSquaredSymmeticTransferError;
        if(LOG_DEBUG) std::cout << "-- Mean squared symmetric transfer error: " << bestSubset.meanSquaredSymmeticTransferError << ", rel. Error change: "<< dError << std::endl;

        if(fabs(dError) <= MAX_ERROR_CHANGE) stableSolutions++;
        else stableSolutions = 0;

        if(LOG_DEBUG) std::cout << "-- Stable solutions: " << stableSolutions << std::endl;

    } while(stableSolutions < 3 && bestSubset.meanSquaredSymmeticTransferError > MIN_ERROR);

    bestSubset.pointCorrespondencies = lastIterLineMatches;
    findHomography(bestSubset);
    bestSubset.meanSquaredSymmeticTransferError = sampsonDistance(bestSubset.Hs, bestSubset.pointCorrespondencies);
    //levenbergMarquardt(bestSubset);

    if(LOG_DEBUG) std::cout << "-- Final number of used matches: " << bestSubset.pointCorrespondencies.size() << ", Mean squared symmetric transfer error: " << bestSubset.meanSquaredSymmeticTransferError << std::endl;

    return true;
}

pointSubsetStruct FEstimatorHPoints::estimateHomography(std::vector<pointCorrespStruct> pointCorresp, int method, int sets) {
    int numOfPairs = pointCorresp.size();
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
            } while(!isUniqe(subsetsIdx, subsetIdx));

            subsetsIdx.push_back(subsetIdx);
            subset.pointCorrespondencies.push_back(getPointCorrespStruct(pointCorresp.at(subsetIdx)));
        }

        findHomography(subset);

        if(subset.Hs.data) {
            subsets.push_back(subset);
        } else i--;
    }

    if(method == RANSAC) return calcRANSAC(subsets, 0.6, pointCorresp);
    else return calcLMedS(subsets, pointCorresp);
}

void FEstimatorHPoints::findHomography(pointSubsetStruct &subset) {
    std::vector<Point2d> _x1;
    std::vector<Point2d> _x2;

    for(std::vector<pointCorrespStruct>::const_iterator iter = subset.pointCorrespondencies.begin(); iter != subset.pointCorrespondencies.end(); ++iter) {
        _x1.push_back(iter->x1);
        _x2.push_back(iter->x2);
    }

    subset.Hs = cv::findHomography(_x1, _x2, 0); //CV_RANSAC
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
            error = sampsonDistance(it->Hs, H_inv, *pointIter);
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
    double error;
    for(std::vector<pointCorrespStruct>::const_iterator pointIter = pointCorresp.begin(); pointIter != pointCorresp.end(); ++pointIter) {
        error = sampsonDistance(subset.Hs, H_inv, *pointIter);
        errors.push_back(error);
        subset.meanSquaredSymmeticTransferError += error;
    }
    subset.meanSquaredSymmeticTransferError /= pointCorresp.size();
    std::sort(errors.begin(), errors.end());
    return errors.at(errors.size()/2);
}

double FEstimatorHPoints::sampsonDistance(Mat H, std::vector<pointCorrespStruct> pointCorresp) {
    Mat H_inv = H.inv(DECOMP_SVD);
    double error = 0;
    for(std::vector<pointCorrespStruct>::const_iterator pointIter = pointCorresp.begin(); pointIter != pointCorresp.end(); ++pointIter) {
        error += sampsonDistance(H, H_inv, *pointIter);
    }
    return error/pointCorresp.size();
}

double FEstimatorHPoints::sampsonDistance(Mat H, pointCorrespStruct pointCorresp) {
    Mat mx1 = matVector(pointCorresp.x1);
    Mat mx2 = matVector(pointCorresp.x2);
    return computeSampsonHDistance(H, mx1, mx2);
}

double FEstimatorHPoints::sampsonDistance(Mat H, Mat H_inv, pointCorrespStruct pointCorresp) {
    Mat mx1 = matVector(pointCorresp.x1);
    Mat mx2 = matVector(pointCorresp.x2);
    return computeSampsonHDistance(H, H_inv, mx1, mx2);
}

void FEstimatorHPoints::filterUsedPointMatches(std::vector<pointCorrespStruct> &pointCorresp, std::vector<pointCorrespStruct> usedPointCorresp) {
    std::vector<pointCorrespStruct>::iterator it= pointCorresp.begin();
    while (it!=pointCorresp.end()) {
        bool remove = false;
        for(std::vector<pointCorrespStruct>::const_iterator used = usedPointCorresp.begin(); used != usedPointCorresp.end(); ++used) {
            if(it->id == used->id) {
                remove = true;
                break;
            }
        }
        if(remove) pointCorresp.erase(it);
        else {
            it++;
        }
    }
    if(LOG_DEBUG) std::cout << "-- Refined number of matches: " << pointCorresp.size() <<  ", removed: " << usedPointCorresp.size() << std::endl;
}
