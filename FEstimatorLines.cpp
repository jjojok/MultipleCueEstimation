#include "FEstimatorLines.h"

FEstimatorLines::FEstimatorLines(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name) {
    image_1 = img1.clone();
    image_2 = img2.clone();
    image_1_color = img1_c.clone();
    image_2_color = img2_c.clone();
    this->name = name;
    if(LOG_DEBUG) std::cout << "Estimating: " << name << std::endl;
    successful = false;
}

FEstimatorLines::~FEstimatorLines() {

}

int FEstimatorLines::extractMatches() {
    /********************************************************************
     * From: http://docs.opencv.org/trunk/modules/line_descriptor/doc/tutorial.html
     * ******************************************************************/

    /* create binary masks */
    cv::Mat mask1 = Mat::ones( image_1.size(), CV_8UC1 );
    cv::Mat mask2 = Mat::ones( image_2.size(), CV_8UC1 );

    /* create a pointer to a BinaryDescriptor object with default parameters */
    Ptr<cv::line_descriptor::BinaryDescriptor> bd = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();

    bd->setNumOfOctaves(OCTAVES);
    bd->setReductionRatio(SCALING);
    bd->setWidthOfBand(9);

    Ptr<cv::line_descriptor::LSDDetector> lsd = cv::line_descriptor::LSDDetector::createLSDDetector();

    /* lines */
    std::vector<cv::line_descriptor::KeyLine> keylines1, keylines2;

    /* extract lines */
    lsd->detect( image_1, keylines1, OCTAVES, SCALING, mask1 );
    lsd->detect( image_2, keylines2, OCTAVES, SCALING, mask2 );

    //Filter detected lines:
    float minLenght = sqrt(image_1.cols*image_1.cols + image_1.rows*image_1.rows)*MIN_LENGTH_FACTOR;
    if(LOG_DEBUG) std::cout << "-- Min line segment length: " << minLenght << std::endl;
    int filtered1 = filterLineExtractions(minLenght, keylines1);
    int filtered2 = filterLineExtractions(minLenght, keylines2);

    if(LOG_DEBUG) {
        std::cout << "-- First image: " << keylines1.size() << " filtered: " << filtered1 << std::endl;
        std::cout << "-- Second image: " << keylines2.size() << " filtered: " << filtered2 << std::endl;
    }

    std::vector<DMatch> matches;

    /* compute descriptors */
    cv::Mat descr1, descr2;
    bd->compute( image_1, keylines1, descr1 );
    bd->compute( image_2, keylines2, descr2 );

    /* create a BinaryDescriptorMatcher object */
    Ptr<cv::line_descriptor::BinaryDescriptorMatcher> bdm = cv::line_descriptor::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

    /* require match */
    bdm->match( descr1, descr2, matches );

    /************************************************************************/

    cv::line_descriptor::KeyLine l1, l2;
    int filteredMatches = 0;
    int id = 0;
    //Reduce max hemming distance if number of matches are high
    int maxHemmingDist = MIN_HEMMING_DIST + std::min((int)((MAX_HEMMING_DIST - MIN_HEMMING_DIST)*1400.0/matches.size()), (MAX_HEMMING_DIST - MIN_HEMMING_DIST));
    if(LOG_DEBUG) std::cout << "-- Min match hemming dist: " << maxHemmingDist << std::endl;

    for (std::vector<DMatch>::const_iterator it= matches.begin(); it!=matches.end(); ++it) {

        l1 = keylines1[it->queryIdx];
        l2 = keylines2[it->trainIdx];

        if (it->distance > maxHemmingDist || filterLineMatch(l1,l2)) {  //Bad match
            filteredMatches++;
        } else {    //Good match, add to correspondence list
            lineCorrespStruct lc = getlineCorrespStruct(l1,l2, id);
            id++;
            matchedLines.push_back(lc);
        }
    }

    if(LOG_DEBUG) {
        std::cout << "-- Number of matches : " << matchedLines.size() << " filtered: " << filteredMatches << std::endl;
    }

    if(VISUAL_DEBUG) visualizeMatches(matchedLines, 2, true, "Line matches");
    return matchedLines.size();
    if(LOG_DEBUG) std::cout << std::endl;
}

bool FEstimatorLines::filterLineMatch(cv::line_descriptor::KeyLine l1, cv::line_descriptor::KeyLine l2) {
    if(smallestRelAngle(l1.angle, l2.angle) > MAX_LINE_ANGLE) return true;
    return false;
}

int FEstimatorLines::filterLineExtractions(float minLenght, std::vector<cv::line_descriptor::KeyLine> &keylines) {
    int filtered = 0;
    std::vector<cv::line_descriptor::KeyLine>::iterator it= keylines.begin();
    while (it!=keylines.end()) {
        if ((it->octave > 0 && it->octave*SCALING*it->lineLength < minLenght) || (it->octave == 0 && it->lineLength < minLenght)) {
            keylines.erase(it);
            filtered++;
        } else it++;
    }
    return filtered;
}

bool FEstimatorLines::compute() {

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

    if(!findHomography(goodLineMatches, LMEDS, H1)) return false;

    if(VISUAL_DEBUG) {
        visualizeHomography(H1.Hs, image_1, image_2, "H21");
        visualizeMatches(H1.lineCorrespondencies, 8, true, "H21 used Matches");
        visualizeProjectedLines(H1, 8, true, "H21 used lines projected to image 2");
    }

    if(LOG_DEBUG) std::cout << "-- Second estimation..." << std::endl;

    filterUsedLineMatches(matchedLines, goodLineMatches);
    if(LOG_DEBUG) std::cout << "-- Refined number of matches: " << matchedLines.size() <<  ", removed: " << goodLineMatches.size() << std::endl;

    lineSubsetStruct H2;
    int estCnt = 0;
    bool H_close_to_unitiy = true;
    Mat H;

    while(H_close_to_unitiy && matchedLines.size() > NUM_CORRESP && MAX_H2_ESTIMATIONS > estCnt) {

        goodLineMatches.clear();
        for(std::vector<lineCorrespStruct>::const_iterator it = matchedLines.begin() ; it != matchedLines.end(); ++it) {
            goodLineMatches.push_back(*it);
        }
        if(!findHomography(goodLineMatches, RANSAC, H2)) return false;
        H = H1.Hs*H2.Hs.inv(DECOMP_SVD); // H = (H1*H2⁻1)

        H_close_to_unitiy = isUnity(H);
        if(H_close_to_unitiy) {
            if(LOG_DEBUG) std::cout << "-- H close to unity, repeating estimation..." << std::endl << "H = " << std::endl << H << std::endl;
            filterUsedLineMatches(matchedLines, goodLineMatches);
            if(LOG_DEBUG) std::cout << "-- Refined number of matches: " << matchedLines.size() <<  ", removed: " << goodLineMatches.size() << std::endl;
        }
        estCnt++;
    }

    if(H_close_to_unitiy) {     //Not able to find a secont homographie
        if(LOG_DEBUG) std::cout << "-- Estimation failed!" << std::endl;
        return false;
    }

    if(VISUAL_DEBUG) {
        visualizeHomography(H2.Hs, image_1, image_2, "H21_2");
        visualizeMatches(H2.lineCorrespondencies, 8, true, "H21_2 used Matches");
        visualizeProjectedLines(H2, 8, true, "H21_2 used lines projected to image 2");
    }

    // Map the OpenCV matrix with Eigen:
    Eigen::Matrix3f HEigen;
    cv2eigen(H, HEigen);
    //http://eigen.tuxfamily.org/dox/classEigen_1_1EigenSolver.html#a8c287af80cfd71517094b75dcad2a31b
    Eigen::EigenSolver<Eigen::Matrix3f> solver;
    solver.compute(HEigen);

    Mat eigenvalues = Mat::zeros(CV_64FC1, 3, 1), eigenvectors = Mat::zeros(CV_64FC1, 3, 3);
    eigen2cv(solver.eigenvalues(), eigenvalues);
    eigen2cv(solver.eigenvectors(), eigenvectors);

    if(LOG_DEBUG) {
        std::cout << "H1*H2^-1 = " << std::endl << H << std::endl;

        for(int i = 0; i < eigenvalues.rows; i++) {
            std::cout << i+1 << "th Eigenvalue: " << eigenvalues.at<float>(i,0) << ", Eigenvector = " << std::endl << eigenvectors.col(i) << std::endl;
        }
    }

    double dist[eigenvalues.rows];
    double lastDist = 0;
    int col = 0;
    for(int i = 0; i < eigenvalues.rows; i ++) {        //find non-unary eigenvalue & its eigenvector
        Mat eig = eigenvalues.row(i);
        dist[i] = 0;
        for(int j = 0; j < eigenvalues.rows; j ++) {
            dist[i] += squaredError(eig, eigenvalues.row(j));
        }
        if(dist[i] > lastDist) {
            col = i;
            lastDist = dist[i];
        }
    }

    std::vector<Mat> e;
    split(eigenvectors.col(col),e); //Remove channel for imaginary part
    if(LOG_DEBUG) std::cout << "e = " << std::endl << e.at(0) << std::endl;
    F = crossProductMatrix(e.at(0))*H1.Hs;
    enforceRankTwoConstraint(F);
    successful = true;

    return true;
}

bool FEstimatorLines::findHomography(std::vector<lineCorrespStruct> &goodLineMatches, int method, lineSubsetStruct &result) {
    float lastError = 0;
    lineSubsetStruct bestSubset;
    bestSubset.meanSquaredSymmeticTransferError = 0;
    int iteration = 0;

    do {

        lastError = bestSubset.meanSquaredSymmeticTransferError;
        bestSubset = estimateHomography(goodLineMatches, method);
        goodLineMatches.clear();

        for(std::vector<lineCorrespStruct>::const_iterator it = matchedLines.begin() ; it != matchedLines.end(); ++it) {
            if(squaredSymmeticTransferError(bestSubset.Hs, *it) < MAX_PROJ_DIST) goodLineMatches.push_back(*it);
        }

        if(LOG_DEBUG) std::cout << "-- Mean squared symmetric transfer error: " << bestSubset.meanSquaredSymmeticTransferError << std::endl;
        iteration++;
        if(iteration == MAX_REFINEMENT_ITERATIONS) {
            return false;
        }

    } while(goodLineMatches.size() > NUM_CORRESP && fabs(lastError - bestSubset.meanSquaredSymmeticTransferError)/bestSubset.meanSquaredSymmeticTransferError > 0.01 && bestSubset.meanSquaredSymmeticTransferError > 10E-12);

    bestSubset.lineCorrespondencies = goodLineMatches;
    computeHomography(bestSubset);
    result = bestSubset;

    return true;
}

bool FEstimatorLines::isUnity(Mat m) {
    Mat diff = abs(m - Mat::eye(m.rows, m.cols, CV_32FC1));
    for(int i = 0; i < m.cols; i++) {
        if(diff.at<float>(i,i) > MARGIN) return false;
    }
    return true;
}

lineSubsetStruct FEstimatorLines::estimateHomography(std::vector<lineCorrespStruct> lineCorrespondencies, int method) {
    int numOfPairs = lineCorrespondencies.size();
    int numOfPairSubsets = NUM_LINE_PAIR_SUBSETS_FACTOR*numOfPairs;
    std::vector<lineSubsetStruct> subsets;
    if(LOG_DEBUG) std::cout << "-- Computing Homographies" << std::endl;
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

    if(method == RANSAC) return calcRANSAC(subsets, MAX_PROJ_DIST, lineCorrespondencies);
    else return calcLMedS(subsets, lineCorrespondencies);
}

bool FEstimatorLines::computeHomography(lineSubsetStruct &subset) {
    Mat linEq = Mat::ones(2*subset.lineCorrespondencies.size(),9,CV_32FC1);
    Mat* T = normalizeLines(subset.lineCorrespondencies);
    fillHLinEq(linEq, subset.lineCorrespondencies);
    Mat A = linEq.colRange(0, linEq.cols-1);
    Mat x = -linEq.col(linEq.cols-1);
    solve(A, x, subset.Hs_normalized, DECOMP_SVD);
    subset.Hs_normalized.resize(9);
    subset.Hs_normalized = subset.Hs_normalized.reshape(1,3);
    subset.Hs_normalized.at<float>(2,2) = 1.0;
    subset.Hs = denormalize(subset.Hs_normalized, T[0], T[1]);
    return true;
}

bool FEstimatorLines::hasGeneralPosition(std::vector<int> subsetsIdx, int newIdx, std::vector<lineCorrespStruct> lineCorrespondencies) {
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

bool FEstimatorLines::isUniqe(std::vector<int> subsetsIdx, int newIdx) {
    if(subsetsIdx.size() == 0) return true;
    for(std::vector<int>::const_iterator iter = subsetsIdx.begin(); iter != subsetsIdx.end(); ++iter) {
        if(*iter == newIdx) return false;
    }
    return true;
}

void FEstimatorLines::filterUsedLineMatches(std::vector<lineCorrespStruct> &matches, std::vector<lineCorrespStruct> usedMatches) {
    std::vector<lineCorrespStruct>::iterator it= matches.begin();
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
}

void FEstimatorLines::visualizeMatches(std::vector<lineCorrespStruct> correspondencies, int lineWidth, bool drawConnections, std::string name) {
    Mat img;
    hconcat(image_1_color.clone(), image_2_color.clone(), img);
    for(std::vector<lineCorrespStruct>::iterator it = correspondencies.begin() ; it != correspondencies.end(); ++it) {
        Scalar color = Scalar(rand()%255, rand()%255, rand()%255);
        cv::line(img, cvPoint2D32f(it->line1Start.at<float>(0,0), it->line1Start.at<float>(1,0)), cvPoint2D32f(it->line1End.at<float>(0,0), it->line1End.at<float>(1,0)), color, lineWidth);
        cv::line(img, cvPoint2D32f(it->line2Start.at<float>(0,0) + image_1_color.cols, it->line2Start.at<float>(1,0)), cvPoint2D32f(it->line2End.at<float>(0,0) + image_1_color.cols, it->line2End.at<float>(1,0)), color, lineWidth);
        if(drawConnections) {
            cv::line(img, cvPoint2D32f(it->line1Start.at<float>(0,0), it->line1Start.at<float>(1,0)), cvPoint2D32f(it->line2Start.at<float>(0,0) + image_1_color.cols, it->line2Start.at<float>(1,0)), color, lineWidth);
        }
    }
    showImage(name, img, WINDOW_NORMAL, 1600);
}

void FEstimatorLines::visualizeProjectedLines(lineSubsetStruct subset, int lineWidth, bool drawConnections, std::string name) {
    Mat img;
    hconcat(image_1_color.clone(), image_2_color.clone(), img);
    for(std::vector<lineCorrespStruct>::iterator it = subset.lineCorrespondencies.begin() ; it != subset.lineCorrespondencies.end(); ++it) {
        Mat start2 = subset.Hs*it->line1Start;
        start2 /= start2.at<float>(2,0);
        Mat end2 = subset.Hs*it->line1End;
        end2 /= end2.at<float>(2,0);
        Scalar color = Scalar(rand()%255, rand()%255, rand()%255);
        cv::line(img, cvPoint2D32f(it->line1Start.at<float>(0,0), it->line1Start.at<float>(1,0)), cvPoint2D32f(it->line1End.at<float>(0,0), it->line1End.at<float>(1,0)), color, lineWidth);
        cv::line(img, cvPoint2D32f(start2.at<float>(0,0) + image_1_color.cols, start2.at<float>(1,0)), cvPoint2D32f(end2.at<float>(0,0) + image_1_color.cols, end2.at<float>(1,0)), color, lineWidth);
        if(drawConnections) {
            cv::line(img, cvPoint2D32f(it->line1Start.at<float>(0,0), it->line1Start.at<float>(1,0)), cvPoint2D32f(start2.at<float>(0,0) + image_1_color.cols, start2.at<float>(1,0)), color, lineWidth);
        }
    }
    showImage(name, img, WINDOW_NORMAL, 1600);
}

void FEstimatorLines::fillHLinEq(Mat &linEq, std::vector<lineCorrespStruct> correspondencies) {
    lineCorrespStruct lc;
    for(int i = 0; i < correspondencies.size(); i++) {
        lc = correspondencies.at(i);
        float A = lc.line2StartNormalized.at<float>(1,0) - lc.line2EndNormalized.at<float>(1,0);
        float B = lc.line2EndNormalized.at<float>(0,0) - lc.line2StartNormalized.at<float>(0,0);
        float C = lc.line2StartNormalized.at<float>(0,0)*lc.line2EndNormalized.at<float>(1,0) - lc.line2EndNormalized.at<float>(0,0)*lc.line2StartNormalized.at<float>(1,0);
        int row = 2*i;
        fillHLinEqBase(linEq, lc.line1StartNormalized.at<float>(0,0), lc.line1StartNormalized.at<float>(1,0), A, B, C, row);
        fillHLinEqBase(linEq, lc.line1EndNormalized.at<float>(0,0), lc.line1EndNormalized.at<float>(1,0), A, B, C, row + 1);
    }
}

void FEstimatorLines::fillHLinEqBase(Mat &linEq, float x, float y, float A, float B, float C, int row) {
    linEq.at<float>(row, 0) = A*x;
    linEq.at<float>(row, 1) = A*y;
    linEq.at<float>(row, 2) = A;
    linEq.at<float>(row, 3) = B*x;
    linEq.at<float>(row, 4) = B*y;
    linEq.at<float>(row, 5) = B;
    linEq.at<float>(row, 6) = C*x;
    linEq.at<float>(row, 7) = C*y;
    linEq.at<float>(row, 8) = C;
}

lineSubsetStruct FEstimatorLines::calcRANSAC(std::vector<lineSubsetStruct> &subsets, double threshold, std::vector<lineCorrespStruct> lineCorrespondencies) {
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

lineSubsetStruct FEstimatorLines::calcLMedS(std::vector<lineSubsetStruct> &subsets, std::vector<lineCorrespStruct> lineCorrespondencies) {
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


float FEstimatorLines::calcMedS(lineSubsetStruct &subset, std::vector<lineCorrespStruct> lineCorrespondencies) {
    Mat H_invT = subset.Hs.inv(DECOMP_SVD).t();
    Mat H_T = subset.Hs.t();
    std::vector<float> errors;
    float error;
    for(std::vector<lineCorrespStruct>::iterator it = lineCorrespondencies.begin() ; it != lineCorrespondencies.end(); ++it) {
        error = squaredSymmeticTransferError(H_invT, H_T, *it);
        errors.push_back(error);
        subset.meanSquaredSymmeticTransferError += error;
    }
    subset.meanSquaredSymmeticTransferError /= lineCorrespondencies.size();
    std::sort(errors.begin(), errors.end());
    return errors.at(errors.size()/2);
}

double FEstimatorLines::squaredSymmeticTransferError(Mat H, lineCorrespStruct lc) {
    Mat H_invT = H.inv(DECOMP_SVD).t();
    Mat H_T = H.t();
    return squaredSymmeticTransferError(H_invT, H_T, lc);
}

double FEstimatorLines::squaredSymmeticTransferError(Mat H_invT, Mat H_T, lineCorrespStruct lc) {
    Mat A = H_T*crossProductMatrix(lc.line2Start)*lc.line2End;
    Mat start1 = lc.line1Start.t()*A;
    Mat end1 = lc.line1End.t()*A;
    Mat B = H_invT*crossProductMatrix(lc.line1Start)*lc.line1End;
    Mat start2 = lc.line2Start.t()*B;
    Mat end2 = lc.line2End.t()*B;
    Mat result = (start1*start1 + end1*end1)/(A.at<float>(0,0)*A.at<float>(0,0) + A.at<float>(1,0)*A.at<float>(1,0)) + (start2*start2 + end2*end2)/(B.at<float>(0,0)*B.at<float>(0,0) + B.at<float>(1,0)*B.at<float>(1,0));
    return result.at<float>(0,0);
}

Mat* FEstimatorLines::normalizeLines(std::vector<lineCorrespStruct> &correspondencies) {

    //Normalization: Hartley, Zisserman, Multiple View Geometry in Computer Vision, p. 109

    Mat* normalizationMats = new Mat[2];
    float sum1x = 0, sum1y = 0, sum2x = 0, sum2y = 0, N = 0;
    float mean1x = 0, mean1y = 0, mean2x = 0, mean2y = 0, v1 = 0, v2 = 0, scale1 = 0, scale2 = 0;

    for (std::vector<lineCorrespStruct>::iterator it= correspondencies.begin(); it!=correspondencies.end(); ++it) {

        sum1x += it->line1Start.at<float>(0,0) + it->line1End.at<float>(0,0);
        sum2x += it->line2Start.at<float>(0,0) + it->line2End.at<float>(0,0);

        sum1y += it->line1Start.at<float>(1,0) + it->line1End.at<float>(1,0);
        sum2y += it->line2Start.at<float>(1,0) + it->line2End.at<float>(1,0);

    }

    normalizationMats[0] = Mat::eye(3,3, CV_32FC1);
    normalizationMats[1] = Mat::eye(3,3, CV_32FC1);
    N = 2*correspondencies.size();

    mean1x = sum1x/N;
    mean1y = sum1y/N;
    mean2x = sum2x/N;
    mean2y = sum2y/N;

    for (std::vector<lineCorrespStruct>::iterator it= correspondencies.begin(); it!=correspondencies.end(); ++it) {
        v1 += fnorm(it->line1Start.at<float>(0,0)-mean1x, it->line1Start.at<float>(1,0)-mean1y);
        v1 += fnorm(it->line1End.at<float>(0,0)-mean1x, it->line1End.at<float>(1,0)-mean1y);
        v2 += fnorm(it->line2Start.at<float>(0,0)-mean2x, it->line2Start.at<float>(1,0)-mean2y);
        v2 += fnorm(it->line2End.at<float>(0,0)-mean2x, it->line2End.at<float>(1,0)-mean2y);
    }

    v1 /= N;
    v2 /= N;

    scale1 = sqrt(2.0)/v1;
    scale2 = sqrt(2.0)/v2;

    normalizationMats[0].at<float>(0,0) = scale1;
    normalizationMats[0].at<float>(1,1) = scale1;
    normalizationMats[0].at<float>(0,2) = -scale1*mean1x;
    normalizationMats[0].at<float>(1,2) = -scale1*mean1y;

    normalizationMats[1].at<float>(0,0) = scale2;
    normalizationMats[1].at<float>(1,1) = scale2;
    normalizationMats[1].at<float>(0,2) = -scale2*mean2x;
    normalizationMats[1].at<float>(1,2) = -scale2*mean2y;

    //Carry out normalization:

    for (std::vector<lineCorrespStruct>::iterator it= correspondencies.begin(); it!=correspondencies.end(); ++it) {

        it->line1StartNormalized = normalizationMats[0]*it->line1Start;
        it->line2StartNormalized = normalizationMats[1]*it->line2Start;

        it->line1EndNormalized = normalizationMats[0]*it->line1End;
        it->line2EndNormalized = normalizationMats[1]*it->line2End;

    }

    return normalizationMats;
}

lineCorrespStruct FEstimatorLines::getlineCorrespStruct(lineCorrespStruct lcCopy) {
    lineCorrespStruct* lc = new lineCorrespStruct;
    lc->line1Angle = lcCopy.line1Angle;
    lc->line2Angle = lcCopy.line2Angle;

    lc->line1Length = lcCopy.line1Length;
    lc->line2Length = lcCopy.line2Length;

    lc->line1Start = lcCopy.line1Start.clone();
    lc->line2Start = lcCopy.line2Start.clone();
    lc->line1End = lcCopy.line1End.clone();
    lc->line2End = lcCopy.line2End.clone();

    lc->id = lcCopy.id;

    return *lc;
}

lineCorrespStruct FEstimatorLines::getlineCorrespStruct(cv::line_descriptor::KeyLine l1, cv::line_descriptor::KeyLine l2, int id) {
    lineCorrespStruct* lc = new lineCorrespStruct;
    double scaling = 1;
    if(l1.octave > 0) { //TODO: OpenCV bug: coordinates are from downscaled versions of image pyramid
        scaling = l1.octave*SCALING;
    }

    lc->line1Angle = l1.angle;
    lc->line2Angle = l2.angle;

    lc->line1Length = l1.lineLength*scaling;
    lc->line2Length = l1.lineLength*scaling;

    lc->line1Start = matVector(l1.startPointX*scaling, l1.startPointY*scaling, 1);
    lc->line2Start = matVector(l2.startPointX*scaling, l2.startPointY*scaling, 1);
    lc->line1End = matVector(l1.endPointX*scaling, l1.endPointY*scaling, 1);
    lc->line2End = matVector(l2.endPointX*scaling, l2.endPointY*scaling, 1);

    lc->id = id;

    return *lc;
}

lineCorrespStruct FEstimatorLines::getlineCorrespStruct(float start1x, float start1y, float end1x, float end1y, float start2x, float start2y , float end2x, float end2y, int id) {
    lineCorrespStruct* lc = new lineCorrespStruct;

    lc->line1Angle = atan2(end1y - start1y, end1x - start1x);
    lc->line2Angle = atan2(end2y - start2y, end2x - start2x);

    lc->line1Length = fnorm(start1x-end1x, start1y-end1y);
    lc->line2Length = fnorm(start2x-end2x, start2y-end2y);

    lc->line1Start = matVector(start1x, start1y, 1);
    lc->line2Start = matVector(start2x, start2y, 1);
    lc->line1End = matVector(end1x, end1y, 1);
    lc->line2End = matVector(end2x, end2y, 1);

    lc->id;

    return *lc;
}

bool compareLineCorrespErrors(lineCorrespSubsetError ls1, lineCorrespSubsetError ls2) {
    return ls1.lineCorrespError < ls2.lineCorrespError;
}
