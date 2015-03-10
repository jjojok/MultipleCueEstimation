#include "FEstimatorLines.h"

FEstimatorLines::FEstimatorLines(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name) {
    image_1 = img1.clone();
    image_2 = img2.clone();
    image_1_color = img1_c.clone();
    image_2_color = img2_c.clone();
    this->name = name;
    std::cout << "Estimating: " << name << std::endl;
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
    //Reduce max hemming distance if number of matches are high
    int maxHemmingDist = MIN_HEMMING_DIST + std::min((int)((MAX_HEMMING_DIST - MIN_HEMMING_DIST)*1000.0/matches.size()), (MAX_HEMMING_DIST - MIN_HEMMING_DIST));
    if(LOG_DEBUG) std::cout << "-- Min match hemming dist: " << maxHemmingDist << std::endl;

    for (std::vector<DMatch>::const_iterator it= matches.begin(); it!=matches.end(); ++it) {

        l1 = keylines1[it->queryIdx];
        l2 = keylines2[it->trainIdx];

        if (it->distance > maxHemmingDist || filterLineMatch(l1,l2)) {  //Bad match
            filteredMatches++;
        } else {    //Good match, add to correspondence list

            lineCorrespStruct lc;
            lc.line1 = l1;
            lc.line2 = l2;

            if(l1.octave > 0) { //TODO: OpenCV bug: coordinates are from downscaled versions of image pyramid
                double scaling = l1.octave*SCALING;

                lc.line1.startPointX *= scaling;
                lc.line1.startPointY *= scaling;
                lc.line2.startPointX *= scaling;
                lc.line2.startPointY *= scaling;

                lc.line1.endPointX *= scaling;
                lc.line1.endPointY *= scaling;
                lc.line2.endPointX *= scaling;
                lc.line2.endPointY *= scaling;

                lc.line1.lineLength *= scaling;
                lc.line2.lineLength *= scaling;
            }

            lc.line1Start = matVector(l1.startPointX, l1.startPointY, 1);
            lc.line2Start = matVector(l2.startPointX, l2.startPointY, 1);
            lc.line1End = matVector(l1.endPointX, l1.endPointY, 1);
            lc.line2End = matVector(l2.endPointX, l2.endPointY, 1);


//            std::cout << "l1s: " << lc.line1Start << std::endl << "l2s: " << lc.line2Start << std::endl;
//            std::cout << "l1e: " << lc.line1End << std::endl << "l2e: " << lc.line2End << std::endl;

            lineCorrespondencies.push_back(lc);

        }
    }

    if(LOG_DEBUG) {
        std::cout << "-- Number of matches : " << lineCorrespondencies.size() << " filtered: " << filteredMatches << std::endl;
    }

    if(VISUAL_DEBUG) visualizeMatches(lineCorrespondencies, 2, true, "Line matches");

    return lineCorrespondencies.size();

    std::cout << std::endl;
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

Mat FEstimatorLines::compute() {

    extractMatches();

    if(lineCorrespondencies.size() < 2*NUM_CORRESP) {
        if(LOG_DEBUG) std::cout << "-- Estimation failed, not enough line correspondencies!" << std::endl;
        return Mat::zeros(CV_32FC1, 3, 3);
    }

    if(LOG_DEBUG) std::cout << "-- First estimation..." << std::endl;

    lineSubsetStruct lineSubsetStruct1 = estimateHomography();

//    for(int i = 0; i < lineSubsetStruct1.lineCorrespondencies.size(); i++) {
//        std::cout << "l1s: " << lineSubsetStruct1.lineCorrespondencies.at(i).line1Start << std::endl << "l2s: " << lineSubsetStruct1.lineCorrespondencies.at(i).line2Start << std::endl;
//        std::cout << "l1e: " << lineSubsetStruct1.lineCorrespondencies.at(i).line1End << std::endl << "l2e: " << lineSubsetStruct1.lineCorrespondencies.at(i).line2End << std::endl;
//    }

    visualizeProjectedLines(lineSubsetStruct1, 2, true, "Projected lines 1");

    if(VISUAL_DEBUG) {
        visualizeHomography(lineSubsetStruct1.Hs, image_1, image_2, "H21");
        visualizeMatches(lineSubsetStruct1.lineCorrespondencies, 8, true, "H21 used Matches");
    }

    //    for(int i = 0; i < lineSubsetStruct1.lineCorrespondenceIdx.size(); i++) {
    //        refinedlineCorrespondencies.push_back(lineCorrespondencies.at(lineSubsetStruct1.lineCorrespondenceIdx.at(i)));
    //    }

    std::vector<lineCorrespStruct> refinedlineCorrespondencies;
    int removed = refineLineMatches(lineSubsetStruct1, refinedlineCorrespondencies);
    lineCorrespondencies = refinedlineCorrespondencies;
    if(LOG_DEBUG) std::cout << "-- Refined number of matches: " << refinedlineCorrespondencies.size() <<  ", removed: " << removed << std::endl;

//    float threshold = MAX_PROJ_DIST;//pow(1.4826*(1.0 + 5.0/(lineCorrespondencies.size() - NUM_CORRESP))*sqrt(lineSubsetStruct1.errorMeasure), 2)*OUTLIER_THESHOLD_FACTOR;
//    int removed = refineLineMatches(lineSubsetStruct1, threshold);



    int estCnt = 0;
    bool H_close_to_unitiy = true;
    Mat H;

    while(H_close_to_unitiy && lineCorrespondencies.size() > NUM_CORRESP && MAX_H2_ESTIMATIONS > estCnt) {

        if(LOG_DEBUG) std::cout << "-- Second estimation..." << std::endl;

        lineSubsetStruct lineSubsetStruct2 = estimateHomography();

        if(VISUAL_DEBUG) {
            visualizeMatches(lineCorrespondencies, 4, true, "remaining line Matches");
            visualizeHomography(lineSubsetStruct2.Hs, image_1, image_2, "H21_2");
            visualizeMatches(lineSubsetStruct2.lineCorrespondencies, 8, true, "H21_2 used Matches");
        }

        H = lineSubsetStruct1.Hs*lineSubsetStruct2.Hs.inv(DECOMP_SVD); // H = (H1*H2⁻1)

        H_close_to_unitiy = isUnity(H);

        if(H_close_to_unitiy) {
            if(LOG_DEBUG) std::cout << "-- H close to unity, repeating estimation..." << std::endl << "H = " << std::endl << H << std::endl;
            std::vector<lineCorrespStruct> refinedlineCorrespondencies;
            int removed = refineLineMatches(lineSubsetStruct2, refinedlineCorrespondencies);
            lineCorrespondencies = refinedlineCorrespondencies;
            if(LOG_DEBUG) std::cout << "-- Refined number of matches: " << refinedlineCorrespondencies.size() <<  ", removed: " << removed << std::endl;
        }

        estCnt++;

    }

    if(H_close_to_unitiy) {
        if(LOG_DEBUG) std::cout << "-- Estimation failed!" << std::endl;
        return Mat::zeros(CV_32FC1, 3, 3);
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
    for(int i = 0; i < eigenvalues.rows; i ++) {        //find non-unary eigenvalue
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
    std::cout << "e = " << std::endl << e.at(0) << std::endl;
    F = crossProductMatrix(e.at(0))*lineSubsetStruct1.Hs;

    //Enforce Rank 2 constraint:
    SVD svd;
    Mat u, vt, w;
    svd.compute(F, w, u, vt);
    Mat newW = Mat::zeros(3,3,CV_32FC1);
    newW.at<float>(0,0) = w.at<float>(0,0);
    newW.at<float>(1,1) = w.at<float>(1,0);
    F = u*newW*vt;
    F /= F.at<float>(2,2);

    cvWaitKey(0);

    return F;
}

bool FEstimatorLines::isUnity(Mat m) {

    Mat diff = abs(m - Mat::eye(m.rows, m.cols, CV_32FC1));

    std::cout << diff << std::endl;

    for(int i = 0; i < m.cols; i++) {
        if(diff.at<float>(i,i) > MARGIN) return false;
    }

    return true;
}

lineSubsetStruct FEstimatorLines::estimateHomography() {
    int numOfPairs = lineCorrespondencies.size();
    int numOfPairSubsets = NUM_LINE_PAIR_SUBSETS_FACTOR*numOfPairs;
    std::vector<lineSubsetStruct> subsets;
    if(LOG_DEBUG) std::cout << "-- Computing Homographies" << std::endl;
    //Compute H_21 from NUM_CORRESP line correspondencies
    srand(time(NULL));  //Init random generator
    for(int i = 0; i < numOfPairSubsets; i++) {
        std::vector<int> subsetsIdx;
        Mat linEq = Mat::ones(2*NUM_CORRESP,9,CV_32FC1);
        lineSubsetStruct subset;

        for(int j = 0; j < NUM_CORRESP; j++) {
            int subsetIdx = 0;

            do {        //Generate NUM_CORRESP uniqe random indices for line pairs where not 3 are parallel
                subsetIdx = std::rand() % numOfPairs;
            } while(std::find(subsetsIdx.begin(), subsetsIdx.end(), subsetIdx) != subsetsIdx.end() && hasGeneralPosition(subsetsIdx, subsetIdx));

            subsetsIdx.push_back(subsetIdx);
            subset.lineCorrespondencies.push_back(lineCorrespondencies.at(subsetIdx));
//            std::cout << "l1s: " << lineCorrespondencies.at(subsetIdx).line1Start << std::endl << "l2s: " << lineCorrespondencies.at(subsetIdx).line2Start << std::endl;
//            std::cout << "l1e: " << lineCorrespondencies.at(subsetIdx).line1End << std::endl << "l2e: " << lineCorrespondencies.at(subsetIdx).line2End << std::endl;
        }

//        for(int i = 0; i < subset.lineCorrespondencies.size(); i++) {
//            std::cout << "l1s: " << subset.lineCorrespondencies.at(i).line1Start << std::endl << "l2s: " << subset.lineCorrespondencies.at(i).line2Start << std::endl;
//            std::cout << "l1e: " << subset.lineCorrespondencies.at(i).line1End << std::endl << "l2e: " << subset.lineCorrespondencies.at(i).line2End << std::endl;
//        }

        subset.lineCorrespondenceIdx = subsetsIdx;
        Mat* T = normalizeLines(subset.lineCorrespondencies);
        fillHLinEq(&linEq, subset.lineCorrespondencies);

        Mat A = linEq.colRange(0, linEq.cols-1);
        Mat x = -linEq.col(linEq.cols-1);
        solve(A, x, subset.Hs_normalized, DECOMP_SVD);
        subset.Hs_normalized.resize(9);
        subset.Hs_normalized = subset.Hs_normalized.reshape(1,3);
        subset.Hs_normalized.at<float>(2,2) = 1.0;
        subset.Hs = denormalize(subset.Hs_normalized, T[0], T[1]);
        subsets.push_back(subset);
    }

    //return calcLMedS(subsets);
    //return calcRANSAC(subsets, MAX_PROJ_DIST);
    return calcRANSAC(subsets, 0.1);
}

bool FEstimatorLines::hasGeneralPosition(std::vector<int> subsetsIdx, int newIdx) {
    if(subsetsIdx.size() < 3) return true;
    int parallelCout = 0;
    lineCorrespStruct lc;
    lineCorrespStruct lcNew = lineCorrespondencies.at(newIdx);
    for(int i = 0; i < subsetsIdx.size(); i++) {
        lc = lineCorrespondencies.at(subsetsIdx.at(i));
        if(smallestRelAngle(lc.line1.angle, lcNew.line1.angle) < MAX_ANGLE_DIFF || smallestRelAngle(lc.line2.angle, lcNew.line2.angle) < MAX_ANGLE_DIFF) parallelCout++;
    }
    if(parallelCout >= 3) return false;
    return true;
}

int FEstimatorLines::refineLineMatches(lineSubsetStruct subset, std::vector<lineCorrespStruct> &refinedlineCorrespondencies) {
    for(int i = 0; i < lineCorrespondencies.size(); i++) {
        bool add = true;
        for(int j = 0; j < subset.consensusCorrespIndices.size(); j++) {
            if(subset.consensusCorrespIndices.at(j) == i) {
                add = false;
                break;
            }
        }
        if(add) refinedlineCorrespondencies.push_back(lineCorrespondencies.at(i));
    }
    return subset.consensusCorrespIndices.size();
}

void FEstimatorLines::visualizeMatches(std::vector<lineCorrespStruct> correspondencies, int lineWidth, bool drawConnections, std::string name) {
    Mat img;
    hconcat(image_1_color.clone(), image_2_color.clone(), img);
    for(std::vector<lineCorrespStruct>::iterator it = correspondencies.begin() ; it != correspondencies.end(); ++it) {
        Scalar color = Scalar(rand()%255, rand()%255, rand()%255);
        cv::line(img, cvPoint2D32f(it->line1.startPointX, it->line1.startPointY), cvPoint2D32f(it->line1.endPointX, it->line1.endPointY), color, lineWidth);
        cv::line(img, cvPoint2D32f(it->line2.startPointX + image_1_color.cols, it->line2.startPointY), cvPoint2D32f(it->line2.endPointX + image_1_color.cols, it->line2.endPointY), color, lineWidth);
        if(drawConnections) {
            cv::line(img, cvPoint2D32f(it->line1.startPointX, it->line1.startPointY), cvPoint2D32f(it->line2.startPointX + image_1_color.cols, it->line2.startPointY), color, lineWidth);
        }
    }
    showImage(name, img, WINDOW_NORMAL, 1600);
}

void FEstimatorLines::visualizeProjectedLines(lineSubsetStruct subset, int lineWidth, bool drawConnections, std::string name) {
    Mat img;
    hconcat(image_1_color.clone(), image_2_color.clone(), img);
    for(std::vector<lineCorrespStruct>::iterator it = subset.lineCorrespondencies.begin() ; it != subset.lineCorrespondencies.end(); ++it) {

            std::cout << "l1s: " << it->line1Start << std::endl << "l2s: " << it->line2Start << std::endl;
            std::cout << "l1e: " << it->line1End << std::endl << "l2e: " << it->line2End << std::endl;

        Mat start2 = subset.Hs*it->line1Start;
        start2 /= start2.at<float>(2,0);
        Mat end2 = subset.Hs*it->line1End;
        end2 /= end2.at<float>(2,0);
        std::cout << "s2: " << start2 << std::endl << "e2: " << end2 << std::endl;
        Scalar color = Scalar(rand()%255, rand()%255, rand()%255);
        cv::line(img, cvPoint2D32f(it->line1.startPointX, it->line1.startPointY), cvPoint2D32f(it->line1.endPointX, it->line1.endPointY), color, lineWidth);
        cv::line(img, cvPoint2D32f(start2.at<float>(0,0) + image_1_color.cols, start2.at<float>(1,0)), cvPoint2D32f(end2.at<float>(0,0) + image_1_color.cols, end2.at<float>(1,0)), color, lineWidth);
        if(drawConnections) {
            cv::line(img, cvPoint2D32f(it->line1.startPointX, it->line1.startPointY), cvPoint2D32f(start2.at<float>(0,0) + image_1_color.cols, start2.at<float>(1,0)), color, lineWidth);
        }
    }
    showImage(name, img, WINDOW_NORMAL, 1600);
}

void FEstimatorLines::fillHLinEq(Mat* linEq, std::vector<lineCorrespStruct> correspondencies) {
    lineCorrespStruct lc;
    for(int i = 0; i < correspondencies.size(); i++) {
        lc = correspondencies.at(i);
        float A = lc.line2StartNormalized.at<float>(1,0) - lc.line2EndNormalized.at<float>(1,0);
        float B = lc.line2EndNormalized.at<float>(0,0) - lc.line2StartNormalized.at<float>(0,0);
        float C = lc.line1StartNormalized.at<float>(0,0)*lc.line2EndNormalized.at<float>(1,0) - lc.line2EndNormalized.at<float>(0,0)*lc.line2StartNormalized.at<float>(1,0);
        int row = 2*i;
        fillHLinEqBase(linEq, lc.line1StartNormalized.at<float>(0,0), lc.line1StartNormalized.at<float>(1,0), A, B, C, row);
        fillHLinEqBase(linEq, lc.line1EndNormalized.at<float>(0,0), lc.line1EndNormalized.at<float>(1,0), A, B, C, row + 1);
    }
}

void FEstimatorLines::fillHLinEqBase(Mat* linEq, float x, float y, float A, float B, float C, int row) {
    linEq->at<float>(row, 0) = A*x;
    linEq->at<float>(row, 1) = A*y;
    linEq->at<float>(row, 2) = A;
    linEq->at<float>(row, 3) = B*x;
    linEq->at<float>(row, 4) = B*y;
    linEq->at<float>(row, 5) = B;
    linEq->at<float>(row, 6) = C*x;
    linEq->at<float>(row, 7) = C*y;
    linEq->at<float>(row, 8) = C;
}

//lineSubsetStruct FEstimatorLines::calcRANSAC(std::vector<lineSubsetStruct> subsets, double threshold) {
//    if(LOG_DEBUG) std::cout << "-- Computing RANSAC of " << subsets.size() << " Homographies" << std::endl;
//    lineSubsetStruct bestSolution = *subsets.begin();
//    bestSolution.errorMeasure = 0;
//    for(std::vector<lineSubsetStruct>::iterator it = subsets.begin() ; it != subsets.end(); ++it) {
//        Mat H_T = it->Hs.t();
//        Mat H_invT = it->Hs.inv(DECOMP_SVD).t();
//        for(int i = 0; i < lineCorrespondencies.size(); i++) {
//            if(sqrt(squaredProjectionDistance(H_invT, H_T, lineCorrespondencies.at(i))) <= threshold) it->consensusSetIdx.push_back(i);
//            //if(algebraicDistance(H_T, lineCorrespondencies.at(i)) <= threshold) it->consensusSetIdx.push_back(i);
//        }
//        it->errorMeasure = it->consensusSetIdx.size();

//        if(it->errorMeasure > bestSolution.errorMeasure) bestSolution = *it;
//    }
//    if(LOG_DEBUG) std::cout << "-- RANSAC inlaiers: " << bestSolution.errorMeasure << std::endl;
//    return bestSolution;
//}

lineSubsetStruct FEstimatorLines::calcRANSAC(std::vector<lineSubsetStruct> subsets, double threshold) {
    if(LOG_DEBUG) std::cout << "-- Computing RANSAC of " << subsets.size() << " Homographies" << std::endl;
    std::vector<lineCorrespWrapper> largestSet;
    lineSubsetStruct largestSetSubset;
    for(std::vector<lineSubsetStruct>::iterator it = subsets.begin() ; it != subsets.end(); ++it) {
        std::vector<lineCorrespWrapper> lineCorrespError;
        Mat H_T = it->Hs.t();
        Mat H_invT = it->Hs.inv(DECOMP_SVD).t();
        for(int i = 0; i < lineCorrespondencies.size(); i++) {
            lineCorrespWrapper lcw;
            //lcw.lineCorrespError = sqrt(squaredProjectionDistance(H_invT, H_T, lineCorrespondencies.at(i)));
            lcw.lineCorrespError = squaredProjectionDistance(H_invT, H_T, lineCorrespondencies.at(i));
            lcw.lineCorrespIdx = i;
            lineCorrespError.push_back(lcw);
        }
        std::sort(lineCorrespError.begin(), lineCorrespError.end(), compareLineCorrespWrapper);

        float lastError = lineCorrespError.begin()->lineCorrespError;
        for(std::vector<lineCorrespWrapper>::iterator lcwIter = lineCorrespError.begin() ; lcwIter != lineCorrespError.end(); ++lcwIter) {
            if(((lcwIter->lineCorrespError - lastError) / lcwIter->lineCorrespError) > threshold) { //If error growes more then threshold -> stop
                lineCorrespError.erase(lcwIter, lineCorrespError.end());    //Erase correspondencies with error > threshold
                break;
            }
            lastError = lcwIter->lineCorrespError;
        }

        if(lineCorrespError.size() > largestSet.size()) {
            largestSet = lineCorrespError;
            largestSetSubset = *it;
        }
    }

    for(std::vector<lineCorrespWrapper>::iterator it = largestSet.begin() ; it != largestSet.end(); ++it) {
        largestSetSubset.consensusCorrespIndices.push_back(it->lineCorrespIdx);
    }

    if(LOG_DEBUG) std::cout << "-- RANSAC inlaiers: " << largestSet.size() << std::endl;
    return largestSetSubset;
}

lineSubsetStruct FEstimatorLines::calcLMedS(std::vector<lineSubsetStruct> subsets) {
    if(LOG_DEBUG) std::cout << "-- Computing LMedS of " << subsets.size() << " Homographies" << std::endl;
    std::vector<lineSubsetStruct>::iterator it = subsets.begin();
    lineSubsetStruct lMedSsubset = *it;
    lMedSsubset.errorMeasure = calcMedS(it->Hs);
    it++;
    do {
        it->errorMeasure = calcMedS(it->Hs);
        //std::cout << meds << std::endl;
        if(it->errorMeasure < lMedSsubset.errorMeasure) {
            lMedSsubset = *it;
        }
        it++;
    } while(it != subsets.end());
    if(LOG_DEBUG) std::cout << "-- LMEDS: " << lMedSsubset.errorMeasure << std::endl;
    return lMedSsubset;
}


float FEstimatorLines::calcMedS(Mat Hs) {
    std::vector<float> dist;
    Mat H_invT = Hs.inv(DECOMP_SVD).t();
    Mat H_T = Hs.t();
    for(std::vector<lineCorrespStruct>::iterator it = lineCorrespondencies.begin() ; it != lineCorrespondencies.end(); ++it) {
        dist.push_back(squaredProjectionDistance(H_invT, H_T, *it));
        //dist.push_back(algebraicDistance(H_T, *it));
    }

    std::sort(dist.begin(), dist.end());

    return dist.at(dist.size()/2);    //TODO: change back
    //return absError/lineCorrespondencies.size();
}

double FEstimatorLines::squaredProjectionDistance(Mat H, lineCorrespStruct lc) {
    Mat H_invT = H.inv(DECOMP_SVD).t();
    Mat H_T = H.t();
    return squaredProjectionDistance(H_invT, H_T, lc);
}

double FEstimatorLines::squaredProjectionDistance(Mat H_invT, Mat H_T, lineCorrespStruct lc) {
    Mat A = H_T*crossProductMatrix(lc.line2Start)*lc.line2End;
    Mat start1 = lc.line1Start.t()*A;
    Mat end1 = lc.line1End.t()*A;
    Mat B = H_invT*crossProductMatrix(lc.line1Start)*lc.line1End;
    Mat start2 = lc.line2Start.t()*B;
    Mat end2 = lc.line2End.t()*B;
    Mat result = (start1*start1 + end1*end1)/(A.at<float>(0,0)*A.at<float>(0,0) + A.at<float>(1,0)*A.at<float>(1,0)) + (start2*start2 + end2*end2)/(B.at<float>(0,0)*B.at<float>(0,0) + B.at<float>(1,0)*B.at<float>(1,0));
    return result.at<float>(0,0);
}

double FEstimatorLines::algebraicDistance(Mat H_T, lineCorrespStruct lc) {
    Mat A = H_T*crossProductMatrix(lc.line2Start)*lc.line2End;
    Mat result = lc.line1Start.t()*A + lc.line1End.t()*A;
    return result.at<float>(0,0);
}

Mat* FEstimatorLines::normalizeLines(std::vector<lineCorrespStruct> &correspondencies) {

    //Normalization: Hartley, Zisserman, Multiple View Geometry in Computer Vision, p. 109

    Mat* normalizationMats = new Mat[2];
    float sum1x = 0, sum1y = 0, sum2x = 0, sum2y = 0, N = 0;

    for (std::vector<lineCorrespStruct>::iterator it= correspondencies.begin(); it!=correspondencies.end(); ++it) {

        sum1x += it->line1.startPointX + it->line1.endPointX;
        sum2x += it->line2.startPointX + it->line2.endPointX;

        sum1y += it->line1.startPointY + it->line1.endPointY;
        sum2y += it->line2.startPointY + it->line2.endPointY;

    }

    normalizationMats[0] = Mat::eye(3,3, CV_32FC1);
    normalizationMats[1] = Mat::eye(3,3, CV_32FC1);
    N = 2*correspondencies.size();

    normalizationMats[0].at<float>(0,0) = N/sum1x;
    normalizationMats[0].at<float>(1,1) = N/sum1y;
    normalizationMats[0].at<float>(0,2) = -1;
    normalizationMats[0].at<float>(1,2) = -1;

    normalizationMats[1].at<float>(0,0) = N/sum2x;
    normalizationMats[1].at<float>(1,1) = N/sum2y;
    normalizationMats[1].at<float>(0,2) = -1;
    normalizationMats[1].at<float>(1,2) = -1;

    //Carry out normalization:

    for (std::vector<lineCorrespStruct>::iterator it= correspondencies.begin(); it!=correspondencies.end(); ++it) {

        it->line1StartNormalized = normalizationMats[0]*it->line1Start;
        it->line2StartNormalized = normalizationMats[1]*it->line2Start;

        it->line1EndNormalized = normalizationMats[0]*it->line1End;
        it->line2EndNormalized = normalizationMats[1]*it->line2End;

    }

    return normalizationMats;
}

lineCorrespStruct FEstimatorLines::getlineCorrespStruct(float start1x, float start1y, float start2x, float start2y, float end1x, float end1y, float end2x, float end2y) {
    lineCorrespStruct* lc = new lineCorrespStruct;
    cv::line_descriptor::KeyLine l1, l2;
    lc->line1 = l1;
    lc->line2 = l2;
    lc->line1.startPointX = start1x;
    lc->line1.endPointX = end1x;
    lc->line1.startPointY = start1y;
    lc->line1.endPointY = end1y;

    lc->line2.startPointX = start2x;
    lc->line2.endPointX = end2x;
    lc->line2.startPointY = start2y;
    lc->line2.endPointY = end2y;

    lc->line1Start = matVector(start1x, start1y, 1);
    lc->line2Start = matVector(start2x, start2y, 1);
    lc->line1End = matVector(end1x, end1y, 1);
    lc->line2End = matVector(end2x, end2y, 1);

    return *lc;
}

bool compareLineCorrespWrapper(lineCorrespWrapper ls1, lineCorrespWrapper ls2) {
    return ls1.lineCorrespError < ls2.lineCorrespError;
}
