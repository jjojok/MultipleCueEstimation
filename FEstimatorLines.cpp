#include "FEstimatorLines.h"

FEstimatorLines::FEstimatorLines(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name) {
    image_1 = img1.clone();
    image_2 = img2.clone();
    image_1_color = img1_c.clone();
    image_2_color = img2_c.clone();
    this->name = name;
    std::cout << "Estimating: " << name << std::endl;
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

    Ptr<cv::line_descriptor::LSDDetector> lsd = cv::line_descriptor::LSDDetector::createLSDDetector();

    /* compute lines */
    std::vector<cv::line_descriptor::KeyLine> keylines1, keylines2;
    //bd->detect( image_1, keylines1, mask1 );
    //bd->detect( image_2, keylines2, mask2 );

    lsd->detect( image_1, keylines1, OCTAVES, SCALING, mask1 );
    lsd->detect( image_2, keylines2, OCTAVES, SCALING, mask2 );

    if(LOG_DEBUG) {
        std::cout << "-- First image before filter: " << keylines1.size() << std::endl;
        std::cout << "-- Second image before filter: " << keylines2.size() << std::endl;
    }

    //Filter detected lines:
    int filtered1 = 0;//filterLineExtractions(keylines1);
    int filtered2 = 0;//filterLineExtractions(keylines2);

    /* compute descriptors */
    cv::Mat descr1, descr2;
    bd->compute( image_1, keylines1, descr1 );
    bd->compute( image_2, keylines2, descr2 );

    std::cout << "-- First image: " << keylines1.size() << " filtered: " << filtered1 << std::endl;
    std::cout << "-- Second image: " << keylines2.size() << " filtered: " << filtered2 << std::endl;

    /* create a BinaryDescriptorMatcher object */
    Ptr<cv::line_descriptor::BinaryDescriptorMatcher> bdm = cv::line_descriptor::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

    /* require match */
    std::vector<DMatch> matches;
    bdm->match( descr1, descr2, matches );

    std::cout << "-- Number of matches : " << matches.size() << std::endl;

    if(VISUAL_DEBUG) {
        /* plot matches */
        cv::Mat outImg;
        std::vector<char> mask( matches.size(), 1 );
        cv::line_descriptor::drawLineMatches( image_1_color, keylines1, image_2_color, keylines2, matches, outImg, Scalar::all( -1 ), Scalar::all( -1 ), mask, cv::line_descriptor::DrawLinesMatchesFlags::DEFAULT );
        showImage("Line matches", outImg);
    }

    /************************************************************************/

    cv::line_descriptor::KeyLine l1, l2;
    for (std::vector<DMatch>::const_iterator it= matches.begin(); it!=matches.end(); ++it) {

        l1 = keylines1[it->queryIdx];
        l2 = keylines2[it->trainIdx];

        lineCorrespStruct lc;
        lc.line1 = l1;
        lc.line2 = l2;

        lc.line1Start = matVector(l1.startPointX, l1.startPointY, 1);
        lc.line2Start = matVector(l2.startPointX, l2.startPointY, 1);
        lc.line1End = matVector(l1.endPointX, l1.endPointY, 1);
        lc.line2End = matVector(l2.endPointX, l2.endPointY, 1);

        lineCorrespondencies.push_back(lc);

    }

    //showImage( "Matches", outImg, WINDOW_NORMAL, 1600);

    return lineCorrespondencies.size();

    std::cout << std::endl;
}

int FEstimatorLines::filterLineExtractions(std::vector<cv::line_descriptor::KeyLine> &keylines) {
    float minLenght = image_1.cols*image_1.rows*MIN_LENGTH_FACTOR;
    int filtered = 0;
    for (std::vector<cv::line_descriptor::KeyLine>::iterator it= keylines.begin(); it!=keylines.end(); ++it) {
        if (it->lineLength < minLenght) {
            keylines.erase(it);
            filtered++;
            if(it == keylines.end()) break;
        }
    }
    std::cout << keylines.size() << std::endl;
    return filtered;
}

Mat FEstimatorLines::compute() {

    //TODO: Hard coded test data for castle 4 & 5:

    lineCorrespStruct lc1, lc2, lc3, lc4;
//    lc1.line1StartNormalized = Mat::ones(3, 1, CV_32FC1);
//    lc1.line1EndNormalized = Mat::ones(3, 1, CV_32FC1);
//    lc1.line2EndNormalized = Mat::ones(3, 1, CV_32FC1);
//    lc1.line2StartNormalized = Mat::ones(3, 1, CV_32FC1);

//    lc2.line1StartNormalized = Mat::ones(3, 1, CV_32FC1);
//    lc2.line1EndNormalized = Mat::ones(3, 1, CV_32FC1);
//    lc2.line2EndNormalized = Mat::ones(3, 1, CV_32FC1);
//    lc2.line2StartNormalized = Mat::ones(3, 1, CV_32FC1);

//    lc3.line1StartNormalized = Mat::ones(3, 1, CV_32FC1);
//    lc3.line1EndNormalized = Mat::ones(3, 1, CV_32FC1);
//    lc3.line2EndNormalized = Mat::ones(3, 1, CV_32FC1);
//    lc3.line2StartNormalized = Mat::ones(3, 1, CV_32FC1);

//    lc4.line1StartNormalized = Mat::ones(3, 1, CV_32FC1);
//    lc4.line1EndNormalized = Mat::ones(3, 1, CV_32FC1);
//    lc4.line2EndNormalized = Mat::ones(3, 1, CV_32FC1);
//    lc4.line2StartNormalized = Mat::ones(3, 1, CV_32FC1);

//    lc1.line1StartNormalized.at<float>(0,0) = 343;
//    lc1.line1StartNormalized.at<float>(1,0) = 600;
//    lc1.line2StartNormalized.at<float>(0,0) = 311;
//    lc1.line2StartNormalized.at<float>(1,0) = 610;
//    lc1.line1EndNormalized.at<float>(0,0) = 287;
//    lc1.line1EndNormalized.at<float>(1,0) = 1217;
//    lc1.line2EndNormalized.at<float>(0,0) = 264;
//    lc1.line2EndNormalized.at<float>(1,0) = 1196;

//    lc2.line1StartNormalized.at<float>(0,0) = 219;
//    lc2.line1StartNormalized.at<float>(1,0) = 1318;
//    lc2.line2StartNormalized.at<float>(0,0) = 203;
//    lc2.line2StartNormalized.at<float>(1,0) = 1293;
//    lc2.line1EndNormalized.at<float>(0,0) = 1041;
//    lc2.line1EndNormalized.at<float>(1,0) = 1336;
//    lc2.line2EndNormalized.at<float>(0,0) = 984;
//    lc2.line2EndNormalized.at<float>(1,0) = 1291;

//    lc3.line1StartNormalized.at<float>(0,0) = 2628;
//    lc3.line1StartNormalized.at<float>(1,0) = 544;
//    lc3.line2StartNormalized.at<float>(0,0) = 2723;
//    lc3.line2StartNormalized.at<float>(1,0) = 386;
//    lc3.line1EndNormalized.at<float>(0,0) = 2752;
//    lc3.line1EndNormalized.at<float>(1,0) = 491;
//    lc3.line2EndNormalized.at<float>(0,0) = 2858;
//    lc3.line2EndNormalized.at<float>(1,0) = 317;

//    lc4.line1StartNormalized.at<float>(0,0) = 2641;
//    lc4.line1StartNormalized.at<float>(1,0) = 660;
//    lc4.line2StartNormalized.at<float>(0,0) = 2738;
//    lc4.line2StartNormalized.at<float>(1,0) = 513;
//    lc4.line1EndNormalized.at<float>(0,0) = 2717;
//    lc4.line1EndNormalized.at<float>(1,0) = 1358;
//    lc4.line2EndNormalized.at<float>(0,0) = 2824;
//    lc4.line2EndNormalized.at<float>(1,0) = 1278;

    lc1 = getlineCorrespStruct(343, 600, 311, 610, 287, 1217, 264, 1196);
    lc2 = getlineCorrespStruct(219, 1318, 203, 1293, 1041, 1336, 984, 1291);
    lc3 = getlineCorrespStruct(2628, 544, 2723, 386, 2752, 491, 2858, 317);
    lc4 = getlineCorrespStruct(2641, 660, 2738, 513, 2717, 1358, 2824, 1278);

    lineCorrespondencies.push_back(lc1);
    lineCorrespondencies.push_back(lc2);
    lineCorrespondencies.push_back(lc3);
    lineCorrespondencies.push_back(lc4);

    extractMatches();

    //lineCorrespondencies.clear();
    lineCorrespondencies.erase(lineCorrespondencies.begin()+5, lineCorrespondencies.end());   //TODO: REMOVE

    int numOfPairs = lineCorrespondencies.size();
    int numOfPairSubsets = NUM_LINE_PAIR_SUBSETS;//numOfPairs*NUM_OF_PAIR_SUBSETS_FACTOR;
    std::vector<lineSubsetStruct> subsets;

    //Compute H_21 from 4 line correspondencies
    for(int i = 0; i < numOfPairSubsets; i++) {
        std::vector<int> subsetsIdx;
        Mat linEq = Mat::ones(8,9,CV_32FC1);
        lineSubsetStruct subset;
        for(int j = 0; j < NUM_CORRESP; j++) {
            int subsetIdx = 0;
            do {        //Generate NUM_CORRESP uniqe random indices for line pair selection
                subsetIdx = std::rand() % numOfPairs;
            } while(std::find(subsetsIdx.begin(), subsetsIdx.end(), subsetIdx) != subsetsIdx.end());

            subsetsIdx.push_back(subsetIdx);
            subset.lineCorrespondencies.push_back(lineCorrespondencies.at(subsetIdx));
        }
        subset.lineCorrespondenceIdx = subsetsIdx;
        Mat* T = normalizeLines(subset.lineCorrespondencies);
        fillHLinEq(&linEq, subset.lineCorrespondencies);

        Mat A = linEq.colRange(0, linEq.cols-1);
        Mat x = -linEq.col(linEq.cols-1);
        solve(A, x, subset.Hs, DECOMP_SVD);
        subset.Hs.resize(9);
        subset.Hs = subset.Hs.reshape(1,3);
        subset.Hs.at<float>(2,2) = 1.0;
        subset.Hs = denormalize(subset.Hs, T[0], T[1]);
        subsets.push_back(subset);
    }

    F = calcLMedS(subsets);
    return F;

//    [4.8692675, 0.1330507, -1096.776;
//     1.6317964, 3.1708324, -2950.1914;
//     0.001353227, -3.3458778e-05, 1]

//    Mat result;
//    Mat linEq2 = Mat::ones(8,9,CV_32FC1);
//    fillHLinEq(&linEq2, lc1, 0);
//    fillHLinEq(&linEq2, lc2, 1);
//    fillHLinEq(&linEq2, lc3, 2);
//    fillHLinEq(&linEq2, lc4, 3);
//    Mat A = linEq2.colRange(0, linEq2.cols-1);
//    Mat x = -linEq2.col(linEq2.cols-1);
//    solve(A, x, result, DECOMP_SVD);
//    result.resize(9);
//    result = result.reshape(1,3);
//    result.at<float>(2,2) = 1;

//    //result =result.inv(DECOMP_SVD);


//    std::cout << "A result " << " = " << std::endl << A << std::endl;
//    std::cout << "x result " << " = " << std::endl << x << std::endl;
//    std::cout << "linEq result " << " = " << std::endl << result << std::endl;

//    result = denormalize(result);

//    result = result / result.at<float>(2,2);

//    std::cout << "linEq result after denormalization" << " = " << std::endl << result << std::endl;

//    F = result;
//    return result;
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

Mat FEstimatorLines::calcLMedS(std::vector<lineSubsetStruct> subsets) {
    std::vector<lineSubsetStruct>::iterator it = subsets.begin();
    float lMedS = calcMedS(it->Hs);
    Mat lMedSH = it->Hs;
    it++;
    do {
        float meds = calcMedS(it->Hs);
        std::cout << meds << std::endl;
        if(meds < lMedS) {
            lMedS = meds;
            lMedSH = it->Hs;
        }
        it++;
    } while(it != subsets.end());
    if(LOG_DEBUG) std::cout << "LMEDS: " << lMedS << std::endl;
    return lMedSH;
}


float FEstimatorLines::calcMedS(Mat Hs) {
    std::vector<float> dist;
    for(std::vector<lineCorrespStruct>::iterator it = lineCorrespondencies.begin() ; it != lineCorrespondencies.end(); ++it) {
//        Mat n1 = it->line1StartNormalized.cross(it->line1EndNormalized);
//        Mat n2real = it->line2StartNormalized.cross(it->line2EndNormalized);
//        Mat n2comp = Hs.inv(DECOMP_SVD).t()*n1;
//        dist.push_back(sqrt(squaredError(n2real,n2comp)[0]));
        dist.push_back(squaredDistance(Hs, *it));
        //squaredDistance(Hs, *it);
    }

    std::sort(dist.begin(), dist.end());

    return dist.at(dist.size()/2);
}

double FEstimatorLines::squaredDistance(Mat H, lineCorrespStruct lc) {
    Mat A = H.t()*lc.line2Start.cross(lc.line2End);
    Mat start1 = lc.line1Start.t()*A;
    Mat end1 = lc.line1End.t()*A;
    Mat B = H.inv(DECOMP_SVD).t()*lc.line1Start.cross(lc.line1End);
    Mat start2 = lc.line2Start.t()*B;
    Mat end2 = lc.line2End.t()*B;
    Mat result = (start1*start1 + end1*end1)/(A.at<float>(0,0)*A.at<float>(0,0) + A.at<float>(1,0)*A.at<float>(1,0)) + (start2*start2 + end2*end2)/(B.at<float>(0,0)*B.at<float>(0,0) + B.at<float>(1,0)*B.at<float>(1,0));
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
    normalizationMats[0].at<float>(0,2) = -1;//-sum1x/N;
    normalizationMats[0].at<float>(1,2) = -1;//-sum1y/N;

    normalizationMats[1].at<float>(0,0) = N/sum2x;
    normalizationMats[1].at<float>(1,1) = N/sum2y;
    normalizationMats[1].at<float>(0,2) = -1;//-sum2x/N;
    normalizationMats[1].at<float>(1,2) = -1;//-sum2y/N;

    //if(LOG_DEBUG) std::cout << "Normalization Matrix 1 = "<< std::endl << normalizationMats[0] << std::endl;
    //if(LOG_DEBUG) std::cout << "Normalization Matrix 2 = "<< std::endl << normalizationMats[1] << std::endl;

    //Carry normalization out:

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
    lc->line2End = matVector(end2x, end1y, 1);

    return *lc;
}
