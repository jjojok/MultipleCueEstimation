#include "Utility.h"

void enforceRankTwoConstraint(Mat &F) {
    //Enforce Rank 2 constraint:
    SVD svd;
    Mat u, vt, w;
    svd.compute(F, w, u, vt);
    Mat newW = Mat::zeros(3,3,CV_64FC1);
    newW.at<double>(0,0) = w.at<double>(0,0);
    newW.at<double>(1,1) = w.at<double>(1,0);
    F = u*newW*vt;
    F /= F.at<double>(2,2);
}

double fnorm(double x, double y) {
    return sqrt(pow(x, 2) + pow(y, 2));
}

double computeCamCenterDist(Mat F, Mat F_gt, Mat K1_gt, Mat K2_gt) {
    Mat R1, R2, R1_gt, R2_gt, t, t_gt;
    Mat E = K2_gt.t()*F*K1_gt;
    Mat E_gt = K2_gt.t()*F*K1_gt;

    decomposeEssentialMat(E, R1, R2, t);
    decomposeEssentialMat(E_gt, R1_gt, R2_gt, t_gt);
}

void visualizeHomography(Mat H21, Mat img1, Mat img2, std::string name) {
    Mat transformed;
    Mat result;
    warpPerspective(img1, transformed, H21, Size(img1.cols,img1.rows));
    hconcat(img2, transformed, result);
    showImage(name, result, WINDOW_NORMAL, 1600);
}

double smallestRelAngle(double ang1, double ang2) {
    double diff = fabs(ang1 - ang2);
    if(diff > 2.0*M_PI) diff = diff - 2.0*M_PI;
    if(diff > M_PI) diff = diff - M_PI;
    if(diff > M_PI/2.0) diff = M_PI - diff;
    return diff;
}

void showImage(std::string name, Mat image, int type, int width, int height) {
    double tx = 0;
    double ty = 0;
    Mat resized;
    if (width > 0) tx= (double)width/image.cols; {
        if (height > 0) ty= (double)height/image.rows;
        else ty= tx;
    }

    if(SHOW_DEBUG_IMG) {
        namedWindow(name, type);
        int method = INTER_LINEAR;
        if(tx < 1) method = INTER_AREA;
        resize(image, resized, Size(0,0), tx, ty, method);
        imshow(name, resized);
    }


    if(SAVE_DEBUG_IMG) writeImage(name, image);


}

void writeImage(std::string file, Mat img) {
    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(3);

    std::replace(file.begin(), file.end(), ':', '_');
    std::replace(file.begin(), file.end(), ' ', '_');

    try {
        imwrite(file+".png", img, compression_params);
    } catch(...) {}
//    catch (std::runtime_error& ex) {
//        if(LOG_DEBUG) fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
//    }
}

std::string getType(Mat m) {
    std::string type = "Type: ";
    switch(m.type() & Mat::TYPE_MASK) {
        case CV_8U: "CV_8U"; break;
        case CV_8S: type+="CV_8U";  break;
        case CV_16U: type+="CV_16U"; break;
        case CV_16S: type+="CV_16S"; break;
        case CV_32S: type+="CV_32S"; break;
        case CV_32F: type+="CV_32F"; break;
        case CV_64F: type+="CV_64F"; break;
        default: type+="unknown"; break;
    }
    return type;
}

void drawEpipolarLines(std::vector<Point2d> p1, std::vector<Point2d> p2, Mat F, Mat img1, Mat img2, std::string name) {

    if(p1.size() == 0 || p2.size() == 0) return;

    double maxLines = std::min(100, (int)p1.size());

    Mat image1 = img1.clone();
    Mat image2 = img2.clone();

    //#################################################################################
    //From: http://opencv-cookbook.googlecode.com/svn/trunk/Chapter%2009/estimateF.cpp
    //#################################################################################

    std::vector<cv::Vec3d> lines1, lines2;
    cv::computeCorrespondEpilines(p1, 1, F, lines1);
//    for (std::vector<cv::Vec3d>::const_iterator it= lines1.begin();
//         it!=lines1.end(); ++it) {
    for(int i = 0; i < maxLines; i++) {
        cv::Vec3d* it = &lines1.at(i);
             cv::line(image2,cv::Point(0,-(*it)[2]/(*it)[1]),
                             cv::Point(image2.cols,-((*it)[2]+(*it)[0]*image2.cols)/(*it)[1]),
                             cv::Scalar(255,255,255));
    }

    cv::computeCorrespondEpilines(p2,2,F,lines2);
//    for (std::vector<cv::Vec3d>::const_iterator it= lines2.begin();
//         it!=lines2.end(); ++it) {
    for(int i = 0; i < maxLines; i++) {
            cv::Vec3d* it = &lines2.at(i);

             cv::line(image1,cv::Point(0,-(*it)[2]/(*it)[1]),
                             cv::Point(image1.cols,-((*it)[2]+(*it)[0]*image1.cols)/(*it)[1]),
                             cv::Scalar(255,255,255));
    }

    // Draw points
//    std::vector<cv::Point2d>::const_iterator itPts= p1.begin();
//    while (itPts!=p1.end()) {
    for(int i = 0; i < maxLines; i++) {
            cv::circle(image1,p1.at(i),3,cv::Scalar(255,255,255),2);
        //++itPts;
    }

    //itPts= p2.begin();
    //while (itPts!=p2.end()) {
    for(int i = 0; i < maxLines; i++) {
            cv::circle(image2,p2.at(i),3,cv::Scalar(255,255,255),2);
        //++itPts;
    }

    // Display the images with points

    showImage(name+" 1",image1);
    showImage(name+" 2",image2);

    //#############################################################################
}

Mat crossProductMatrix(Mat input) {    //3 Vector to cross product matrix
    Mat crossMat = Mat::zeros(3,3, input.type());
    crossMat.at<double>(0,1) = -input.at<double>(2,0);
    crossMat.at<double>(0,2) = input.at<double>(1,0);
    crossMat.at<double>(1,0) = input.at<double>(2,0);
    crossMat.at<double>(1,2) = -input.at<double>(0,0);
    crossMat.at<double>(2,0) = -input.at<double>(1,0);
    crossMat.at<double>(2,1) = input.at<double>(0,0);
    return crossMat;
}

void rectify(std::vector<Point2d> p1, std::vector<Point2d> p2, Mat F, Mat image1, Mat image2, std::string windowName) {
    Mat H1, H2, rectified1, rectified2;
    if(stereoRectifyUncalibrated(p1, p2, F, Size(image1.cols,image1.rows), H1, H2, 2 )) {

        warpPerspective(image1, rectified1, H1, Size(image1.cols,image1.rows));
        warpPerspective(image2, rectified2, H2, Size(image1.cols,image1.rows));

        showImage(windowName+" 1", rectified1);
        showImage(windowName+" 2", rectified2);
    }
}

Mat MatFromFile(std::string file, int rows) {

    Mat matrix = Mat::zeros(0,0,CV_64FC1);
    std::ifstream inputStream;
    double x;
    inputStream.open(file.c_str());
    if (inputStream.is_open()) {
        while(inputStream >> x) {
            matrix.push_back(x);
        }
        matrix = matrix.reshape(1, rows);
        inputStream.close();
    } else {
        std::cerr << "Unable to open file: " << file;
    }
    return matrix;
}

void matToFile(std::string file, Mat m) {
    std::ofstream outputStream;
    outputStream.open(file.c_str());
    if (outputStream.is_open()) {
        outputStream << m;
        outputStream.close();
    } else {
        std::cerr << "Unable to open file: " << file;
    }
}

bool ImgParamsFromFile(std::string file, Mat &K, Mat &R, Mat &t) {
    int values = 1;
    std::ifstream inputStream;
    double x;
    inputStream.open(file.c_str());
    if (inputStream.is_open()) {
        while(inputStream >> x) {
            if(values >= 1 && values <= 9) {    //K
                K.push_back(x);
            } else if(values >= 13 && values <= 21) { //R
                R.push_back(x);
            } else if(values >= 22 && values <= 24) { //t
                t.push_back(x);
            }
            values++;
        }
        inputStream.close();
        K = K.reshape(1, 3);
        R = R.reshape(1, 3);
        t = t.reshape(1, 3);
        return true;
    } else {
        std::cerr << "Unable to open file: " << file <<std::endl;
        return false;
    }
}

Mat matVector(double x, double y, double z) {
    Mat vect = Mat::zeros(3,1,CV_64FC1);
    vect.at<double>(0,0) = x;
    vect.at<double>(1,0) = y;
    vect.at<double>(2,0) = z;
    return vect;
}

Mat matVector(Point2d p) {
    Mat vect = Mat::zeros(3,1,CV_64FC1);
    vect.at<double>(0,0) = p.x;
    vect.at<double>(1,0) = p.y;
    vect.at<double>(2,0) = 1;
    return vect;
}

lineCorrespStruct getlineCorrespStruct(lineCorrespStruct lcCopy) {
    lineCorrespStruct* lc = new lineCorrespStruct;
    lc->line1Angle = lcCopy.line1Angle;
    lc->line2Angle = lcCopy.line2Angle;

    lc->line1Length = lcCopy.line1Length;
    lc->line2Length = lcCopy.line2Length;

    lc->line1Start = lcCopy.line1Start.clone();
    lc->line2Start = lcCopy.line2Start.clone();
    lc->line1End = lcCopy.line1End.clone();
    lc->line2End = lcCopy.line2End.clone();

    lc->line1StartNormalized = lcCopy.line1StartNormalized.clone();
    lc->line2StartNormalized = lcCopy.line2StartNormalized.clone();
    lc->line1EndNormalized = lcCopy.line1EndNormalized.clone();
    lc->line2EndNormalized = lcCopy.line2EndNormalized.clone();

    lc->isGoodMatch = lcCopy.isGoodMatch;

    lc->id = lcCopy.id;

    return *lc;
}

pointCorrespStruct getPointCorrespStruct(pointCorrespStruct pcCopy) {
    pointCorrespStruct* pc = new pointCorrespStruct;
    pc->x1.x = pcCopy.x1.x;
    pc->x2.x = pcCopy.x2.x;

    pc->x1norm = pcCopy.x1norm.clone();
    pc->x2norm = pcCopy.x2norm.clone();

    pc->x1.y = pcCopy.x1.y;
    pc->x2.y = pcCopy.x2.y;

    pc->id = pcCopy.id;

    pc->isGoodMatch = pcCopy.isGoodMatch;

    return *pc;
}

lineCorrespStruct getlineCorrespStruct(cv::line_descriptor::KeyLine l1, cv::line_descriptor::KeyLine l2, int id) {
    lineCorrespStruct* lc = new lineCorrespStruct;
    double scaling1 = 1;
    double scaling2 = 1;
    if(l1.octave > 0) { //OpenCV: coordinates are from downscaled versions of image pyramid
        scaling1 = l1.octave*SCALING;
    }
    if(l2.octave > 0) {
        scaling2 = l2.octave*SCALING;
    }

    lc->line1Angle = l1.angle;
    lc->line2Angle = l2.angle;

    lc->line1Length = l1.lineLength*scaling1;
    lc->line2Length = l2.lineLength*scaling2;

    lc->line1Start = matVector(l1.startPointX*scaling1, l1.startPointY*scaling1, 1);
    lc->line2Start = matVector(l2.startPointX*scaling2, l2.startPointY*scaling2, 1);
    lc->line1End = matVector(l1.endPointX*scaling1, l1.endPointY*scaling1, 1);
    lc->line2End = matVector(l2.endPointX*scaling2, l2.endPointY*scaling2, 1);

    lc->id = id;

    return *lc;
}

lineCorrespStruct getlineCorrespStruct(double start1x, double start1y, double end1x, double end1y, double start2x, double start2y , double end2x, double end2y, int id) {
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

void visualizeLineMatches(Mat image_1_color, Mat image_2_color, std::vector<lineCorrespStruct> correspondencies, int lineWidth, bool drawConnections, std::string name) {
    Mat img;
    hconcat(image_1_color.clone(), image_2_color.clone(), img);
    for(std::vector<lineCorrespStruct>::iterator it = correspondencies.begin() ; it != correspondencies.end(); ++it) {
        //Scalar color = Scalar(rand()%255, rand()%255, rand()%255);
        Scalar color = Scalar(0, 0, 255);
        cv::line(img, cvPoint2D32f(it->line1Start.at<double>(0,0), it->line1Start.at<double>(1,0)), cvPoint2D32f(it->line1End.at<double>(0,0), it->line1End.at<double>(1,0)), color, lineWidth);
        cv::line(img, cvPoint2D32f(it->line2Start.at<double>(0,0) + image_1_color.cols, it->line2Start.at<double>(1,0)), cvPoint2D32f(it->line2End.at<double>(0,0) + image_1_color.cols, it->line2End.at<double>(1,0)), color, lineWidth);
        if(drawConnections) {
            cv::line(img, cvPoint2D32f(it->line1Start.at<double>(0,0), it->line1Start.at<double>(1,0)), cvPoint2D32f(it->line2Start.at<double>(0,0) + image_1_color.cols, it->line2Start.at<double>(1,0)), color, lineWidth);
        }
    }
    showImage(name, img, WINDOW_NORMAL, 1600);
}

void visualizePointMatches(Mat image_1_color, Mat image_2_color, std::vector<Point2d> p1, std::vector<Point2d> p2, int radius, int lineWidth, bool drawConnections, std::string name) {
    Mat img;
    hconcat(image_1_color.clone(), image_2_color.clone(), img);
    for(int i = 0; i < p1.size(); i++) {
        //Scalar color = Scalar(rand()%255, rand()%255, rand()%255);
        Scalar color = Scalar(0, 0, 255);
        cv::circle(img, p1.at(i), radius, color, lineWidth);
        cv::circle(img, cvPoint2D32f(p2.at(i).x + image_1_color.cols, p2.at(i).y), radius, color, lineWidth);
        if(drawConnections) {
            cv::line(img, p1.at(i), cvPoint2D32f(p2.at(i).x + image_1_color.cols, p2.at(i).y), color, lineWidth);
        }
    }
    showImage(name, img, WINDOW_NORMAL, 1600);
}

void visualizePointMatches(Mat image_1_color, Mat image_2_color, std::vector<pointCorrespStruct> pointCorresp, int radius, int lineWidth, bool drawConnections, std::string name) {
    Mat img;
    hconcat(image_1_color.clone(), image_2_color.clone(), img);
    for(int i = 0; i < pointCorresp.size(); i++) {
        Point2f p1, p2;
        p1 = pointCorresp.at(i).x1;
        p2 = pointCorresp.at(i).x2;

        //Scalar color = Scalar(rand()%255, rand()%255, rand()%255);
        Scalar color = Scalar(0, 0, 255);
        cv::circle(img, p1, radius, color, lineWidth);
        cv::circle(img, cvPoint2D32f(p2.x + image_1_color.cols, p2.y), radius, color, lineWidth);
        if(drawConnections) {
            cv::line(img, p1, cvPoint2D32f(p2.x + image_1_color.cols, p2.y), color, lineWidth);
        }
    }
    showImage(name, img, WINDOW_NORMAL, 1600);
}

void visualizePointMatches(Mat image_1_color, Mat image_2_color, std::vector<Mat> x1, std::vector<Mat> x2, int radius, int lineWidth, bool drawConnections, std::string name) {
    Mat img;
    hconcat(image_1_color.clone(), image_2_color.clone(), img);
    for(int i = 0; i < x1.size(); i++) {
        Mat p1 = x1.at(i);
        Mat p2 = x2.at(i);

        //Scalar color = Scalar(rand()%255, rand()%255, rand()%255);
        Scalar color = Scalar(0, 0, 255);
        cv::circle(img, cvPoint2D32f(p1.at<double>(0,0), p1.at<double>(1,0)), radius, color, lineWidth);
        cv::circle(img, cvPoint2D32f(p2.at<double>(0,0) + image_1_color.cols, p2.at<double>(1,0)), radius, color, lineWidth);
        if(drawConnections) {
            cv::line(img, cvPoint2D32f(p1.at<double>(0,0), p1.at<double>(1,0)), cvPoint2D32f(p2.at<double>(0,0) + image_1_color.cols, p2.at<double>(1,0)), color, lineWidth);
        }
    }
    showImage(name, img, WINDOW_NORMAL, 1600);
}

bool isUnity(Mat m) {       //Check main diagonal for being close to 1
    Mat diff = abs(m - Mat::eye(m.rows, m.cols, CV_64FC1));
    for(int i = 0; i < m.cols; i++) {
        //for(int j = 0; j < m.rows; j++) {
            if(fabs(diff.at<double>(i,i)) > 0.1) {
                if(LOG_DEBUG) std::cout << "-- Is Unity matrix: false" << std::endl;
                return false;
            }
        //}
    }
    if(LOG_DEBUG) std::cout << "-- Is Unity matrix: true" << std::endl;
    return true;
}

bool computeUniqeEigenvector(Mat H, Mat &e) {
    // Map the OpenCV matrix with Eigen:
    Eigen::Matrix3d HEigen;
    cv2eigen(H, HEigen);
    //http://eigen.tuxfamily.org/dox/classEigen_1_1EigenSolver.html#a8c287af80cfd71517094b75dcad2a31b
    Eigen::EigenSolver<Eigen::Matrix3d> solver;
    solver.compute(HEigen);

    Mat eigenvalues = Mat::zeros(3, 1, CV_64FC1), eigenvectors = Mat::zeros(3, 3, CV_64FC1);
    eigen2cv(solver.eigenvalues(), eigenvalues);
    eigen2cv(solver.eigenvectors(), eigenvectors);

    if(LOG_DEBUG) {
        std::cout << "-- H1*H2^-1 = " << std::endl << H << std::endl;

        for(int i = 0; i < eigenvalues.rows; i++) {
            std::cout << "-- " << i+1 << "th Eigenvalue: " << eigenvalues.at<double>(i,0) << ", Eigenvector = " << std::endl << eigenvectors.col(i) << std::endl;
        }
    }

    bool eigenvalueOK = false;
    double dist[eigenvalues.rows];
    double lastDist = 0;
    int col = 0;
    for(int i = 0; i < eigenvalues.rows; i ++) {        //find non-unary eigenvalue & its eigenvector
        Mat eig = eigenvalues.row(i);
        if(eig.at<double>(0,0) < MARGIN) {
            if(LOG_DEBUG) std::cout << "-- Eigenvalue to small: " << eig.at<double>(0,0) << std::endl;
            return false;
        }
        dist[i] = 0;
        for(int j = 0; j < eigenvalues.rows; j ++) {
            dist[i] += fabs(eig.at<double>(0,0) - eigenvalues.row(j).at<double>(0,0));
        }
        if(dist[i] > MARGIN)
            eigenvalueOK = true;
        if(dist[i] > lastDist) {
            col = i;
            lastDist = dist[i];
        }
    }

    if(LOG_DEBUG) std::cout << "-- Largest eigenvalue distance: " << lastDist << std::endl;

    if(LOG_DEBUG && eigenvalueOK) std::cout << "-- Found uniqe eigenvalue: " << eigenvalues.row(col).at<double>(0,0) << std::endl;
    if(LOG_DEBUG && !eigenvalueOK) std::cout << "-- Found no uniqe eigenvalue!" << std::endl;

    double eigenValDiff = 0;
    if(col == 0) eigenValDiff = eigenvalues.row(1).at<double>(0,0) - eigenvalues.row(2).at<double>(0,0);
    if(col == 1) eigenValDiff = eigenvalues.row(0).at<double>(0,0) - eigenvalues.row(2).at<double>(0,0);
    if(col == 2) eigenValDiff = eigenvalues.row(0).at<double>(0,0) - eigenvalues.row(1).at<double>(0,0);

    if(fabs(eigenValDiff) > 0.2) {
        if(LOG_DEBUG) std::cout << "-- Other eigenvalues are not equal!" << std::endl;
        return false;
    }

    if(!eigenvalueOK) {
        if(isUnity(H)) return false;
        eigenvalueOK = true;
    }

    std::vector<Mat> e2;
    split(eigenvectors.col(col),e2); //Remove channel for imaginary part
    if(LOG_DEBUG) std::cout << "-- e = " << std::endl << e2.at(0) << std::endl;

    Mat(e2.at(0)).copyTo(e);

    return eigenvalueOK;
}

std::vector<double> computeCombinedSquaredErrorVect(std::vector<Mat> x1, std::vector<Mat> x2, Mat F) {

    std::vector<double> *errorVect = new std::vector<double>();

    for(int i = 0; i < x1.size(); i++) {
        Mat p1 = x1.at(i);
        Mat p2 = x2.at(i);
        errorVect->push_back(sampsonDistanceFundamentalMat(F, p1, p2));
    }
    return *errorVect;
}

void errorFunctionCombinedMeanSquared(std::vector<Mat> x1, std::vector<Mat> x2, Mat impF, double &error, int &inliers, double inlierThr, double &standardDeviation) {
    std::vector<double> errorVect = computeCombinedSquaredErrorVect(x1, x2, impF);
    error = 0;
    inliers = 0;
    standardDeviation = 0;
    for(std::vector<double>::const_iterator errorIter = errorVect.begin(); errorIter != errorVect.end(); ++errorIter) {
        error += *errorIter;
        if(sqrt(*errorIter) < inlierThr) inliers++;
    }
    error = error/((double)errorVect.size());
    for(std::vector<double>::const_iterator errorIter = errorVect.begin(); errorIter != errorVect.end(); ++errorIter) {
        standardDeviation += std::pow(*errorIter - error, 2);
    }
    standardDeviation = standardDeviation/((double)errorVect.size() - 1.0);
    standardDeviation = std::sqrt(standardDeviation);
}

void errorFunctionCombinedMean(std::vector<Mat> x1, std::vector<Mat> x2, Mat impF, double &error, int &inliers, double inlierThr, double &standardDeviation) {
    std::vector<double> errorVect = computeCombinedSquaredErrorVect(x1, x2, impF);
    error = 0;
    inliers = 0;
    standardDeviation = 0;
    for(std::vector<double>::const_iterator errorIter = errorVect.begin(); errorIter != errorVect.end(); ++errorIter) {
        error += sqrt(*errorIter);
        if(sqrt(*errorIter) <= inlierThr) inliers++;
    }
    error = error/((double)errorVect.size());
    for(std::vector<double>::const_iterator errorIter = errorVect.begin(); errorIter != errorVect.end(); ++errorIter) {
        standardDeviation += std::pow(*errorIter - error, 2);
    }
    standardDeviation = standardDeviation/((double)errorVect.size() - 1.0);
    standardDeviation = std::sqrt(standardDeviation);
}

void findGoodCombinedMatches(std::vector<Mat> x1Combined, std::vector<Mat> x2Combined, std::vector<Mat> &x1, std::vector<Mat> &x2, Mat F, double maxDist) {
    x1.clear();
    x2.clear();
    for(int i = 0; i < x1Combined.size(); i++) {
        if(sqrt(sampsonDistanceFundamentalMat(F, x1Combined.at(i), x2Combined.at(i))) < maxDist) {
            x1.push_back(x1Combined.at(i));
            x2.push_back(x2Combined.at(i));
        }
    }
}

void findGoodCombinedMatches(std::vector<Point2d> x1Combined, std::vector<Point2d> x2Combined, std::vector<Point2d> &x1, std::vector<Point2d> &x2, Mat F, double maxDist, double minDist) {
    x1.clear();
    x2.clear();
    for(int i = 0; i < x1Combined.size(); i++) {
        double err = sqrt(sampsonDistanceFundamentalMat(F, matVector(x1Combined.at(i)), matVector(x2Combined.at(i))));
        if(minDist < err && err < maxDist) {
            x1.push_back(x1Combined.at(i));
            x2.push_back(x2Combined.at(i));
        }
    }
}

void computeEpipoles(Mat F, Mat &e1, Mat &e2) {     //See Hartley, Ziss p.246

    /**********************************
     * http://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
     * ********************************/

    e1 = Mat::zeros(3, 1, CV_64FC1);
    e2 = Mat::zeros(3, 1, CV_64FC1);
    Eigen::Matrix3d eigenF;
    Eigen::Matrix3d eigenF_T;
    Eigen::Vector3d b;
    b << 0,0,0;
    cv2eigen(F, eigenF);
    cv2eigen(F.t(), eigenF_T);
    Eigen::FullPivLU<Eigen::Matrix3d> lu_decomp(eigenF);
    Eigen::FullPivLU<Eigen::Matrix3d> lu_decompT(eigenF_T);
    Eigen::Vector3d eigenE1 = lu_decomp.kernel();
    Eigen::Vector3d eigenE2 = lu_decompT.kernel();
    eigen2cv(eigenE1, e1);
    eigen2cv(eigenE2, e2);
}

double computeRelativeOutliers(double generalOutliers, double uesdCorresp, double correspCount) {
    double outliers = generalOutliers*(uesdCorresp/correspCount);
    if(LOG_DEBUG) std::cout << "-- Filtering matches, new outlier/matches ratio: " << outliers << std::endl;
    return outliers;
}

int computeNumberOfEstimations(double confidence, double outliers, int corrspNumber) {
    int num = std::min(MAX_NUM_COMPUTATIONS, (int)std::ceil((std::log(1.0 - confidence)/std::log(1.0 - std::pow(1.0 - outliers, corrspNumber))))); //See Hartley, Zisserman p119
    if (num < 0) return MAX_NUM_COMPUTATIONS;
    return num;
}

bool isUniqe(std::vector<int> subsetsIdx, int newIdx) {
    if(subsetsIdx.size() == 0) return true;
    for(std::vector<int>::const_iterator iter = subsetsIdx.begin(); iter != subsetsIdx.end(); ++iter) {
        if(*iter == newIdx) return false;
    }
    return true;
}

void homogMat(Mat &m) {
    m /= m.at<double>(m.rows-1,m.cols-1);
}

Mat* normalize(std::vector<Mat> x1, std::vector<Mat> x2, std::vector<Mat> &x1norm, std::vector<Mat> &x2norm) {

    //Normalization: Hartley, Zisserman, Multiple View Geometry in Computer Vision, p. 109

    Mat* normalizationMats = new Mat[2];
    Mat sum1 = Mat::zeros(3,1,CV_64FC1), sum2 = Mat::zeros(3,1,CV_64FC1);
    Mat mean1, mean2;
    double N = 0;
    double mean1x = 0, mean1y = 0, mean2x = 0, mean2y = 0, v1 = 0, v2 = 0, scale1 = 0, scale2 = 0;

    for (int i = 0; i < x1.size(); i++) {

        sum1 += x1.at(i);
        sum2 += x2.at(i);

    }

    normalizationMats[0] = Mat::eye(3,3, CV_64FC1);
    normalizationMats[1] = Mat::eye(3,3, CV_64FC1);
    N = x1.size();

    mean1 = sum1/N;
    mean2 = sum2/N;

    mean1x = mean1.at<double>(0,0);
    mean1y = mean1.at<double>(1,0);
    mean2x = mean2.at<double>(0,0);
    mean2y = mean2.at<double>(1,0);

    for (int i = 0; i < x1.size(); i++) {
        v1 += fnorm(x1.at(i).at<double>(0,0)-mean1x, x1.at(i).at<double>(1,0)-mean1y);
        v2 += fnorm(x2.at(i).at<double>(0,0)-mean2x, x2.at(i).at<double>(1,0)-mean2y);
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

    if(LOG_DEBUG) std::cout << "-- Normalization: " << std::endl <<"-- T1 = " << std::endl << normalizationMats[0] << std::endl << "-- T2 = " << std::endl << normalizationMats[1] << std::endl;

    //Carry out normalization:

    for (int i = 0; i < x1.size(); i++) {

        x1norm.push_back(normalizationMats[0]*x1.at(i));
        x2norm.push_back(normalizationMats[1]*x2.at(i));

    }

    return normalizationMats;
}

void meanSampsonFDistanceGoodMatches(Mat Fgt, Mat F, std::vector<Mat> x1, std::vector<Mat> x2, double &error, int &used) {
    error = 0;
    used = 0;
    for(int i = 0; i < x1.size(); i++) {
        if(sqrt(sampsonDistanceFundamentalMat(Fgt, x1.at(i), x2.at(i))) <= INLIER_THRESHOLD) {
            error += sampsonDistanceFundamentalMat(F, x1.at(i), x2.at(i));
            used++;
        }
    }
    error/=used;
    if(LOG_DEBUG) std::cout << "-- Computed sampson distance for " << used << "/" << x1.size() << " points: " << error << std::endl;
}

int goodMatchesCount(Mat Fgt, std::vector<Mat> x1, std::vector<Mat> x2, double thr) {
    int used = 0;
    for(int i = 0; i < x1.size(); i++) {
        if(sqrt(sampsonDistanceFundamentalMat(Fgt, x1.at(i), x2.at(i))) <= thr) {
            used++;
        }
    }
    return used;
}

double meanSampsonFDistanceGoodMatches(Mat Fgt, Mat F, std::vector<Mat> x1, std::vector<Mat> x2) {
    double error = 0;
    double used = 0;
    for(int i = 0; i < x1.size(); i++) {
        if(sqrt(sampsonDistanceFundamentalMat(Fgt, x1.at(i), x2.at(i))) <= INLIER_THRESHOLD) {
            error += sampsonDistanceFundamentalMat(F, x1.at(i), x2.at(i));
            used++;
        }
    }
    return error/used;
}

bool isEqualPointCorresp(Mat x11, Mat x12, Mat x21, Mat x22) {
    if(fabs(x11.at<double>(0,0) - x21.at<double>(0,0)) > 1.0) return false;
    if(fabs(x11.at<double>(1,0) - x21.at<double>(1,0)) > 1.0) return false;
    if(fabs(x12.at<double>(0,0) - x22.at<double>(0,0)) > 1.0) return false;
    if(fabs(x12.at<double>(1,0) - x22.at<double>(1,0)) > 1.0) return false;
    return true;
}

void matToPoint(std::vector<Mat> xin, std::vector<Point2d> &xout) {
    for(int i = 0; i < xin.size(); i++) {
        Point2f p;
        p.x = xin.at(i).at<double>(0,0);
        p.y = xin.at(i).at<double>(1,0);
        xout.push_back(p);
    }
}

//error functions

//Fundamental matrix

double sampsonDistanceFundamentalMat(Mat F, Mat x1, Mat x2) {
    return sampsonDistanceFundamentalMatSingle(F, x1, x2);
}

double sampsonDistanceFundamentalMatSingle(Mat F, Mat x1, Mat x2) {      //See: Hartley Ziss, p287
    double n = Mat(x2.t()*F*x1).at<double>(0,0);
    Mat b1 = F*x1;
    Mat b2 = F.t()*x2;
    return std::pow(n, 2)/(std::pow(b1.at<double>(0,0), 2) + std::pow(b1.at<double>(1,0), 2) + std::pow(b2.at<double>(0,0), 2) + std::pow(b2.at<double>(1,0), 2));
}

double sampsonDistanceFundamentalMat(Mat F, std::vector<Point2d> points1, std::vector<Point2d> points2) {
    double error = 0;
    for(int i = 0; i < points1.size(); i++) {
        error+=sampsonDistanceFundamentalMat(F, matVector(points1.at(i)), matVector(points2.at(i)));
    }
    return error/points1.size();
}

double sampsonDistanceFundamentalMat(Mat F, std::vector<Mat> points1, std::vector<Mat> points2) {
    double error = 0;
    for(int i = 0; i < points1.size(); i++) {
        error+=sampsonDistanceFundamentalMat(F, points1.at(i), points2.at(i));
    }
    return error/points1.size();
}

//error point homogrpahy

double sampsonDistanceHomographySingle(Mat H, Mat x1, Mat x2) {

    Mat E = Mat::zeros(2,1,CV_64FC1);
    Mat J = Mat::zeros(2,4,CV_64FC1);

    Mat h1 = H.row(0);
    Mat h2 = H.row(1);
    Mat h3 = H.row(2);

    //std::cout << Mat(x1*h2) << std::endl;

    E.at<double>(0,0) = Mat(-x2.at<double>(2,0)*x1.t()*h2.t() + x2.at<double>(1,0)*x1.t()*h3.t()).at<double>(0,0);
    E.at<double>(1,0) = Mat(x2.at<double>(2,0)*x1.t()*h1.t() - x2.at<double>(0,0)*x1.t()*h3.t()).at<double>(0,0);

    //std::cout << E << std::endl;

    //dE/dx
    J.at<double>(0,0) = -x2.at<double>(2,0)*h2.at<double>(0,0) + x2.at<double>(1,0)*h3.at<double>(0,0);
    J.at<double>(1,0) = x2.at<double>(2,0)*h1.at<double>(0,0) + x2.at<double>(0,0)*h3.at<double>(0,0);
    //dE/dy
    J.at<double>(0,1) = -x2.at<double>(2,0)*h2.at<double>(0,1) + x2.at<double>(1,0)*h3.at<double>(0,1);
    J.at<double>(1,1) = x2.at<double>(2,0)*h1.at<double>(0,1) + x2.at<double>(0,0)*h3.at<double>(0,1);
    //dE/dx'
    J.at<double>(0,2) = 0;
    J.at<double>(1,2) = -x1.at<double>(0,0)*h3.at<double>(0,0) - x1.at<double>(1,0)*h3.at<double>(0,1) - x1.at<double>(2,0)*h3.at<double>(0,2);
    //dE/dy'
    J.at<double>(0,3) = x1.at<double>(0,0)*h3.at<double>(0,0) + x1.at<double>(1,0)*h3.at<double>(0,1) + x1.at<double>(2,0)*h3.at<double>(0,2);
    J.at<double>(1,3) = 0;

    Mat error = E.t()*(J*J.t()).inv(DECOMP_SVD)*E;

    return error.at<double>(0,0);
}

double sampsonDistanceHomography(Mat H, Mat H_inv, Mat x1, Mat x2) {
    return sampsonDistanceHomographySingle(H, x1, x2);
}

//error line homography

double sampsonDistanceHomographySingle(Mat H, Mat line1Start, Mat line1End, Mat line2Start, Mat line2End) {

//    //Geometric error
    Mat AA = H.t()*crossProductMatrix(line2Start)*line2End;
    double start = Mat(line1Start.t()*AA).at<double>(0,0);
    double end = Mat(line1End.t()*AA).at<double>(0,0);
    double Ax = AA.at<double>(0,0);
    double Ay = AA.at<double>(1,0);
    return (start*start + end*end)/(Ax*Ax + Ay*Ay);
}

double sampsonDistanceHomography(Mat H, Mat H_inv, Mat line1Start, Mat line1End, Mat line2Start, Mat line2End) {
    return sampsonDistanceHomographySingle(H, line1Start, line1End, line2Start, line2End) + sampsonDistanceHomographySingle(H_inv, line2Start, line2End, line1Start, line1End);
}
