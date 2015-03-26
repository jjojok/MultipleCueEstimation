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

void visualizeHomography(Mat H21, Mat img1, Mat img2, std::string name) {
    Mat transformed;
    Mat result;
    warpPerspective(img1, transformed, H21, Size(img1.cols,img1.rows));
    hconcat(img2, transformed, result);
    showImage(name, result, WINDOW_NORMAL, 1600);
}

double smallestRelAngle(double ang1, double ang2) {
    double diff = fabs(ang1 - ang2);
    if(diff > M_PI) diff = (2*M_PI) - diff;
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
    namedWindow(name, type);
    int method = INTER_LINEAR;
    if(tx < 1) method = INTER_AREA;
    resize(image, resized, Size(0,0), tx, ty, method);
    imshow(name, resized);
}

double squaredError(Mat A, Mat B) {
    return cv::sum((A-B).mul(A-B))[0];
}

int calcMatRank(Mat M) {
    Mat U,V,W;
    int rank = 0, diag;
    SVD::compute(M,U,V,W);
    if (W.cols < W.rows) diag = W.cols;
    else diag = W.rows;
    for(int i = 0; i < diag; i++) {
        if(fabs(W.at<double>(i,i)) > 10^(-10)) {
            rank++;
        }
    }
    return rank;
}

//returns: 0 = no solution, 1 = one solution, -1 = inf solutions
int calcNumberOfSolutions(Mat linEq) {
    Mat coefficients = linEq.colRange(0, linEq.cols-1);
    int coeffRank = calcMatRank(coefficients);
    int augmentedRank = calcMatRank(linEq);
    if (augmentedRank > coeffRank) return 0;
    if (augmentedRank == coeffRank) return 1;
    return -1;
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
//    type+=", depth: ";
//    type+=(DEPTH_MASK & m.type());
    return type;
}

void drawEpipolarLines(std::vector<Point2d> p1, std::vector<Point2d> p2, Mat F, Mat img1, Mat img2, std::string name) {

    if(p1.size() == 0 || p2.size() == 0) return;

    Mat image1 = img1.clone();
    Mat image2 = img2.clone();

    //#################################################################################
    //From: http://opencv-cookbook.googlecode.com/svn/trunk/Chapter%2009/estimateF.cpp
    //#################################################################################

    std::vector<cv::Vec3d> lines1, lines2;
    cv::computeCorrespondEpilines(p1, 1, F, lines1);
    for (std::vector<cv::Vec3d>::const_iterator it= lines1.begin();
         it!=lines1.end(); ++it) {

             cv::line(image2,cv::Point(0,-(*it)[2]/(*it)[1]),
                             cv::Point(image2.cols,-((*it)[2]+(*it)[0]*image2.cols)/(*it)[1]),
                             cv::Scalar(255,255,255));
    }

    cv::computeCorrespondEpilines(p2,2,F,lines2);
    for (std::vector<cv::Vec3d>::const_iterator it= lines2.begin();
         it!=lines2.end(); ++it) {

             cv::line(image1,cv::Point(0,-(*it)[2]/(*it)[1]),
                             cv::Point(image1.cols,-((*it)[2]+(*it)[0]*image1.cols)/(*it)[1]),
                             cv::Scalar(255,255,255));
    }

    // Draw points
    std::vector<cv::Point2d>::const_iterator itPts= p1.begin();
    //std::vector<uchar>::const_iterator itIn= inliers.begin();
    while (itPts!=p1.end()) {

        // draw a circle at each inlier location
        //if (*itIn) {
            cv::circle(image1,*itPts,3,cv::Scalar(255,255,255),2);
            //points1In.push_back(*itPts);
       // }
        ++itPts;
        //++itIn;
    }

    itPts= p2.begin();
    //itIn= inliers.begin();
    while (itPts!=p2.end()) {

        // draw a circle at each inlier location
        //if (*itIn) {
            cv::circle(image2,*itPts,3,cv::Scalar(255,255,255),2);
            //points2In.push_back(*itPts);
        //}
        ++itPts;
        //++itIn;
    }

    // Display the images with points

    showImage(name+" 1",image1);
    showImage(name+" 2",image2);

    //#############################################################################
}

Mat crossProductMatrix(Mat input) {    //3 Vector to cross product matrix
    Mat crossMat = Mat::zeros(3,3, input.type());
    crossMat.at<double>(0,1) = -input.at<double>(2);
    crossMat.at<double>(0,2) = input.at<double>(1);
    crossMat.at<double>(1,0) = input.at<double>(2);
    crossMat.at<double>(1,2) = -input.at<double>(0);
    crossMat.at<double>(2,0) = -input.at<double>(1);
    crossMat.at<double>(2,1) = input.at<double>(0);
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

void PointsToFile(std::vector<Point2d>* points, std::string file) {

    Point2d point;
    std::ofstream outputStream;
    outputStream.open(file.c_str());
    for (int i = 0; points->size(); i++) {
            point = points->at(i);
            outputStream << point.x;
            outputStream << ' ';
            outputStream << point.y;
            outputStream << '\n';
    }
    outputStream.flush();
    outputStream.close();
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

double meanSquaredSymmeticTransferError(Mat F, std::vector<Point2d> points1, std::vector<Point2d> points2) {    //Reprojection error, epipolar line
    double error = 0;
    for(int i = 0; i < points1.size(); i++) {
        error+=std::pow(symmeticTransferError(F, matVector(points1.at(i)), matVector(points2.at(i))),2);
    }
    return error/points1.size();
}

double randomSampleSymmeticTransferError(Mat F1, Mat F2, Mat image1, Mat image2, int numOfSamples) {   //Computes an error mesure between epipolar lines using arbitrary points, see Determining the Epipolar Geometry and its Uncertainty, p24
    //std::srand(std::time(0));
    std::srand(1);  //Pseudo random: Try to use the same points for every image
    double err1 = randomSampleSymmeticTransferErrorSub(F1, F2, image1, image2, numOfSamples);
    if(err1 == -1) return -1;
    double err2 = randomSampleSymmeticTransferErrorSub(F2, F1, image1, image2, numOfSamples);
    if(err2 == -1) return -1;
    return (err1 + err2)/2.0;
}

double randomSampleSymmeticTransferErrorSub(Mat F1, Mat F2, Mat image1, Mat image2, int numOfSamples) {    //Computes an error mesure between epipolar lines using arbitrary points, see Determining the Epipolar Geometry and its Uncertainty, p24
    double epipolarDistSum = 0;
    int imgWidth = image1.cols;
    int imgHeight = image1.rows;
    for(int i = 0; i < numOfSamples; i++) {
        //line ax + by + c = 0 <-> ax + c = -by <-> (-a/b)x + (-c/b) = y; (-a/b) = -l1/l2, (-c/b) = -l3/l2
        Mat p1homog;
        int xMin;
        int xMax;
        double l2F1a = 0, l2F1b = 0;

        Mat l2F1homog;
       //Mat img1;

        int trys = 1;
        do {    //Draw random point until it's epipolar line intersects image 2
            int x = rand()%(imgWidth-20)+10;
            int y = rand()%(imgHeight-20)+10;
            p1homog = matVector(x, y, 1);
            l2F1homog = F1*p1homog;
            l2F1a = -l2F1homog.at<double>(0,0) / l2F1homog.at<double>(1,0);
            l2F1b = -l2F1homog.at<double>(2,0) / l2F1homog.at<double>(1,0);
            if(l2F1a > 0) {
                xMax = std::min(imgWidth, (int)std::floor((imgHeight-l2F1b)/l2F1a));
                xMin = std::max(0, (int)std::ceil(-l2F1b/l2F1a));
            } else if(l2F1a < 0) {
                xMax = std::min(imgWidth, (int)std::floor(-l2F1b/l2F1a));
                xMin = std::max(0, (int)std::ceil((imgHeight-l2F1b)/l2F1a));
            } else {
                xMax = imgWidth;
                xMin = 0;
            }

            trys++;

//            if(trys > 20) {
//            img1 = image1.clone();
//            circle(img1, cvPoint(x,y), 5, Scalar(255,255,255), 3);
//            circle(img1, cvPoint(x,y), 30, Scalar(255,255,255), 6);
//            Mat img2 = image2.clone();
//            cv::line(img2,cv::Point(0,-l2F1homog.at<double>(2,0)/l2F1homog.at<double>(1,0)), cv::Point(image1.cols,-(l2F1homog.at<double>(2,0)+l2F1homog.at<double>(0,0)*image1.cols)/l2F1homog.at<double>(1,0)),cv::Scalar(255,255,255), 2);
//            Mat img;
//            hconcat(img1, img2, img);
//            showImage("test 1, 2", img);
//            waitKey(0);}

        } while((xMin > xMax) && (trys < MAX_SAMPLE_TRYS));

        if(trys == MAX_SAMPLE_TRYS) return -1;      //Cant find a point that projects to an epipolar line in image 2

        double x, y;
        if(xMax == xMin) x = xMax;
        else x = (rand()%(xMax-xMin)) + xMin;
        y = l2F1a*x + l2F1b;

        Mat img2 = image2.clone();
        circle(img2, cvPoint(x,y), 5, Scalar(255,255,255), 3);
        circle(img2, cvPoint(x,y), 30, Scalar(255,255,255), 6);

 //       cv::line(img2,cv::Point(0,-l2F1homog.at<double>(2,0)/l2F1homog.at<double>(1,0)), cv::Point(image1.cols,-(l2F1homog.at<double>(2,0)+l2F1homog.at<double>(0,0)*image1.cols)/l2F1homog.at<double>(1,0)),cv::Scalar(255,0,0), 2);

        //Compute distance of chosen random point to epipolar line of F2
        Mat p2homog = matVector(x, y, 1);
        Mat l2F2homog = F2*p1homog;
        l2F2homog /= l2F2homog.at<double>(1,0);
        epipolarDistSum+=fabs(Mat(p2homog.t()*l2F2homog).at<double>(0,0));

//        cv::line(img2,cv::Point(0,-l2F2homog.at<double>(2,0)/l2F2homog.at<double>(1,0)), cv::Point(image1.cols,-(l2F2homog.at<double>(2,0)+l2F2homog.at<double>(0,0)*image1.cols)/l2F2homog.at<double>(1,0)),cv::Scalar(255,255,255), 2);

        //Compute distance of point1 to epipolar line from random point using F2^T in image 1
        Mat l1F2homog = F2.t()*p2homog;
        l1F2homog /= l1F2homog.at<double>(1,0);
        epipolarDistSum+=fabs(Mat(p1homog.t()*l1F2homog).at<double>(0,0));

//        cv::line(img1,cv::Point(0,-l1F2homog.at<double>(2,0)/l1F2homog.at<double>(1,0)), cv::Point(image1.cols,-(l1F2homog.at<double>(2,0)+l1F2homog.at<double>(0,0)*image1.cols)/l1F2homog.at<double>(1,0)),cv::Scalar(255,255,255), 2);

//        showImage("test1_", img1);
//        showImage("test2", img2);
//        waitKey(0);

    }
    return epipolarDistSum/(2.0*numOfSamples);
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

    lc->id = lcCopy.id;

    return *lc;
}

lineCorrespStruct getlineCorrespStruct(cv::line_descriptor::KeyLine l1, cv::line_descriptor::KeyLine l2, int id) {
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

void visualizeMatches(Mat image_1_color, Mat image_2_color, std::vector<lineCorrespStruct> correspondencies, int lineWidth, bool drawConnections, std::string name) {
    Mat img;
    hconcat(image_1_color.clone(), image_2_color.clone(), img);
    for(std::vector<lineCorrespStruct>::iterator it = correspondencies.begin() ; it != correspondencies.end(); ++it) {
        Scalar color = Scalar(rand()%255, rand()%255, rand()%255);
        cv::line(img, cvPoint2D32f(it->line1Start.at<double>(0,0), it->line1Start.at<double>(1,0)), cvPoint2D32f(it->line1End.at<double>(0,0), it->line1End.at<double>(1,0)), color, lineWidth);
        cv::line(img, cvPoint2D32f(it->line2Start.at<double>(0,0) + image_1_color.cols, it->line2Start.at<double>(1,0)), cvPoint2D32f(it->line2End.at<double>(0,0) + image_1_color.cols, it->line2End.at<double>(1,0)), color, lineWidth);
        if(drawConnections) {
            cv::line(img, cvPoint2D32f(it->line1Start.at<double>(0,0), it->line1Start.at<double>(1,0)), cvPoint2D32f(it->line2Start.at<double>(0,0) + image_1_color.cols, it->line2Start.at<double>(1,0)), color, lineWidth);
        }
    }
    showImage(name, img, WINDOW_NORMAL, 1600);
}

void visualizeMatches(Mat image_1_color, Mat image_2_color, std::vector<Point2d> p1, std::vector<Point2d> p2, int lineWidth, bool drawConnections, std::string name) {
    Mat img;
    hconcat(image_1_color.clone(), image_2_color.clone(), img);
    for(int i = 0; i < p1.size(); i++) {
        Scalar color = Scalar(rand()%255, rand()%255, rand()%255);
        cv::circle(img, p1.at(i), 2, color, lineWidth);
        cv::circle(img, cvPoint2D32f(p2.at(i).x + image_1_color.cols, p2.at(i).y), 2, color, lineWidth);
        //cv::line(img, p1.at(i), p1.at(i), color, lineWidth);
        //cv::line(img, cvPoint2D32f(p2.at(i).x + image_1_color.cols, p2.at(i).y), cvPoint2D32f(p2.at(i).x + image_1_color.cols, p2.at(i).y), color, lineWidth);
        if(drawConnections) {
            cv::line(img, p1.at(i), cvPoint2D32f(p2.at(i).x + image_1_color.cols, p2.at(i).y), color, lineWidth);
        }
    }
    showImage(name, img, WINDOW_NORMAL, 1600);
}

bool isUnity(Mat m) {       //Check main diagonal for being close to 1
    Mat diff = abs(m - Mat::eye(m.rows, m.cols, CV_64FC1));
    for(int i = 0; i < m.cols; i++) {
        if(diff.at<double>(i,i) > MARGIN) return false;
    }
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

    bool allEigenvaluesEqual = true;
    double dist[eigenvalues.rows];
    double lastDist = 0;
    int col = 0;
    for(int i = 0; i < eigenvalues.rows; i ++) {        //find non-unary eigenvalue & its eigenvector
        Mat eig = eigenvalues.row(i);
        dist[i] = 0;
        for(int j = 0; j < eigenvalues.rows; j ++) {
            dist[i] += squaredError(eig, eigenvalues.row(j));
        }
        if(dist[i] > MARGIN) allEigenvaluesEqual = false;
        if(dist[i] > lastDist) {
            col = i;
            lastDist = dist[i];
        }
    }

    std::vector<Mat> e2;
    split(eigenvectors.col(col),e2); //Remove channel for imaginary part
    if(LOG_DEBUG) std::cout << "-- e = " << std::endl << e2.at(0) << std::endl;

    Mat(e2.at(0)).copyTo(e);

    return allEigenvaluesEqual;
}

double symmeticTransferError(Mat F, Mat x1, Mat x2) {
    return Mat(x2.t()*F*x1 + x1.t()*F.t()*x2).at<double>(0,0);
}

std::vector<double> computeCombinedErrorVect(std::vector<FEstimationMethod> estimations, Mat F) {

    std::vector<double> *errorVect = new std::vector<double>();

    for(std::vector<FEstimationMethod>::iterator estimationIter = estimations.begin(); estimationIter != estimations.end(); ++estimationIter) {

        if(estimationIter->getType() == F_FROM_LINES_VIA_H) {   //Line correspondencies != point correspondencies
            Mat H = computeGeneralHomography(F);
            Mat H_invT = H.inv(DECOMP_SVD).t();
            Mat H_T = H.t();
            for(unsigned int i = 0; i < estimationIter->getFeaturesImg1().size()/2; i++)
            {
                double err1, err2;
                err1 = transferLineError(H_T, estimationIter->getFeaturesImg1().at(2*i), estimationIter->getFeaturesImg1().at(2*i+1), estimationIter->getFeaturesImg2().at(2*i), estimationIter->getFeaturesImg2().at(2*i+1));
                err2 = transferLineError(H_invT, estimationIter->getFeaturesImg2().at(2*i), estimationIter->getFeaturesImg2().at(2*i+1), estimationIter->getFeaturesImg1().at(2*i), estimationIter->getFeaturesImg1().at(2*i+1));
                errorVect->push_back((err1 + err2)/2.0);
            }
        } else {
            for(unsigned int i = 0; i < estimationIter->getFeaturesImg1().size(); i++)   //Distance form features to correspondig epipolarline in other image
            {
                Mat x1 = estimationIter->getFeaturesImg1().at(i);
                Mat x2 = estimationIter->getFeaturesImg2().at(i);
                //if(estimationIter->getType() != F_FROM_LINES_VIA_H || (estimationIter->getType() == F_FROM_LINES_VIA_H && symmeticTransferError(F, x1, x2) < MAX_TRANSFER_DIST)) {      //Remove line correspondencies where line tips are no point correspondencies
                    errorVect->push_back(symmeticTransferError(F, x1, x2));
                //}
            }
        }
    }
    return *errorVect;
}

std::vector<double> computeCombinedErrorVect(std::vector<Mat> x1, std::vector<Mat> x2, Mat F) {

    std::vector<double> *errorVect = new std::vector<double>();

    for(int i = 0; i < x1.size(); i++) {
        Mat p1 = x1.at(i);
        Mat p2 = x2.at(i);
        errorVect->push_back(symmeticTransferError(F, p1, p2));
    }
    return *errorVect;
}

double computeCombinedMeanSquaredError(std::vector<FEstimationMethod> estimations, Mat impF) {
    std::vector<double> errorVect = computeCombinedErrorVect(estimations, impF);
    double combinedError = 0;
    for(std::vector<double>::const_iterator errorIter = errorVect.begin(); errorIter != errorVect.end(); ++errorIter) {
        combinedError += std::pow(*errorIter,2);
    }
    return combinedError/errorVect.size();
}

double computeCombinedMeanSquaredError(std::vector<Mat> x1, std::vector<Mat> x2, Mat impF) {
    std::vector<double> errorVect = computeCombinedErrorVect(x1, x2, impF);
    double combinedError = 0;
    for(std::vector<double>::const_iterator errorIter = errorVect.begin(); errorIter != errorVect.end(); ++errorIter) {
        combinedError += std::pow(*errorIter,2);
    }
    return combinedError/errorVect.size();
}

void findGoodCombinedMatches(std::vector<FEstimationMethod> estimations, std::vector<Mat> &x1, std::vector<Mat> &x2, Mat F, double maxDist) {
    for(std::vector<FEstimationMethod>::iterator estimationIter = estimations.begin(); estimationIter != estimations.end(); ++estimationIter) {
        for(int i = 0; i < estimationIter->getFeaturesImg1().size(); i++) {
            if(symmeticTransferError(F, estimationIter->getFeaturesImg1().at(i), estimationIter->getFeaturesImg2().at(i)) < maxDist) {
                x1.push_back(estimationIter->getFeaturesImg1().at(i));
                x2.push_back(estimationIter->getFeaturesImg2().at(i));
            }
        }
    }
}

void computeEpipoles(Mat F, Mat &e1, Mat &e2) {     //See Hartley, Ziss p.246

//    SVD svd;
//    Mat u, vt, w;
//    svd.compute(F, w, u, vt);

//    e1 = Mat(vt.row(2).t());
//    svd.compute(F.t(), w, u, vt);

//    e2 = Mat(vt.row(2).t());

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

Mat computeGeneralHomography(Mat F) {       //See Hartley, Ziss p.243
    Mat e1, e2;

    computeEpipoles(F, e1, e2);
    Mat H = crossProductMatrix(e2).inv(DECOMP_SVD)*F;
    H /= H.at<double>(2,2);
    //if(LOG_DEBUG) std::cout << "-- computed H = " << std::endl << H << std::endl;

    return H;
}

//void symmeticTransferLineError(Mat H_invT, Mat H_T, Mat line1Start, Mat line1End, Mat line2Start, Mat line2End, double *err1, double *err2) {
////    Mat A = H_T*crossProductMatrix(line2Start)*line2End;
////    Mat start1 = line1Start.t()*A;
////    Mat end1 = line1End.t()*A;
////    Mat B = H_invT*crossProductMatrix(line1Start)*line1End;
////    Mat start2 = line2Start.t()*B;
////    Mat end2 = line2End.t()*B;

////    *err1 = Mat((start1+end1)/).at<double>(0,0);
////    *err2 = Mat(start2+end2).at<double>(0,0);

//    *err1 = squaredTransferLineError(H_T, line1Start, line1End, line2Start, line2End);
//    *err2 = squaredTransferLineError(H_invT, line2Start, line2End, line1Start, line1End);
//}

double transferLineError(Mat H, Mat line1Start, Mat line1End, Mat line2Start, Mat line2End) {
    return sqrt(squaredTransferLineError(H, line1Start, line1End, line2Start, line2End));
}

double squaredTransferLineError(Mat H, Mat line1Start, Mat line1End, Mat line2Start, Mat line2End) {
    Mat A = H*crossProductMatrix(line2Start)*line2End;
    Mat start1 = line1Start.t()*A;
    Mat end1 = line1End.t()*A;
    double Ax = std::pow(A.at<double>(0,0), 2);
    double Ay = std::pow(A.at<double>(1,0), 2);
    Mat result = (start1*start1 + end1*end1)/(Ax + Ay);
    return result.at<double>(0,0);
}

double squaredTransferPointError(Mat H, Mat p1, Mat p2) {
    return std::pow(norm(p1*H - p2));
}
