#include "Utility.h"
#include "Statics.h"

#include <ctime>

void showImage(std::string name, Mat image, int type, int width, int height) {
    float tx = 0;
    float ty = 0;
    Mat resized;
    if (width > 0) tx= (float)width/image.cols; {
        if (height > 0) ty= (float)height/image.rows;
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
        if(fabs(W.at<float>(i,i)) > 10^(-10)) {
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

void drawEpipolarLines(std::vector<Point2f> p1, std::vector<Point2f> p2, Mat F, Mat img1, Mat img2, std::string name) {

    Mat image1 = img1.clone();
    Mat image2 = img2.clone();

    //#################################################################################
    //From: http://opencv-cookbook.googlecode.com/svn/trunk/Chapter%2009/estimateF.cpp
    //#################################################################################

    std::vector<cv::Vec3f> lines1, lines2;
    cv::computeCorrespondEpilines(p1, 1, F, lines1);
    for (std::vector<cv::Vec3f>::const_iterator it= lines1.begin();
         it!=lines1.end(); ++it) {

             cv::line(image2,cv::Point(0,-(*it)[2]/(*it)[1]),
                             cv::Point(image2.cols,-((*it)[2]+(*it)[0]*image2.cols)/(*it)[1]),
                             cv::Scalar(255,255,255));
    }

    cv::computeCorrespondEpilines(p2,2,F,lines2);
    for (std::vector<cv::Vec3f>::const_iterator it= lines2.begin();
         it!=lines2.end(); ++it) {

             cv::line(image1,cv::Point(0,-(*it)[2]/(*it)[1]),
                             cv::Point(image1.cols,-((*it)[2]+(*it)[0]*image1.cols)/(*it)[1]),
                             cv::Scalar(255,255,255));
    }

    // Draw points
    std::vector<cv::Point2f>::const_iterator itPts= p1.begin();
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

Mat crossProductMatrix(Mat input) {    //3 Vector to cross procut matrix
    Mat crossMat = Mat::zeros(3,3, input.type());
    crossMat.at<float>(0,1) = -input.at<float>(2);
    crossMat.at<float>(0,2) = input.at<float>(1);
    crossMat.at<float>(1,0) = input.at<float>(2);
    crossMat.at<float>(1,2) = -input.at<float>(0);
    crossMat.at<float>(2,0) = -input.at<float>(1);
    crossMat.at<float>(2,1) = input.at<float>(0);
    return crossMat;
}

void rectify(std::vector<Point2f> p1, std::vector<Point2f> p2, Mat F, Mat image1, Mat image2, std::string windowName) {
    Mat H1, H2, rectified1, rectified2;
    if(stereoRectifyUncalibrated(p1, p2, F, Size(image1.cols,image1.rows), H1, H2, 2 )) {

        warpPerspective(image1, rectified1, H1, Size(image1.cols,image1.rows));
        warpPerspective(image2, rectified2, H2, Size(image1.cols,image1.rows));

        showImage(windowName+" 1", rectified1);
        showImage(windowName+" 2", rectified2);
    }
}

void PointsToFile(std::vector<Point2f>* points, std::string file) {

    Point2f point;
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

    Mat matrix = Mat::zeros(0,0,CV_32FC1);
    std::ifstream inputStream;
    float x;
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
    float x;
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

double epipolarSADError(Mat F, std::vector<Point2f> points1, std::vector<Point2f> points2) {    //Reprojection error, epipolar line
    std::vector<cv::Vec3f> lines1;
    std::vector<cv::Vec3f> lines2;
    double epipolarError = 0;
    cv::computeCorrespondEpilines(points1, 1, F, lines2);
    cv::computeCorrespondEpilines(points2, 2, F, lines1);
    for(int i = 0; i < points1.size(); i++) {
        epipolarError+=abs(matVector(points1.at(i)).dot(lines2.at(i))) + abs(matVector(points2.at(i)).dot(lines1.at(i)));
    }
    return epipolarError;
}

double epipolarLineDistanceError(Mat F1, Mat F2, Mat image, int numOfSamples) {   //Computes an error mesure between epipolar lines using arbitrary points, see Determining the Epipolar Geometry and its Uncertainty, p185
    //std::srand(std::time(0));
    std::srand(1);  //Try to use the same points for every image
    double err1 = epipolarLineDistanceErrorSub(F1, F2, image, numOfSamples);
    if(err1 == -1) return -1;
    double err2 = epipolarLineDistanceErrorSub(F2, F1, image, numOfSamples);
    if(err2 == -1) return -1;
    return err1 + err2;
}

double epipolarLineDistanceErrorSub(Mat F1, Mat F2, Mat image, int numOfSamples) {    //Computes an error mesure between epipolar lines using arbitrary points, see Determining the Epipolar Geometry and its Uncertainty, p185
    double epipolarDistSum = 0;
    for(int i = 0; i < numOfSamples; i++) {
        //line: y = ax + b; a = x1/x3, b = x2/x3

        Mat p1homog;
        int xBoundsMin = 0;
        int xBounds = 0;
        float l2F1a = 0, l2F1b = 0;

        int trys = 1;
        do {    //Draw random point until it's epipolar line intersects image 2
            p1homog = matVector(rand()%image.cols, rand()%image.rows, 1);
            Mat l2F1homog = F1*p1homog;
            l2F1a = l2F1homog.at<float>(0,0) / l2F1homog.at<float>(2,0);
            l2F1b = l2F1homog.at<float>(1,0) / l2F1homog.at<float>(2,0);
            l2F1a/=(-l2F1b);
            l2F1b=1.0/(-l2F1b);
            xBoundsMin = std::min(std::max((int)ceil(l2F1b/l2F1a),0), image.cols);
            xBounds = std::min((int)floor(image.rows*l2F1b/l2F1a), image.cols) - xBoundsMin;
            trys++;
        } while((xBounds < 0 || xBoundsMin > image.cols) && (trys < MAX_SAMPLE_TRYS));

        if(trys == MAX_SAMPLE_TRYS) return -1;

        float x, y;
        if(xBounds == 0) x = xBoundsMin;
        else x = rand()%xBounds + xBoundsMin;
        y = l2F1a*x + l2F1b;

        //Compute distance of chosen random point to epipolar line of F2
        Mat p2homog = matVector(x, y, 1);
        Mat l2F2homog = F2*p1homog;
        epipolarDistSum+=fabs(p2homog.dot(l2F2homog));

        //Compute distance of point1 to epipolar line from random point using F2^T in image 1
        Mat l1F2homog = F2.t()*p2homog;
        epipolarDistSum+=fabs(p1homog.dot(l1F2homog));

    }
    return epipolarDistSum/(2*numOfSamples);
}

Mat matVector(float x, float y, float z) {
    Mat vect = Mat::zeros(3,1,CV_32FC1);
    vect.at<float>(0,0) = x;
    vect.at<float>(1,0) = y;
    vect.at<float>(2,0) = z;
    return vect;
}

Mat matVector(Point2f p) {
    Mat vect = Mat::zeros(3,1,CV_32FC1);
    vect.at<float>(0,0) = p.x;
    vect.at<float>(1,0) = p.y;
    vect.at<float>(2,0) = 1;
    return vect;
}
