#include "FEstimationMethod.h"

FEstimationMethod::FEstimationMethod(Mat img1, Mat img2) {
	epipolarError = -1;
    image1 = img1.clone();
    image2 = img2.clone();
}

double FEstimationMethod::getEpipolarError(std::vector<Point2f> points1, std::vector<Point2f> points2) {
	if(epipolarError == -1) return epipolarError;
	if(points1.size() == 0 || points1.size() != points2.size()) return -1;
	
}
