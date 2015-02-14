#include "FEstimationMethod.h"

//FEstimationMethod::FEstimationMethod(Mat img1, Mat img2, Mat img1_c, Mat img2_c, std::string name) {
//    epipolarError = -1;
//    image_1 = img1.clone();
//    image_2 = img2.clone();
//    image_1_color = img1_c.clone();
//    image_2_color = img2_c.clone();
//    this->name = name;
//    std::cout << "Estimating: " << name << std::endl;
//    init();
//}

double FEstimationMethod::getEpipolarError(std::vector<Point2f> points1, std::vector<Point2f> points2) {
    if(epipolarError != -1) return epipolarError;
	if(points1.size() == 0 || points1.size() != points2.size()) return -1;
    std::vector<cv::Vec3f> lines1;
    std::vector<cv::Vec3f> lines2;
    epipolarError = 0;
    cv::computeCorrespondEpilines(points1, 1, F, lines1);
    cv::computeCorrespondEpilines(points2, 2, F, lines2);
    for(int i = 0; i < points1.size(); i++) {
        epipolarError += fabs(points1.at(i).x*lines2.at(i)[0] + points1.at(i).y*lines2.at(i)[1] + lines2.at(i)[2]) + fabs(points2.at(i).x*lines1.at(i)[0] + points2.at(i).y*lines1.at(i)[1] + lines1.at(i)[2]);
    }
    return epipolarError;
}

void FEstimationMethod::init() {

}

Point2f FEstimationMethod::normalize() {
    //TODO
}

Mat FEstimationMethod::denormalize() {
    //TODO
}
