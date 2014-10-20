#ifndef MCE_H
#define MCE_H

#include <opencv2/opencv.hpp>
#include <string>

#define SIFT_FEATURE_COUNT 800

using namespace cv;

class MCE
{
public:
    MCE(int argc, char** argv);

    void run();
    int loadData();
    void extractSIFT();
    Mat calcFwithPoints();
    std::vector<Point2f>* PointsFromFile(String file);
    void PointsToFile(std::vector<Point2f>* points, String file);

private:

    int arguments;
    char** paths;
    Mat image_1, image_2;
    std::vector<Point2f> x1, x2;   //corresponding points in image 1 and 2
};

#endif // MCE_MAIN_H
