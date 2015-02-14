#include "Utility.h"

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

Scalar squaredError(Mat A, Mat B) {
    return cv::sum((A-B).mul(A-B));
}
