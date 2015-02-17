#include "FEstimationMethod.h"

FEstimationMethod::FEstimationMethod() {

}

FEstimationMethod::~FEstimationMethod() {

}

int FEstimationMethod::extractMatches() {

}

Mat FEstimationMethod::compute() {

}

Mat FEstimationMethod::getF() {
    return F;
}

void FEstimationMethod::init() {

}


Point3f FEstimationMethod::normalize(Point3f p) {
    //return (normT1*normT2)*p;
}

Point3f FEstimationMethod::normalize(float x, float y, float z) {
//   Point3f* p = new Point3f();
//   p->x = (2*x / image_1.) - 1;
//   p->y = (2*y / img_height) - 1;
//   p->z = 1;
//   return *p;

}

Mat FEstimationMethod::denormalize() {
    //return
}
