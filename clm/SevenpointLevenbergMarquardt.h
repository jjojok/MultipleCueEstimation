#ifndef SEVENPOINTLEVENBERGMARQUARDT_H
#define SEVENPOINTLEVENBERGMARQUARDT_H

#include <vector>

void SevenpointLevenbergMarquardtExit();
void SevenpointLevenbergMarquardtInit();
bool SevenpointLevenbergMarquardt(std::vector<double>  *F, std::vector<double> x1, std::vector<double> y1, std::vector<double> x2, std::vector<double> y2, int F0, int maxIter, double stopDist);

#endif // SEVENPOINTLEVENBERGMARQUARDT_H
