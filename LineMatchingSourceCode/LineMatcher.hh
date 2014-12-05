#include <Base/Image/ImageIO.hh>
#include <Base/Image/ImageConvert.hh>
#include <Base/Math/Vector2.hh>
#include <Base/Math/Vector3.hh>
#include <Base/Math/Matrix3x3.hh>
#include <Base/Debug/TimeMeasure.hh>
#include <Utils/Param.hh>
#include <bias_config.h>
#ifndef BIAS_HAVE_OPENCV
#  error You need to enable OPENCV to compile this file. Please reconfigure MIP with USE_OPENCV (jw)
#endif
#include <Base/Common/W32Compat.hh>
#include <math.h>
#include <time.h>
#include <fstream>
#include <cv.h>
#include <highgui.h>


#include "LineDescriptor.hh"
#include "PairwiseLineMatching.hh"

class LineMatcher
{
public:
    	LineMatcher();

	int match(const char* image1,const char* image2, std::vector<cv::Point2f>* matches);
	std::vector<cv::Point2f>* getCorrespondencies();

private:
	std::vector<cv::Point2f>* m_matches;

};
