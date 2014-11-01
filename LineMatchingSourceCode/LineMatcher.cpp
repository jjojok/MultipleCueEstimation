/*
 * TestLineDescriptor.cpp
 *
 *  Created on: Feb 23, 2012
 *      Author: lz
 */
#include "LineMatcher.hh"

using namespace BIAS;
using namespace std;

LineMatcher::LineMatcher()
{
	
}

std::vector<CvPoint>* LineMatcher::getCorrespondencies() {
	return m_matches;
}

int LineMatcher::match(const char* image1,const char* image2, std::vector<CvPoint>* matches)
{
	cout<<"Entered LineMatcher::match" <<endl;
	m_matches = matches;
  //load first image from file
	std::string imageName1(image1);
	BIAS::Image<unsigned char> imageUC;
	if(BIAS::ImageIO::Load(imageName1,imageUC) != 0){
		cout<<"Cann't read the input image: "<< image1 <<endl;
		return -1;
	}
  //convert the input image into gray image
	BIAS::Image<unsigned char> leftImage;
	if(imageUC.GetChannelCount()!=1||imageUC.GetColorModel()!=ImageBase::CM_Grey){
		BIAS::ImageConvert::ToGrey(imageUC,leftImage);
	}else{
		leftImage = imageUC;
	}

  //load second image from file
	std::string imageName2(image2);
	if(BIAS::ImageIO::Load(imageName2,imageUC) != 0){
		cout<<"Cann't read the input image: "<< image2 <<endl;
		return -1;
	}
  //convert the input image into gray image
	BIAS::Image<unsigned char> rightImage;
	if(imageUC.GetChannelCount()!=1||imageUC.GetColorModel()!=ImageBase::CM_Grey){
		BIAS::ImageConvert::ToGrey(imageUC,rightImage);
	}else{
		rightImage = imageUC;
	}

	unsigned int imageWidth  = leftImage.GetWidth();
	unsigned int imageHeight = leftImage.GetHeight();

	cout<<"Loaded two images (" << imageWidth <<"x" << imageHeight << ") and (" << rightImage.GetWidth() << "x" << rightImage.GetHeight() << ")" <<endl;

	if((0==rightImage.GetWidth())||(0==rightImage.GetHeight())||(imageWidth==0)||(imageHeight==0))
	{
		cout<<"One of the image is empty!"<<endl;
		return -1;
	}

	cout<<"Images OK!" <<endl;

	srand((unsigned)time(0));
	int lowest=100, highest=255;
	int range=(highest-lowest)+1;
	unsigned int r, g, b; //the color of lines

	cout<<"Rand" <<endl;

	//initial variables
	IplImage      *cvLeftImage = NULL;
	IplImage      *cvRightImage = NULL;
	IplImage      *cvLeftColorImage = NULL;
	IplImage      *cvRightColorImage = NULL;

	std::cout << "BIAS OpenCW Bridge" << std::endl;

	BIAS::ImageConvert::BIAS2ipl(leftImage,cvLeftImage);
	BIAS::ImageConvert::BIAS2ipl(rightImage,cvRightImage);
	CvSize imgSize = cvGetSize(cvLeftImage);
	cvLeftColorImage     = cvCreateImage(imgSize,IPL_DEPTH_8U, 3);
	imgSize = cvGetSize(cvRightImage);
	cvRightColorImage    = cvCreateImage(imgSize,IPL_DEPTH_8U, 3);
	cvCvtColor( cvLeftImage, cvLeftColorImage,  CV_GRAY2RGB );
	cvCvtColor( cvRightImage,cvRightColorImage, CV_GRAY2RGB );

  ///////////####################################################################
  ///////////####################################################################

	//extract lines, compute their descriptors and match lines
	std::cout << "Extract lines, compute their descriptors and match lines" << std::endl;	
	LineDescriptor lineDesc;
	PairwiseLineMatching lineMatch;

	ScaleLines   linesInLeft;
	ScaleLines   linesInRight;
	std::vector<unsigned int> matchResult;

	BIAS::TimeMeasure timer;
	timer.Start();
	
	lineDesc.GetLineDescriptor(leftImage,linesInLeft);
	lineDesc.GetLineDescriptor(rightImage,linesInRight);

    std::cout << "Get line discriptors, left: "<< linesInLeft.size() << "; right: " << linesInRight.size() << std::endl;

    std::cout << "Line matching" << std::endl;

	lineMatch.LineMatching(linesInLeft,linesInRight,matchResult);
	timer.Stop();
	timer.Print();
	timer.Reset();

  ///////////####################################################################
  ///////////####################################################################
	//draw  extracted lines into images
	std::cout << "Draw  extracted lines into images" << std::endl;
	
	CvPoint startPoint;
	CvPoint endPoint;
	CvPoint point;
	CvFont  font;
	cvInitFont(&font,CV_FONT_HERSHEY_DUPLEX,1.5,1.2,0.2,1,8);

	for(unsigned int i=0; i<linesInLeft.size(); i++){
		r = lowest+int(rand()%range);
		g = lowest+int(rand()%range);
		b = lowest+int(rand()%range);
		startPoint = cvPoint(int(linesInLeft[i][0].startPointX),int(linesInLeft[i][0].startPointY));
		endPoint   = cvPoint(int(linesInLeft[i][0].endPointX),  int(linesInLeft[i][0].endPointY));
		cvLine( cvLeftColorImage,startPoint,endPoint,CV_RGB(r,g,b));
//		char buf[10];
//		sprintf( buf,   "%d ",  i);
//		if(i%2){
//			point = cvPoint(round(0.75*startPoint.x+0.25*endPoint.x),round(0.75*startPoint.y+0.25*endPoint.y));
//			cvPutText(cvLeftColorImage,buf,point,&font,CV_RGB(r,g,b));
//		}else{
//			point = cvPoint(round(0.25*startPoint.x+0.75*endPoint.x),round(0.25*startPoint.y+0.75*endPoint.y));
//			cvPutText(cvLeftColorImage,buf,point,&font,CV_RGB(r,g,b));
//		}
	}
	for(unsigned int i=0; i<linesInRight.size(); i++){
		r = lowest+int(rand()%range);
		g = lowest+int(rand()%range);
		b = lowest+int(rand()%range);
		startPoint = cvPoint(int(linesInRight[i][0].startPointX),int(linesInRight[i][0].startPointY));
		endPoint   = cvPoint(int(linesInRight[i][0].endPointX),  int(linesInRight[i][0].endPointY));
		cvLine( cvRightColorImage,startPoint,endPoint,CV_RGB(r,g,b));
//		char buf[10];
//		sprintf( buf,   "%d ",  i);
//		if(i%2){
//			point = cvPoint(round(0.75*startPoint.x+0.25*endPoint.x),round(0.75*startPoint.y+0.25*endPoint.y));
//			cvPutText(cvRightColorImage,buf,point,&font,CV_RGB(r,g,b));
//		}else{
//			point = cvPoint(round(0.25*startPoint.x+0.75*endPoint.x),round(0.25*startPoint.y+0.75*endPoint.y));
//			cvPutText(cvRightColorImage,buf,point,&font,CV_RGB(r,g,b));
//		}
	}
	cvSaveImage("LinesInImage1.png",cvLeftColorImage);
	cvSaveImage("LinesInImage2.png",cvRightColorImage);
  ///////////####################################################################
  ///////////####################################################################
  //store the matching results of the first and second images into a single image
	double ww1,ww2;
	int lineIDLeft;
	int lineIDRight;
	cvCvtColor( cvLeftImage, cvLeftColorImage,  CV_GRAY2RGB );
	cvCvtColor( cvRightImage,cvRightColorImage, CV_GRAY2RGB );
	int lowest1=0, highest1=255;
	int range1=(highest1-lowest1)+1;
	std::vector<unsigned int> r1(matchResult.size()/2), g1(matchResult.size()/2), b1(matchResult.size()/2); //the color of lines
	//matches = new vector<CvPoint>();
	for(unsigned int pair=0; pair<matchResult.size()/2;pair++){
		r1[pair] = lowest1+int(rand()%range1);
		g1[pair] = lowest1+int(rand()%range1);
		b1[pair] = 255 - r1[pair];
		ww1 = 0.2*(rand()%5);
		ww2 = 1 - ww1;
		char buf[10];
		sprintf( buf,   "%d ",  pair);
		lineIDLeft = matchResult[2*pair];
		lineIDRight= matchResult[2*pair+1];
		startPoint = cvPoint(int(linesInLeft[lineIDLeft][0].startPointX),int(linesInLeft[lineIDLeft][0].startPointY));
		endPoint   = cvPoint(int(linesInLeft[lineIDLeft][0].endPointX),  int(linesInLeft[lineIDLeft][0].endPointY));
		matches->push_back(startPoint);
		matches->push_back(endPoint);
		cvLine( cvLeftColorImage,startPoint,endPoint,CV_RGB(r1[pair],g1[pair],b1[pair]),4, CV_AA);
		startPoint = cvPoint(int(linesInRight[lineIDRight][0].startPointX),int(linesInRight[lineIDRight][0].startPointY));
		endPoint   = cvPoint(int(linesInRight[lineIDRight][0].endPointX),  int(linesInRight[lineIDRight][0].endPointY));
		matches->push_back(startPoint);
		matches->push_back(endPoint);
		cvLine( cvRightColorImage,startPoint,endPoint,CV_RGB(r1[pair],g1[pair],b1[pair]),4, CV_AA);
	}

	IplImage *cvResultColorImage1 = cvCreateImage(cvSize(imageWidth*2,imageHeight),IPL_DEPTH_8U, 3);
	IplImage *cvResultColorImage2 = cvCreateImage(cvSize(imageWidth*2,imageHeight),IPL_DEPTH_8U, 3);
	IplImage *cvResultColorImage = cvCreateImage(cvSize(imageWidth*2,imageHeight),IPL_DEPTH_8U, 3);
	cvSetImageROI(cvResultColorImage1, cvRect(0, 0, imageWidth-1, imageHeight-1));
	cvResize(cvLeftColorImage, cvResultColorImage1);
	cvResetImageROI(cvResultColorImage1);
	cvSetImageROI(cvResultColorImage1, cvRect(imageWidth, 0, imageWidth*2-1, imageHeight-1));
	cvResize(cvRightColorImage, cvResultColorImage1);
	cvResetImageROI(cvResultColorImage1);
  	cvCopy(cvResultColorImage1,cvResultColorImage2);
	for(unsigned int pair=0; pair<matchResult.size()/2;pair++){
		lineIDLeft = matchResult[2*pair];
		lineIDRight= matchResult[2*pair+1];
		startPoint = cvPoint(int(linesInLeft[lineIDLeft][0].startPointX),int(linesInLeft[lineIDLeft][0].startPointY));
		endPoint   = cvPoint(int(linesInRight[lineIDRight][0].startPointX+imageWidth),int(linesInRight[lineIDRight][0].startPointY));
		cvLine( cvResultColorImage2,startPoint,endPoint,CV_RGB(r1[pair],g1[pair],b1[pair]),2, CV_AA);
	}
	cvAddWeighted( cvResultColorImage1, 0.5, cvResultColorImage2, 0.5, 0.0, cvResultColorImage);

	cvSaveImage("LBDSG.png",cvResultColorImage);
	cvReleaseImage(&cvResultColorImage);
	cvReleaseImage(&cvResultColorImage1);
	cvReleaseImage(&cvResultColorImage2);
	cvReleaseImage(&cvLeftImage);
	cvReleaseImage(&cvRightImage);
	cvReleaseImage(&cvLeftColorImage);
	cvReleaseImage(&cvRightColorImage);
	cout<<"number of total matches = "<<matchResult.size()/2<<endl;
	return matchResult.size()/2;
}
  ///////////####################################################################
  ///////////####################################################################
