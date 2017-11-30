#ifndef _EDGE_CHAIN_
#define _EDGE_CHAIN_
#pragma once

#include "opencv/cv.h"
#include "highgui.h"

using namespace std;
using namespace cv;

class CannyChain 
{
public:
	CannyChain();
	~CannyChain();

	void run( cv::Mat &image, double thGradient, std::vector<std::vector<cv::Point> > &CannyChains );

	bool next( int &xSeed, int &ySeed, uchar **ptrE, double **ptrGCur );

private:
	std::vector<int> offsetX, offsetY, offsetTotal;
	int cols_1, rows_1;
	cv::Mat gradientMap;
};

#endif // _PLINKAGE_SUPERPIXEL_
