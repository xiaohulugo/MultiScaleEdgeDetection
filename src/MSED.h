#ifndef _MSED_H_
#define _MSED_H_
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

class MSED
{
public:
	MSED(void);
	~MSED(void);

public:

	void MSEdge( cv::Mat &image, int gaussSize, double thLow, double cth, std::vector<std::vector<cv::Point> > &edges );

	void MSNonMaximumSuppression( cv::Mat image, cv::Mat imgResized, std::vector<std::vector<cv::Point> > &edgeChains, double thContrastDev, int scale, cv::Mat &msEmap, 
		cv::Mat &msOmap );

	void getMSEdgeMap( std::vector<std::vector<cv::Point> > &edgeChains, cv::Mat &msEmap, cv::Mat &msOmap, int scale );

	void thinningGuoHall( cv::Mat& img );

	void edgeTracking( cv::Mat &edgeMap, std::vector<std::vector<cv::Point> > &edgeChains );

	void edgeConnection( std::vector<std::vector<cv::Point> > &edgeChains );

private:

	void makeUpImage( cv::Mat& img );

	void imgThinning( cv::Mat &img, int iter );

	bool next( cv::Point &pt, uchar **ptrM, int &dir );

private:

	int idxAll;
	int rows, cols, rows_1, cols_1;
	cv::Mat EmapAll, OmapAll;
	cv::Mat mask;
	std::vector<int> offsetX, offsetY, offsetTotal;
	std::vector<std::vector<int> > idxSearch;
	cv::Mat maskMakeUp;
};

#endif //_MSED_H_
