#include "CannyChain.h"
#include <math.h>
using namespace std;

#define MIN_FLOAT 0.000101

CannyChain::CannyChain()
{

}

CannyChain::~CannyChain()
{

}

void CannyChain::run( cv::Mat &image, double thGradient, std::vector<std::vector<cv::Point> > &CannyChains )
{
	int cols = image.cols;
	int rows = image.rows;
	cols_1 = cols - 1;
	rows_1 = rows - 1;
	int imgSize = rows * cols;

	cv::Mat imgNew;
	if ( image.channels() == 3 )
	{
		cv::cvtColor( image, imgNew, CV_RGB2GRAY );
	}
	else
	{
		imgNew = image;
	}

	// canny
	cv::Mat edgeMap;
	cv::Canny( imgNew, edgeMap, thGradient, 2.0*thGradient, 3, false );

	// get image information
	gradientMap = cv::Mat::zeros( rows, cols, CV_64FC1 );

	cv::Mat dx(rows, cols, CV_16S, Scalar(0));
	cv::Mat dy(rows, cols, CV_16S, Scalar(0));

	cv::Sobel( imgNew, dx, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
	cv::Sobel( imgNew, dy, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);

	double binStep = 20.0;
	int binSize = 2500.0 / binStep + 10;
	std::vector<std::vector<cv::Point> > edgePixelBins( binSize );

	double *ptrG = (double*) gradientMap.data;
	short *ptrX  = (short*) dx.data;
	short *ptrY  = (short*) dy.data;
	uchar *ptrE = edgeMap.data;

	for ( int y = 0; y< rows; ++y )
	{
		for ( int x=0; x<cols; ++x )
		{
			if ( * ptrE )
			{
				double gx = *ptrX;
				double gy = *ptrY;

				*ptrG = ( abs(gx) + abs(gy) ) / 4.0;

				int binIdx = *ptrG / binStep;
				edgePixelBins[binIdx].push_back( cv::Point( x, y ) );
			}

			ptrG++;  ptrX++;  ptrY++;  ptrE++;
		}
	}

	//find strings
	int xTemp[8] = { 0, 1, 0, -1, 1, -1, -1, 1 };
	offsetX = std::vector<int>( xTemp, xTemp + 8 );

	int yTemp[8] = { 1, 0, -1, 0, 1, 1, -1, -1 };
	offsetY = std::vector<int>( yTemp, yTemp + 8 );

	offsetTotal.resize( 8 );
	for ( int i=0; i<8; ++i )
	{
		offsetTotal[i] = offsetY[i] * cols + offsetX[i];
	}

	int thMeaningfulLength = int( 2.0 * log( (double) rows * cols ) / log(8.0) + 0.5 );	

	for ( int i = binSize-1; i>=0; --i )
	{
		for ( int j=0; j<edgePixelBins[i].size(); ++j )
		{
			std::vector<cv::Point> chain;

			int x = edgePixelBins[i][j].x;
			int y = edgePixelBins[i][j].y;
			int loc = y * cols + x;
			double totalGradient = 0.0;

			uchar *ptrECur = edgeMap.data + loc;
			if ( *ptrECur == 0 )
			{
				continue;
			}

			double *ptrGCur = (double*) gradientMap.data + loc;
			do
			{
				chain.push_back( cv::Point( x, y ) );
				*ptrECur = 0;
				totalGradient += *ptrGCur;
			} while ( next( x, y, &ptrECur, &ptrGCur ) );

			cv::Point temp;
			for ( int m = 0, n = chain.size() - 1; m<n; ++m, --n )
			{
				temp = chain[m];
				chain[m] = chain[n];
				chain[n] = temp;
			}

			// Find and add feature pixels to the begin of the string.
			x = edgePixelBins[i][j].x;
			y = edgePixelBins[i][j].y;
			ptrECur = edgeMap.data + loc;
			ptrGCur = (double*) gradientMap.data + loc;
			if ( next( x, y, &ptrECur, &ptrGCur ) )
			{
				do
				{
					chain.push_back( cv::Point( x, y ) );
					*ptrECur = 0;
					totalGradient += *ptrGCur;
				} while ( next( x, y, &ptrECur, &ptrGCur ) );
			}

			if ( chain.size() > 2.0 * thMeaningfulLength && totalGradient > thMeaningfulLength * thGradient )
			{
				CannyChains.push_back( chain );
			}

// 			if ( chain.size() > thMeaningfulLength )
// 			{
// 				CannyChains.push_back( chain );
// 			}
		}	
	}
}

bool CannyChain::next( int &xSeed, int &ySeed, uchar **ptrE, double **ptrGCur )
{
	if ( xSeed < 1 || xSeed >= cols_1 || ySeed < 1 || ySeed >= rows_1 )
	{
		return false;
	}

	for (int i = 0; i != 8; ++i)
	{
		if ( *( *ptrE + offsetTotal[i] ) )
		{
			xSeed += offsetX[i];
			ySeed += offsetY[i];
			*ptrE += offsetTotal[i];
			*ptrGCur += offsetTotal[i];

			return true;
		}
	}
	return false;
}
