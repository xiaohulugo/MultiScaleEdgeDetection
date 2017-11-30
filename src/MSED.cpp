#include "MSED.h"
#include "CannyChain.h"

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

#define MIN_FLOAT 0.000101
#define INF 1001001000111

MSED::MSED(void)
{
	idxAll = 1;
}


MSED::~MSED(void)
{
}

void MSED::MSEdge( cv::Mat &image, int gaussSize, double thLow, double cth, std::vector<std::vector<cv::Point> > &edges )
{
	rows = image.rows;
	cols = image.cols;
	rows_1 = rows - 1;
	cols_1 = cols - 1;

	OmapAll = cv::Mat( rows, cols, CV_8UC1, cv::Scalar( 0 ) );
	EmapAll = cv::Mat( rows, cols, CV_64FC1, cv::Scalar( 0.0 ) );

	if ( image.channels() == 3 )
	{
		cv::cvtColor(image, image, CV_BGR2GRAY);
	}
	cv::GaussianBlur( image, image, cv::Size( gaussSize,gaussSize ), -1 );

	int levels = 4;
	for ( int i=0; i<levels; ++i )
	{
		int scale = int( pow( 2.0, i ) );

		// resize image
		double ratio = 1.0 / scale;
		cv::Size size( cols * ratio, rows * ratio );
		cv::Mat imgResized;
		cv::resize( image, imgResized, size, 0, 0 );

		// edge chain detection
		std::vector<std::vector<cv::Point> > edgeChains;
		CannyChain cannyChainer;
		cannyChainer.run( imgResized, thLow, edgeChains );

		// multi scale suppression for edge area
		cv::Mat msEmap, msOmap;
		MSNonMaximumSuppression( image, imgResized, edgeChains, cth, scale, msEmap, msOmap );
		
		// get overlapped edge map
		getMSEdgeMap( edgeChains, msEmap, msOmap, scale );
	}

	// edge map thinning
	cv::Mat imgMask( rows, cols, CV_8UC1, cv::Scalar( 0 ) );	
	double *ptrE = (double*) EmapAll.data;
	uchar *ptrM = imgMask.data;
	int imgSize = rows * cols;
	for ( int i=0; i<imgSize; ++i )
	{
		if ( *ptrE )
		{
			*ptrM = 255;
		}
		
		ptrE++;   ptrM++;
	}
	thinningGuoHall( imgMask );

	edgeTracking( imgMask, edges );
	edgeConnection( edges );
}

void MSED::MSNonMaximumSuppression( cv::Mat image, cv::Mat imgResized, std::vector<std::vector<cv::Point> > &edgeChains, double thContrastDev, int scale, cv::Mat &msEmap, 
	cv::Mat &msOmap )
{
	double tanPi8 = tan( CV_PI / 8.0 );
	double tan3Pi8 = tan( CV_PI * 3.0 / 8.0 );

	double ratio = 1.0 / scale;
	int colsR = imgResized.cols;
	int rowsR = imgResized.rows;
	
	// the edge map with edge chain index
	cv::Mat EMapR( rowsR, colsR, CV_64FC1, cv::Scalar(0) );
	double *ptrER = (double*) EMapR.data;
	for ( int i=0; i<edgeChains.size(); ++i )
	{
		int idxTemp = i + 1;
		for ( int j=0; j<edgeChains[i].size(); ++j )
		{
			int loc = edgeChains[i][j].y * colsR + edgeChains[i][j].x;
			ptrER[loc] = idxTemp;
		}
	}

	// get the gradient orientation of the resized image
	int imgSizeR = rowsR * colsR;

	cv::Mat dxR( rowsR, colsR, CV_16S, cv::Scalar(0) );
	cv::Mat dyR( rowsR, colsR, CV_16S, cv::Scalar(0) );

	cv::Sobel( imgResized, dxR, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE );
	cv::Sobel( imgResized, dyR, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE );

	cv::Mat OmapR4( rowsR, colsR, CV_8UC1, cv::Scalar(0) );
	cv::Mat OmapR2( rowsR, colsR, CV_8UC1, cv::Scalar(0) );

	uchar *ptrOR4 = OmapR4.data;
	uchar *ptrOR2 = OmapR2.data;
	short *ptrXR = (short*) dxR.data;
	short *ptrYR = (short*) dyR.data;

	for ( int j = 0; j< imgSizeR; ++j )
	{
		double gx = *ptrXR;
		double gy = *ptrYR;

		if ( *ptrER )
		{
			double tg = gy / ( gx + MIN_FLOAT );
			if( fabs(tg) < tanPi8 )
			{
				*ptrOR4 = 1; 
			}
			else if ( tanPi8 <= tg && tg <= tan3Pi8 )
			{
				*ptrOR4 = 2; 
			}
			else if ( -tan3Pi8 <= tg && tg <= -tanPi8 )
			{
				*ptrOR4 = 3;
			}
			else
			{
				assert( fabs(tg) > tan3Pi8 );
				*ptrOR4 = 4;
			}

			if ( abs( gx ) > abs( gy ) )
			{
				*ptrOR2 = 0;               // vertical
			}
			else
			{
				*ptrOR2 = 1;               // horizontal
			}

		}

		ptrXR ++;  ptrYR ++;  ptrOR4++;  ptrOR2++;  ptrER++;
	}

	if ( scale == 1 )
	{
		msOmap = OmapR2;
		msEmap = EMapR;
		return;
	}
	dxR.release();
	dyR.release();

	// get the multi scale gradient map
	std::vector<std::vector<int> > offset( 4, 2 * scale + 1 );
	for ( int j=0; j<=2*scale; ++j )
	{
		int p = j - scale;
		offset[0][j] = p;
		offset[1][j] = p * cols + p;
		offset[2][j] = p * cols - p;
		offset[3][j] = p * cols;
	}

	cv::Mat msGmap = cv::Mat( rows, cols, CV_64FC1, cv::Scalar( 0.0 ) );
	double *ptrG = (double*) msGmap.data;

	msEmap = cv::Mat( rows, cols, CV_64FC1, cv::Scalar( 0.0 ) );	
	double *ptrE = (double*) msEmap.data;
	ptrER = (double*) EMapR.data;

	cv::Mat msOmap4Dir( rows, cols, CV_8UC1, cv::Scalar( 0 ) );
	cv::Mat msOmap2Dir( rows, cols, CV_8UC1, cv::Scalar( 0 ) );	
	uchar *ptrO4 = msOmap4Dir.data;
	uchar *ptrO2 = msOmap2Dir.data;

	ptrOR4 = OmapR4.data;
	ptrOR2 = OmapR2.data;	

	std::vector<int> offsetTemp( 4 );
	for ( int x=scale; x<cols-scale; ++x )
	{
		for ( int y=scale; y<rows-scale; ++y )
		{
			int loc = y * cols + x;

			// find the edge pixels on the resized image
			double dxR = x * ratio - int( x * ratio );
			double dyR = y * ratio - int( y * ratio );
			int dirx = 0, diry = 0;
			if ( dxR < 0.3 )
			{
				dirx = -1;
			}
			if ( dxR > 0.7 )
			{
				dirx = 1;
			}

			if ( dyR < 0.3 )
			{
				diry = -1;
			}
			if ( dyR > 0.7 )
			{
				diry = 1;
			}

			int xR = int( x * ratio );
			int yR = int( y * ratio );
			if ( xR < 1 || xR > colsR - 2 || yR < 1 || yR > rowsR - 2 )
			{
				continue;
			}

			offsetTemp[0] = 0;
			offsetTemp[1] = dirx;
			offsetTemp[2] = diry * colsR;
			offsetTemp[3] = diry * colsR + dirx;

			int locR = yR * colsR + xR;
			uchar *ptrTemp = ptrOR4 + locR;

			int idx = -1, offsetIdx = -1;
			for ( int j=0; j<4; ++j )
			{
				int temp = *( ptrTemp + offsetTemp[j] );
				if ( temp )
				{
					idx = temp - 1;
					offsetIdx = offsetTemp[j];
					break;
				}
			}

			// get the image contrast
			if ( idx >= 0 )
			{			
				uchar *ptrI = image.data + loc;
				double I0 = 0.0, I1 = 0.0;
				for ( int j=0; j<scale; ++j )
				{
					I0 += *( ptrI + offset[idx][j] );
				}
				for ( int j=scale+1; j<=2*scale; ++j )
				{
					I1 += *( ptrI + offset[idx][j] );
				}

				
				double g = abs( I1 - I0 ) / scale;
				ptrG[loc] = g;
				ptrO4[loc] = idx;
				ptrO2[loc] = *( ptrOR2 + locR + offsetIdx );
				ptrE[loc] = *( ptrER + locR + offsetIdx );
			}
		}
	}
	msOmap = msOmap2Dir;

	// non-maximum-suppression on the original image
	cv::Mat msEmapTemp( rows, cols, CV_64FC1, cv::Scalar(0.0) );
	double *ptrETemp = (double*) msEmapTemp.data;

	for ( int y=scale; y<rows-scale; ++y )
	{
		for ( int x=scale; x<cols-scale; ++x )
		{
			int loc = y * cols + x;
			double gCur = ptrG[loc];
			if ( ! gCur )
			{
				continue;
			}

			bool isMaximum = true;

			if ( ptrO4[loc] == 0 )
			{
				double *ptrGTemp = ptrG + y * cols + x - scale;
				for ( int j=-scale; j<=scale; ++j )
				{
					double contrastDev = gCur - *ptrGTemp;
					if ( contrastDev < -thContrastDev )
					{
						isMaximum = false;
						break;
					}

					ptrGTemp += 1;
				}
			}
			else if ( ptrO4[loc] == 1 )
			{
				double *ptrGTemp = ptrG + ( y - scale )* cols + x - scale;
				for ( int j=-scale; j<=scale; ++j )
				{
					double contrastDev = gCur - *ptrGTemp;
					if ( contrastDev < -thContrastDev )
					{
						isMaximum = false;
						break;
					}

					ptrGTemp += cols + 1;
				}
			}
			else if ( ptrO4[loc] == 2 )
			{
				double *ptrGTemp = ptrG + ( y - scale )* cols + x + scale;
				for ( int j=-scale; j<=scale; ++j )
				{
					double contrastDev = gCur - *ptrGTemp;
					if ( contrastDev < -thContrastDev )
					{
						isMaximum = false;
						break;
					}

					ptrGTemp += cols - 1;
				}
			}
			else
			{
				assert( ptrO4[loc] == 3 );
				double *ptrGTemp = ptrG + ( y - scale ) * cols + x;
				for ( int j=-scale; j<=scale; ++j )
				{
					double contrastDev = gCur - *ptrGTemp;
					if ( contrastDev < -thContrastDev )
					{
						isMaximum = false;
						break;
					}

					ptrGTemp += cols;
				}
			}

			if ( isMaximum )
			{
				ptrETemp[loc] = ptrE[loc];
			}
		}
	}

	msEmapTemp.copyTo( msEmap );
}

void MSED::getMSEdgeMap( std::vector<std::vector<cv::Point> > &edgeChains, cv::Mat &msEmap, cv::Mat &msOmap, int scale )
{
	int rows_n = rows - scale;
	int cols_n = cols - scale;

	if ( scale == 1 )   // keep all the edge chains in level 0
	{
		double *ptrEAll = (double*)EmapAll.data;

		for ( int i=0; i<edgeChains.size(); ++i )
		{
			for ( int j=0; j<edgeChains[i].size(); ++j )
			{
				int loc = edgeChains[i][j].y * cols + edgeChains[i][j].x;
				ptrEAll[loc] = idxAll;
			}

			idxAll++;
		}

		msOmap.copyTo( OmapAll );
	}
	else
	{
		maskMakeUp = cv::Mat( rows, cols, CV_8UC3, cv::Scalar(255,255,255) );
		uchar *ptrMU = maskMakeUp.data;

		for ( int i=0; i<edgeChains.size(); ++i )
		{
			int idCur = i + 1;

			// get the overlapped index of each point on the edge chain
			std::vector<int> overlappedIdx( edgeChains[i].size(), 0 );

			for ( int j=0; j<edgeChains[i].size(); ++j )
			{
				int x0 = edgeChains[i][j].x * scale;
				int y0 = edgeChains[i][j].y * scale;
				if ( x0 < scale || x0 >= cols_n || y0 < scale || y0 >= rows_n )
				{
					continue;
				}

				int offsetCur = ( y0 - scale ) * cols + x0 - scale;
				double *ptrECur = (double*) msEmap.data + offsetCur;
				double *ptrEAll = (double*) EmapAll.data + offsetCur;
				
				int idLowLevel = 0;
				bool foundTwo = false;
				for ( int y = -scale; y<=scale; ++y )
				{
					for ( int x=-scale; x<=scale; ++x )
					{
						if ( * ptrECur == idCur && *ptrEAll )
						{
							if ( ! idLowLevel )
							{
								idLowLevel = *ptrEAll;
							}
							else if( *ptrEAll != idLowLevel )
							{
								foundTwo = true;
								break;
							}
						}

						ptrECur++;  ptrEAll++;
					}
					if ( foundTwo )
					{
						break;
					}

					ptrECur += cols - 2 * scale - 1;
					ptrEAll += cols - 2 * scale - 1;
				}

				if ( ! idLowLevel || ( idLowLevel && foundTwo ) )
				{
					overlappedIdx[j] = 1;  // cover to the EmapAll
				}
				else
				{
					overlappedIdx[j] = 0;  // suppressed by the low level edges
				}
			}

			// find the covering intervals of the edge chain
			std::vector<int> intervals;

			int length = edgeChains[i].size();
			for ( int m=0; m<length-1; ++m )
			{
				if ( ! overlappedIdx[m] )
				{
					continue;
				}

				int count = 1, n = 0;
				for ( n=m+1; n<length; ++n )
				{
					if ( overlappedIdx[n] )
					{
						count ++;
					}
					else
					{
						break;
					}
				}

				if ( count >= 1 )
				{
					intervals.push_back( max( m - 1, 0 ) );
					intervals.push_back( min( n + 1, length - 1 ) );
					m = n;
				}
			}

			// cover the intervals of the current edge chain onto EmapAll
			for ( int j=0; j<intervals.size()/2; ++j )
			{
				int idxStart = intervals[2*j+0];
				int idxEnd = intervals[2*j+1];
				
				for ( int m=idxStart; m<=idxEnd; ++m )
				{
					int x0 = edgeChains[i][m].x * scale;
					int y0 = edgeChains[i][m].y * scale;
					if ( x0 < scale || x0 >= cols_n || y0 < scale || y0 >= rows_n )
					{
						continue;
					}

					int offsetCur = ( y0 - scale ) * cols + x0 - scale;
					double *ptrECur = (double*) msEmap.data + offsetCur;
					uchar  *ptrOCur = msOmap.data + offsetCur;

					double *ptrEAll = (double*) EmapAll.data + offsetCur;
					uchar  *ptrOAll = OmapAll.data +offsetCur;

					for ( int y = -scale; y<=scale; ++y )
					{
						for ( int x=-scale; x<=scale; ++x )
						{
							if ( * ptrECur == idCur )
							{
								*ptrEAll = idxAll;
								*ptrOAll = *ptrOCur;

								int locTemp = ( y0 + y ) * cols + x0 + x;
								ptrMU[3*locTemp+0] = 0;
								ptrMU[3*locTemp+1] = 255;
								ptrMU[3*locTemp+2] = 0;
							}

							ptrECur++;  ptrOCur++;  
							ptrEAll++;  ptrOAll++;
						}
						ptrECur += cols - 2 * scale - 1;
						ptrOCur += cols - 2 * scale - 1;

						ptrEAll += cols - 2 * scale - 1;
						ptrOAll += cols - 2 * scale - 1;
					}
				}

				idxAll ++;
			}
		}
	}
}

bool MSED::next( cv::Point &pt, uchar **ptrM, int &dir )
{
	if ( pt.x < 1 || pt.x >= cols_1 || pt.y < 1 || pt.y >= rows_1 )
	{
		return false;
	}

	for ( int i=0; i<idxSearch[dir].size(); ++i )
	{
		int dirIdx = idxSearch[dir][i];

		if ( *( *ptrM + offsetTotal[dirIdx] ) )
		{
			for ( int j=0; j<idxSearch[dir].size(); ++j )
			{
				int dirIdxTemp = idxSearch[dir][j];
				if ( *( *ptrM + offsetTotal[dirIdxTemp] ) == 1 )  // find a connect pixel
				{
					return false;
				}
			}

			pt.x += offsetX[dirIdx];
			pt.y += offsetY[dirIdx];
			*ptrM += offsetTotal[dirIdx];

			dir = ( dirIdx + 4 ) % 8;

			return true;
		}
	}
	return false;
}

void MSED::thinningGuoHall( cv::Mat& img )
{
	makeUpImage( img );

	img /= 255;
	cv::Mat prev( img.rows, img.cols, CV_8UC1, cv::Scalar::all( 0 ) );
	cv::Mat diff( img.rows, img.cols, CV_8UC1, cv::Scalar::all( 0 ) );
	int iteration = 0;

	do 
	{
		imgThinning( img, 0 );
		imgThinning( img, 1 );
		cv::absdiff( img, prev, diff );
		img.copyTo( prev ); 

		iteration ++;
	} while ( cv::countNonZero( diff ) > 0 );

	cout<<"iteration: "<<iteration<<endl;
	img *= 255;
}

void MSED::imgThinning( cv::Mat &img, int iter )
{
	//Code for thinning a binary image using Guo-Hall algorithm.
	/**
	* Perform one thinning iteration.
	* Normally you wouldn't call this function directly from your code.
	*
	* @param  im    Binary image with range = 0-1
	* @param  iter  0=even, 1=odd
	*/

	int colsCur = img.cols;
	int rowsCur = img.rows;
	int colsCur_1 = colsCur - 1;
	int rowsCur_1 = rowsCur - 1;

	cv::Mat marker( rowsCur, colsCur, CV_8UC1, cv::Scalar::all( 0 ) );

	for ( int i=1; i<rowsCur_1; i++ )
	{
		for ( int j=1; j<colsCur_1; j++ )
		{
			//int marker = 0;
			uchar *ptr = img.data + i * img.cols + j;
			if ( ! *ptr )
			{
				continue;
			}

			uchar p2 = ptr[-colsCur];
			uchar p3 = ptr[-colsCur + 1];
			uchar p4 = ptr[1];
			uchar p5 = ptr[colsCur + 1];
			uchar p6 = ptr[colsCur];
			uchar p7 = ptr[colsCur - 1];
			uchar p8 = ptr[-1];
			uchar p9 = ptr[-colsCur - 1];

			int C = ( !p2 & ( p3 | p4 ) ) + ( !p4 & ( p5 | p6 ) )
				  + ( !p6 & ( p7 | p8 ) ) + ( !p8 & ( p9 | p2 ) ); 
			int N1 = ( p9 | p2 ) + ( p3 | p4 ) + ( p5 | p6 ) + ( p7 | p8 );
			int N2 = ( p2 | p3 ) + ( p4 | p5 ) + ( p6 | p7 ) + ( p8 | p9 );

			int N  = N1 < N2 ? N1 : N2;
			int m  = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

			if ( C == 1 && ( N >= 2 && N <= 3 ) & m == 0)
				marker.at<uchar>( i, j ) = 1;
		} 
	}

	img &= ~marker;
}


void MSED::makeUpImage( cv::Mat& img )
{
	int xTemp[8] = { -1,  0,  1, 1, 1, 0, -1, -1 };
	offsetX = std::vector<int>( xTemp, xTemp + 8 );

	int yTemp[8] = { -1, -1, -1, 0, 1, 1,  1,  0 };
	offsetY = std::vector<int>( yTemp, yTemp + 8 );

	offsetTotal.resize( 8 );
	for ( int i=0; i<8; ++i )
	{
		offsetTotal[i] = offsetY[i] * cols + offsetX[i];
	}

	for ( int y=1; y<rows_1; ++y )
	{
		for ( int x=1; x<cols_1; ++x )
		{
			uchar *ptr = img.data + y * cols + x;

			if ( ! *ptr )
			{
				int count = 0;
				for ( int m=0; m<8; ++m )
				{
					if ( *( ptr + offsetTotal[m] ) )
					{
						count ++;
					}
				}

				if ( count >= 5 )
				{
					*ptr = 255;
				}
			}
		}
	}
}


void MSED::edgeTracking( cv::Mat &edgeMap, std::vector<std::vector<cv::Point> > &edgeChains )
{
	int xTemp[8] = { -1,  0,  1, 1, 1, 0, -1, -1 };
	offsetX = std::vector<int>( xTemp, xTemp + 8 );

	int yTemp[8] = { -1, -1, -1, 0, 1, 1,  1,  0 };
	offsetY = std::vector<int>( yTemp, yTemp + 8 );

	offsetTotal.resize( 8 );
	for ( int i=0; i<8; ++i )
	{
		offsetTotal[i] = offsetY[i] * cols + offsetX[i];
	}

	idxSearch = std::vector<std::vector<int> >( 8, 5 );
	for ( int i=0; i<8; ++i )
	{
		idxSearch[i][0] = ( i + 4 ) % 8;
		idxSearch[i][1] = ( i + 3 ) % 8;
		idxSearch[i][2] = ( i + 5 ) % 8;
		idxSearch[i][3] = ( i + 2 ) % 8;
		idxSearch[i][4] = ( i + 6 ) % 8;
	}

	//
	edgeMap.copyTo( mask );

	for ( int y=0; y<rows; ++y )
	{
		for ( int x=0; x<cols; ++x )
		{
			int loc = y * cols + x;
			if ( ! mask.data[loc] )
			{
				continue;
			}

			int dir1 = 0, dir2 = 0;
			if ( OmapAll.data[loc] == 0 )   // vertical edge
			{
				dir1 = 1;
				dir2 = 5;
			}
			else
			{
				dir1 = 3;
				dir2 = 7;
			}

			std::vector<cv::Point> chain;

			cv::Point pt( x, y );
			uchar* ptrM = mask.data + loc;
			do
			{
				chain.push_back( pt );
				*ptrM = 0;
			} while ( next( pt, &ptrM, dir1 ) );

			cv::Point temp;
			for ( int m = 0, n = chain.size() - 1; m<n; ++m, --n )
			{
				temp = chain[m];
				chain[m] = chain[n];
				chain[n] = temp;
			}

			// Find and add feature pixels to the begin of the string.
			pt.x = x;
			pt.y = y;
			ptrM = mask.data + loc;
			if ( next( pt, &ptrM, dir2 ) )
			{
				do
				{
					chain.push_back( pt );
					*ptrM = 0;
				} while ( next( pt, &ptrM, dir2 ) );
			}

			if ( chain.size() >= 10 )
			{
				edgeChains.push_back( chain );
			}
		}
	}
}

void MSED::edgeConnection( std::vector<std::vector<cv::Point> > &edgeChains )
{
	cv::Mat maskTemp( rows, cols, CV_64FC1, cv::Scalar( -1 ) );
	double * ptrM = (double*) maskTemp.data;

	for ( int i=0; i<edgeChains.size(); ++i )
	{
		for ( int j=0; j<edgeChains[i].size(); ++j )
		{
			int loc = edgeChains[i][j].y * cols + edgeChains[i][j].x;
			ptrM[loc] = i;
		}
	}

	int step = 10;
	std::vector<int> mergedIdx( edgeChains.size(), 0 );
	for ( int i=0; i<edgeChains.size(); ++i )
	{
		if ( mergedIdx[i] )
		{
			continue;
		}

		if ( i == 8 )
		{
			int aa = 0;
		}

		bool merged = false;
		int idxChain = 0, idxPixelStart = 0, idxPixelEnd = 0;

		if ( i == 8 )
		{
			int aa = 0;
		}
		// from the begin
		int t1 = min( step, (int)edgeChains[i].size() );	
		int j = 0;
		for ( j=0; j<t1; ++j )
		{
			int x = edgeChains[i][j].x;
			int y = edgeChains[i][j].y;
			if ( x < 1 || x >= cols_1 || y < 1 || y >= rows_1 )
			{
				continue;
			}

			double *ptrTemp = (double*) maskTemp.data + y * cols + x;
			for ( int m=0; m<8; ++m )
			{
				int idxSearched = *( ptrTemp + offsetTotal[m] );

				if ( idxSearched >= 0 && idxSearched != i )
				{
					if ( mergedIdx[idxSearched] )
					{
						continue;
					}

					int n = 0;
					int xSearched = x + offsetX[m];
					int ySearched = y + offsetY[m];
					for ( n=0; n<edgeChains[idxSearched].size(); ++n )
					{
						if ( edgeChains[idxSearched][n].x == xSearched && edgeChains[idxSearched][n].y == ySearched )
						{
							break;
						}
					}

					if ( n < step || n > edgeChains[idxSearched].size() - step )  // merge these two edge chain
					{
						merged = true;
						idxChain = idxSearched;
						if ( n < step )
						{
							idxPixelStart = n;
							idxPixelEnd = edgeChains[idxSearched].size();
						}
						else
						{
							idxPixelStart = n;
							idxPixelEnd = 0;
						}
						break;
					}
				}
			}

			if ( merged )
			{
				break;
			}
		}

		if ( merged )
		{
			std::vector<cv::Point> mergedChain;
			for ( int m=edgeChains[i].size()-1; m>=j; --m )
			{
				mergedChain.push_back( edgeChains[i][m] );
			}

			int order = 1;
			if ( idxPixelEnd < idxPixelStart )
			{
				order = -1;
			}

			for ( int m=idxPixelStart; m!=idxPixelEnd; m+= order )
			{
				mergedChain.push_back( edgeChains[idxChain][m] );				
			}
			edgeChains.push_back( mergedChain );

			for ( int m=0; m<j; ++m )
			{
				int loc = edgeChains[i][m].y * cols + edgeChains[i][m].x;
				ptrM[loc] = -1;
			}

			for ( int m=idxPixelStart; m!=idxPixelEnd; m+= order )
			{
				int loc = edgeChains[idxChain][m].y * cols + edgeChains[idxChain][m].x;
				ptrM[loc] = -1;
			}

			int idxTemp = edgeChains.size() - 1;
			for ( int m=0; m<mergedChain.size(); ++m )
			{
				int loc = mergedChain[m].y * cols + mergedChain[m].x;
				ptrM[loc] = idxTemp;
			}

			mergedIdx[i] = 1;
			mergedIdx[idxChain] = 1;
			mergedIdx.push_back( 0 );
			continue;
		}

		// from the end
		int t2 = max( 0, (int)edgeChains[i].size() - 1 - step );
		for ( j=edgeChains[i].size() - 1; j>t2; --j )
		{
			int x = edgeChains[i][j].x;
			int y = edgeChains[i][j].y;
			if ( x < 1 || x >= cols_1 || y < 1 || y >= rows_1 )
			{
				continue;
			}


			double *ptrTemp = (double*) maskTemp.data + y * cols + x;
			for ( int m=0; m<8; ++m )
			{
				int idxSearched = *( ptrTemp + offsetTotal[m] );

				if ( idxSearched >= 0  && idxSearched != i )
				{
					if ( mergedIdx[idxSearched] )
					{
						continue;
					}

					int n = 0;
					int xSearched = x + offsetX[m];
					int ySearched = y + offsetY[m];
					for ( n=0; n<edgeChains[idxSearched].size(); ++n )
					{
						if ( edgeChains[idxSearched][n].x == xSearched && edgeChains[idxSearched][n].y == ySearched )
						{
							break;
						}
					}

					if ( n < step || n > edgeChains[idxSearched].size() - step )  // merge these two edge chain
					{
						merged = true;
						idxChain = idxSearched;
						if ( n < step )
						{
							idxPixelStart = n;
							idxPixelEnd = edgeChains[idxSearched].size();
						}
						else
						{
							idxPixelStart = n;
							idxPixelEnd = 0;
						}
						break;
					}
				}
			}

			if ( merged )
			{
				break;
			}
		}

		if ( merged )
		{
			std::vector<cv::Point> mergedChain( edgeChains[i].begin(), edgeChains[i].begin() + j );

			int order = 1;
			if ( idxPixelEnd < idxPixelStart )
			{
				order = -1;
			}

			for ( int m=idxPixelStart; m!=idxPixelEnd; m+= order )
			{
				mergedChain.push_back( edgeChains[idxChain][m] );				
			}
			edgeChains.push_back( mergedChain );

			for ( int m=j; m<edgeChains[i].size(); ++m )
			{
				int loc = edgeChains[i][m].y * cols + edgeChains[i][m].x;
				ptrM[loc] = -1;
			}

			for ( int m=idxPixelStart; m!=idxPixelEnd; m+=order )
			{
				int loc = edgeChains[idxChain][m].y * cols + edgeChains[idxChain][m].x;
				ptrM[loc] = -1;
			}

			int idxTemp = edgeChains.size() - 1;
			for ( int m=0; m<mergedChain.size(); ++m )
			{
				int loc = mergedChain[m].y * cols + mergedChain[m].x;
				ptrM[loc] = idxTemp;
			}

			mergedIdx[i] = 1;
			mergedIdx[idxChain] = 1;
			mergedIdx.push_back( 0 );
			continue;
		}
	}

	std::vector<std::vector<cv::Point> > edgeChainsNew;
	for ( int i=0; i<mergedIdx.size(); ++i )
	{
		if ( !mergedIdx[i] )
		{
			edgeChainsNew.push_back( edgeChains[i] );
		}
	}
	edgeChains = edgeChainsNew;
}
