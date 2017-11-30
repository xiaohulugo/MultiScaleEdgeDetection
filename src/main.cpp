#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>

#include "MSED.h"
#include "Timer.h"

using namespace cv;
using namespace std;


void main()
{
	string inputImage = "D:\\test.jpg";
	string outputEdge = "D:\\test_edge.jpg";
	cv::Mat image=imread( inputImage, 0 );

	// MSEdge
	CTimer mtimer;
	char msg[1024];
	mtimer.Start();

	int gaussSize=3;
	double thLow = 30.0;
	double cth = 5.0;
	std::vector<std::vector<cv::Point> >  edgeChains;
	MSED edgeDetector;
	edgeDetector.MSEdge( image, gaussSize, thLow, cth, edgeChains );
	
	mtimer.Stop();
	double timeTotal = mtimer.elapsedTime;
	cout<<"time: "<<timeTotal<<endl;

	// draw edge chain
	cv::Mat imgShow(image.rows, image.cols, CV_8UC3, cv::Scalar(255,255,255) );
	uchar *ptr = imgShow.data;
	for ( int m=0; m<edgeChains.size(); ++m )
	{
		int R = rand() % 255;
		int G = rand() % 255;
		int B = rand() % 255;

		for ( int n=0; n<edgeChains[m].size(); ++n )
		{
			int x0 = edgeChains[m][n].x;
			int y0 = edgeChains[m][n].y;
			if ( x0 >=1 && x0 < image.cols -1 && y0 >=1 && y0 < image.rows -1 )
			{
				int loc = y0 * image.cols + x0;
				ptr[3*loc+0] = B;  ptr[3*loc+1] = G;  ptr[3*loc+2] = R;

				ptr[3*(loc+1)+0] = B;  ptr[3*(loc+1)+1] = G;  ptr[3*(loc+1)+2] = R;
				ptr[3*(loc-1)+0] = B;  ptr[3*(loc-1)+1] = G;  ptr[3*(loc-1)+2] = R;
				ptr[3*(loc-image.cols)+0] = B;  ptr[3*(loc-image.cols)+1] = G;  ptr[3*(loc-image.cols)+2] = R;
				ptr[3*(loc+image.cols)+0] = B;  ptr[3*(loc+image.cols)+1] = G;  ptr[3*(loc+image.cols)+2] = R;
			}
		}
	}
	imwrite( outputEdge,imgShow );
}
