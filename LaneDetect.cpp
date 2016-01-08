// LaneDetect5.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "core/core.hpp"
#include "highgui/highgui.hpp"
#include "imgproc/imgproc.hpp"
#include "objdetect/objdetect.hpp"
#include <iostream>

#include "LaneDetection.h"

#define videoName "..\\..\\Video\\Test3.avi"
#define START_NUM 0000
#define STOP_NUM 6500

int _tmain(int argc, _TCHAR* argv[])
{	
	LaneDetection laneDetect;
	Mat curFrame;
	int frameNum = START_NUM;
	double t0;
	
	// load video
	VideoCapture cap(videoName);
	if (!cap.isOpened())	return -1;
	cap.set(CV_CAP_PROP_POS_FRAMES, START_NUM);	

	// system init
	laneDetect.initSys();

	while (1)
	{
		cap >> curFrame;
		frameNum++;

		if (frameNum % 40 == 0)
		{
			cout << "Frame No. " << frameNum << endl;
		}
		
		if (!curFrame.empty())
		{
			t0 = (double)getTickCount();

			laneDetect.process(curFrame);
			
			t0 = (double)getTickCount() - t0;
			t0 = t0 / ((double)getTickFrequency()*1000.);
			
			laneDetect.drawLanes(curFrame);
			//laneDetect.nextFrame();

			if (t0 < 30)
			{
				waitKey(40 - t0);
			}
			else
			{
				waitKey(10);
			}		
			//cout << "Total Time = " << t0 << " ms" << endl;
		}
		if (frameNum > STOP_NUM)
		{
			waitKey();
		}
	}	
	return 0;
}

