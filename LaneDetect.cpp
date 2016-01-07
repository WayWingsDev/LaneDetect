// LaneDetect5.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "core/core.hpp"
#include "highgui/highgui.hpp"
#include "imgproc/imgproc.hpp"
#include "objdetect/objdetect.hpp"
#include <iostream>

#include "LaneDetection.h"

#define videoName "D:\\1_Work_sari\\1_��Ŀ�ļ�\\20141226_��������ǰ�����\\1_Code\\3_LaneDetect\\Video\\Test3.avi"
#define START_NUM 4000
#define STOP_NUM 6500

int _tmain(int argc, _TCHAR* argv[])
{	
	// �������
	LaneDetection laneDetect;
	
	// ���Ʋ���
	int frameNum = 0;
	double t0;

	// ��ȡ��Ƶ
	VideoCapture cap(videoName);
	if (!cap.isOpened())	return -1;
	cap.set(CV_CAP_PROP_POS_FRAMES, START_NUM);	

	// ���̲���
	Mat OFrame;		//ԭʼͼ��

	// ��ʼ��
	laneDetect.initSys();
	frameNum = START_NUM;

	while (1)
	{
		cap >> OFrame;
		frameNum++;

		cout << "Frame No. " << frameNum << endl;

		if (!OFrame.empty())
		{
			t0 = (double)cvGetTickCount();

			laneDetect.process(OFrame);
			
			t0 = (double)cvGetTickCount() - t0;
			t0 = t0 / ((double)cvGetTickFrequency()*1000.);
			
			laneDetect.drawLanes(OFrame);
			laneDetect.nextFrame();

			cout << "ʱ�����ģ�Total Time = " << t0 << " ms" << endl;
		}
		cout << endl;
		if (frameNum > STOP_NUM)
		{
			waitKey();
		}
	}
	
	return 0;
}

