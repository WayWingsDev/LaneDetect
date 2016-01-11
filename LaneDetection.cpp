#include "core/core.hpp"
#include "highgui/highgui.hpp"
#include "imgproc/imgproc.hpp"
#include "objdetect/objdetect.hpp"
#include "LaneDetection.h"
#include <iostream>
#include <numeric>

#define cascadeName "LBPcascade_lane.xml"
#define Y_SKYLINE 0.3		//地平线高度

// public part
void LaneDetection::initSys()
{
	// 显示窗口
	namedWindow("LD", 1);
	moveWindow("LD", 520, 30);
	namedWindow("Canny", 1);
	moveWindow("Canny", 900, 30);
	namedWindow("Gradient", 1);
	moveWindow("Gradient", 520, 350);
	namedWindow("Lines", 1);
	moveWindow("Lines", 520, 670);
	namedWindow("Mask", 1);
	moveWindow("Mask", 900, 670);
	namedWindow("BV", 1);
	moveWindow("BV", 1280, 30);

	// 分类器载入
	cascade.load(cascadeName);
	
	// 计算投影矩阵
	initTransMatrix();

	// 初始化deque
	dDirection.resize(12, 0);
}

void LaneDetection::process(Mat& frame)
{
	OFrame = frame.clone();		// copy of the origin frame

	frameInit();
	getCandidatePoints();
	getCannyArea();
	getCandidateLines();
	getTrackedLines();
	nextFrame();
}

void LaneDetection::drawLanes(Mat& frame)
{
	Mat linesFrame = Mat::zeros(SFrame.size(), CV_8U);
	Mat frameShow = SFrame.clone();
	if (departure == 0)
	{	
		plotSingleLine(linesFrame, fitLinesL[0], 500);
		plotSingleLine(linesFrame, fitLinesL[1], 500);
		plotSingleLine(linesFrame, fitLinesR[0], 500);
		plotSingleLine(linesFrame, fitLinesR[1], 500);
		plotLanes(frameShow, lanesPV.LLane, lanesPV.RLane);		
	}
	else
	{
		plotSingleLine(linesFrame, fitLinesM[0], 500);
		plotSingleLine(linesFrame, fitLinesM[1], 500);
		plotLanes(frameShow, lanesPV.MLane);
	}
	
	imshow("Mask", mask);
	imshow("Canny", edge);
	imshow("Gradient", gradient);
	imshow("Lines", linesFrame);
	imshow("LD", frameShow);
}

// private part
void LaneDetection::initTransMatrix()
{
	Point2f src[4];
	Point2f dst[4];

	src[0].x = 186;
	src[0].y = 80;
	src[1].x = 197;
	src[1].y = 80;
	src[2].x = 43;
	src[2].y = 202;
	src[3].x = 348;
	src[3].y = 202;

	dst[0].x = 100;
	dst[0].y = -6000;
	dst[1].x = 200;
	dst[1].y = -6000;
	dst[2].x = 100;
	dst[2].y = 688;
	dst[3].x = 200;
	dst[3].y = 688;

	Mat transMatrix = getPerspectiveTransform(src, dst);
	double* p;
	for (unsigned i = 0; i < 3; i++)
	{
		p = transMatrix.ptr<double>(i);

		for (unsigned j = 0; j < 3; j++)
		{
			transM[3 * i + j] = p[j];
		}
	}

	Mat retransMatrix = getPerspectiveTransform(dst, src);
	for (unsigned i = 0; i < 3; i++)
	{
		p = retransMatrix.ptr<double>(i);

		for (unsigned j = 0; j < 3; j++)
		{
			retransM[3 * i + j] = p[j];
		}
	}
}

void LaneDetection::frameInit()
{
	cvtColor(OFrame, OFrame, CV_BGR2GRAY);
	resize(OFrame, SFrame, Size(OFrame.cols / 2, OFrame.rows / 2));
	SFrame = SFrame(Rect(0, 10, SFrame.cols - 10, SFrame.rows - 10));
	frameSize = SFrame.size();
	// define searchROI
	if (departure == 0)
	{
		searchROI = Rect(0, Y_SKYLINE*SFrame.rows, SFrame.cols, (1 - Y_SKYLINE - 0.1)*SFrame.rows);
	}
	else if (departure == -1)
	{
		searchROI = Rect(0.25*SFrame.cols, Y_SKYLINE*SFrame.rows, 0.75*SFrame.cols, (1 - Y_SKYLINE)*SFrame.rows);
	}
	else if (departure == 1)
	{
		searchROI = Rect(0.25*SFrame.cols, Y_SKYLINE*SFrame.rows, 0.75*SFrame.cols, (1 - Y_SKYLINE)*SFrame.rows);
	}
}

void LaneDetection::getCandidatePoints()
{
	Mat frameROI = SFrame(searchROI);
	lanePoints.clear();
	vector<Rect> laneTargets;
	cascade.detectMultiScale(frameROI, laneTargets, 1.4, 1, 0, Size(30, 30), Size(60, 60));

	for (unsigned i = 0; i < laneTargets.size(); i++)
	{
		Point plotPoint;
		plotPoint.x = laneTargets[i].x + laneTargets[i].width / 2 + searchROI.x;
		plotPoint.y = laneTargets[i].y + laneTargets[i].height / 2 + searchROI.y;
		lanePoints.push_back(plotPoint);
	}
}

void LaneDetection::getCannyArea()
{
	// mask
	mask = Mat::zeros(frameSize, CV_8U);
	if (departure == 0)
	{
		delErrorPoints(lanePoints, selectedPoints);		
		processMask(mask, selectedPoints);
	}
	else
	{
		delErrorPointsDepart(lanePoints, selectedPoints);
		processMaskDepart(mask, selectedPoints);
	}
	
	// canny
	Mat frameROI = SFrame(searchROI);
	cannyFrame = Mat::zeros(frameSize, CV_8U);
	Mat cannyROI = cannyFrame(searchROI);
	Canny(frameROI, cannyROI, 20, 120, 3, false);

	edge = Mat::zeros(frameSize, CV_8U);
	cannyFrame.copyTo(edge, mask);
}

void LaneDetection::processMask(Mat& mask, vector<Point>& points)
{
	for (unsigned i = 0; i < points.size(); i++)
	{
		circle(mask, points[i], 15, 255, -1);
	}

	plotQuadMask(mask, lanesPV.LLane_p);
	plotQuadMask(mask, lanesPV.RLane_p);
}

void LaneDetection::plotQuadMask(Mat& mask, Vec4f line)
{
	if (line[0] != 0 || line[1] != 0)
	{
		int nCols = mask.cols;
		int nRows = mask.rows;

		uchar* p;

		for (unsigned i = 0.45 * nRows; i < nRows; i++)
		{
			p = mask.ptr<uchar>(i);

			Point point0 = pointInLine(line, i);
			int x0 = point0.x;
			for (int j = x0 - 15; j < x0 + 15; j++)
			{
				if (j >= 0 && j < nCols)
				{
					p[j] = 255;
				}
			}
		}
	}
}

void LaneDetection::delErrorPoints(vector<Point>& ps0, vector<Point>& ps)
{
	ps.clear();
	if (lanesPV.LLane_p[1] != 0 && lanesPV.RLane_p[1] != 0)
	{
		for (unsigned i = 0; i < ps0.size(); i++)
		{
			int dist1 = dist2line(ps0[i], lanesPV.LLane_p);
			int dist2 = dist2line(ps0[i], lanesPV.RLane_p);
			
			if (dist1 < 200 || dist2 < 200)
			{
				ps.push_back(ps0[i]);
			}
		}
	}
	else
	{
		ps = ps0;
	}
}

void LaneDetection::delErrorPointsDepart(vector<Point>& ps0, vector<Point>& ps)
{
	ps.clear();
	if (lanesPV.MLane_p[1] != 0)	
	{
		for (unsigned i = 0; i < ps0.size(); i++)
		{
			int dist = dist2line(ps0[i], lanesPV.MLane_p);

			if (dist < 200)
			{
				ps.push_back(ps0[i]);
			}
		}
	}
	else
	{
		ps = ps0;
	}
}


void LaneDetection::getCandidateLines()
{
	Mat frameGrey = SFrame(searchROI);

	// x-direction
	Mat maskX = (Mat_<char>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	filter2D(frameGrey, xFrame, CV_32F, maskX);
	xFrame2 = xFrame.mul(xFrame);

	// y-direction
	Mat maskY = (Mat_<char>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
	filter2D(frameGrey, yFrame, CV_32F, maskY);
	yFrame2 = yFrame.mul(yFrame);

	// gradient
	gFrame2 = xFrame2 + yFrame2;
	sqrt(gFrame2, gFrame);
	tanFrame = yFrame / xFrame;

	Mat element(3, 3, CV_8U, Scalar(1));
	dilate(edge, edge, element);

	// state of lane keeping 
	gradientU = Mat::zeros(searchROI.height, searchROI.width, CV_8U);
	gradientD = Mat::zeros(searchROI.height, searchROI.width, CV_8U);
	
	// departing conditons
	switch (departure)
	{
	case 0:
	{
		vector<Point> fitPointsLU;
		vector<Point> fitPointsLD;
		vector<Point> fitPointsRU;
		vector<Point> fitPointsRD;

		int xDivide = binSplitLaneArea();
		int dx = SFrame.cols - xDivide;
		int y1 = searchROI.y;
		int dy = searchROI.height;

		Rect areaL(0, 0, xDivide, dy);
		Rect areaR(xDivide, 0, dx, dy);

		singleSideLines(areaL, fitLinesL, fitPointsLU, fitPointsLD, Point(0, y1));
		singleSideLines(areaR, fitLinesR, fitPointsRU, fitPointsRD, Point(xDivide, y1));

		lanesPV.relia[0] = fitPointsLU.size();
		lanesPV.relia[1] = fitPointsLD.size();
		lanesPV.relia[2] = fitPointsRU.size();
		lanesPV.relia[3] = fitPointsRD.size();
		fitPoints.clear();
		fitPoints.insert(fitPoints.end(), fitPointsLU.begin(), fitPointsLU.end());
		fitPoints.insert(fitPoints.end(), fitPointsLD.begin(), fitPointsLD.end());
		fitPoints.insert(fitPoints.end(), fitPointsRU.begin(), fitPointsRU.end());
		fitPoints.insert(fitPoints.end(), fitPointsRD.begin(), fitPointsRD.end());
		
		break;
	}

	case -1:
	case 1:
	{
		vector<Point> fitPointsL;
		vector<Point> fitPointsR;

		Rect area(0, 0, searchROI.x, searchROI.y);

		singleSideLines(area, fitLinesM, fitPointsL, fitPointsR, Point(searchROI.x, searchROI.y));

		lanesPV.relia[0] = fitPointsL.size();
		lanesPV.relia[1] = fitPointsR.size();
		fitPoints.clear();
		fitPoints.insert(fitPoints.end(), fitPointsL.begin(), fitPointsL.end());
		fitPoints.insert(fitPoints.end(), fitPointsR.begin(), fitPointsR.end());

		break;
	}
	}

	addWeighted(gradientU, 0.5, gradientD, 0.5, 0.0, gradient);

	// imshow("GradientU", gradientU);
	// imshow("GradientD", gradientD);
}

void LaneDetection::singleSideLines(Rect area, Vec4f (&fitLines)[2], vector<Point>& fitPointsU, vector<Point>& fitPointsD, Point move)
{
	fitPointsU.clear();
	fitPointsD.clear();
	
	Mat GU = gradientU(area);
	Mat GD = gradientD(area);

	// area selecting
	Mat areaEdge = edge(searchROI)(area);
	Mat areaX = xFrame(area);
	Mat areaY = yFrame(area);
	Mat areaG = gFrame(area);
	Mat areaTan = tanFrame(area);
	 
	// calc the oriented-gradient
	float angleStep = CV_PI / 9;
	float gHist[9] = { 0 };
	int gNumHist[9] = { 0 };
	selected = Mat::zeros(areaTan.rows, areaTan.cols, CV_8U);
	float* pTan;	// Angle
	float* pY;		// Y Gradient
	float* pG;		// Gradient
	uchar* pSP;		// Selected Points
	uchar* pSPU;	// Selected Points Up
	uchar* pSPD;	// Selected Points Down
	uchar* pE;		// Edge frame
	for (unsigned j = 0; j < areaTan.rows; j++)
	{
		pTan = areaTan.ptr<float>(j);
		pG = areaG.ptr<float>(j);
		pE = areaEdge.ptr<uchar>(j);
		pSP = selected.ptr<uchar>(j);
		for (unsigned i = 0; i < areaTan.cols; i++)
		{
			if (pE[i]>200)
			{
				pTan[i] = atan(pTan[i]) + CV_PI / 2;
				int groupIndex = pTan[i] / angleStep;
				gHist[groupIndex] += pG[i];
				gNumHist[groupIndex] += 1;
				pSP[i] = groupIndex;
			}
		}
	}

	// search for the direction of the largest gradient
	float maxGradient = 0;
	int maxIndex = 0;
	for (unsigned i = 0; i < 9; i++)
	{
		if (gHist[i] > maxGradient)
		{
			maxGradient = gHist[i];
			maxIndex = i;
		}
	}
	
	Mat areaD;
	switch (departure)
	{
	case 0:
	{
		areaD = areaY;
		break;
	}

	case -1:
	case 1:
	{
		areaD = areaX;
		break;
	}
	}


	// refine the gradient
	if (maxGradient > 0)
	{
		for (unsigned j = 0; j < selected.rows; j++)
		{
			pY = areaD.ptr<float>(j);
			pG = areaG.ptr<float>(j);
			pSP = selected.ptr<uchar>(j);
			pSPU = GU.ptr<uchar>(j);
			pSPD = GD.ptr<uchar>(j);
			for (unsigned i = 0; i < selected.cols; i++)
			{
				if (pSP[i] == maxIndex && pY[i] > 0)
				{
					pSP[i] = 255;
					pSPU[i] = 255;
					fitPointsU.push_back(Point(i, j) + move);
				}
				else if (pSP[i] == maxIndex && pY[i] < 0)
				{
					pSP[i] = 255;
					pSPD[i] = 255;
					fitPointsD.push_back(Point(i, j) + move);
				}
				else
				{
					pSP[i] = 0;
				}
			}
		}
	}

	// calc the line fitting
	Vec4f fitLinesU;
	Vec4f fitLinesD;
	//Mat allBlack = Mat::zeros(GU.size(), GU.type());
	if (fitPointsU.size() > 1000)
	{
		//allBlack.copyTo(GU);
		fitPointsU.clear();
	}
	if (fitPointsD.size() > 1000)
	{
		//allBlack.copyTo(GD);
		fitPointsD.clear();
	}

	// weighted least-square fitting
	if (fitPointsU.size() > 5)
	{
		fitLineWeighted(fitPointsU, fitLinesU);
	}
	if (fitPointsD.size() > 5)
	{
		fitLineWeighted(fitPointsD, fitLinesD);
	}

	fitLines[0] = fitLinesU;
	fitLines[1] = fitLinesD;
}

int LaneDetection::binSplitLaneArea()
{
	return SFrame.cols / 2;
}

// target tracking
void LaneDetection::getTrackedLines()
{
	switch (departure)
	{
	case 0:
	{
		Vec4f LLane;
		Vec4f RLane;

		int nCols = SFrame.cols;
		int nRows = SFrame.rows;

		lanesPV.confirmed[0] = false;
		lanesPV.confirmed[1] = false;

		int leftP;
		int rightP;
		leftP = parallelTest(lanesPV.LLane_p, fitLinesL, lanesPV.tracked[0]);
		rightP = parallelTest(lanesPV.RLane_p, fitLinesR, lanesPV.tracked[1]);

		LLane = mergeLines(lanesPV.LLane_p, fitLinesL, leftP, lanesPV.relia[0], lanesPV.relia[1]);
		RLane = mergeLines(lanesPV.RLane_p, fitLinesR, rightP, lanesPV.relia[2], lanesPV.relia[3]);

		// update the parameter: confirmed
		if (LLane[0] != 0 && LLane[1] != 0)
		{
			lanesPV.confirmed[0] = true;
		}
		if (RLane[0] != 0 && RLane[1] != 0)
		{
			lanesPV.confirmed[1] = true;
		}

		// get the other side lane
		if (lanesPV.confirmed[0] == true && lanesPV.confirmed[1] == false)	// only left
		{
			if (lanesPV.relia[0] + lanesPV.relia[1] > 150)
			{
				RLane = getOtherSideLane(LLane, 110);
			}
		}

		if (lanesPV.confirmed[0] == false && lanesPV.confirmed[1] == true)	// only right
		{
			if (lanesPV.relia[2] + lanesPV.relia[3] > 150)
			{
				LLane = getOtherSideLane(RLane, -110);
			}
		}

		// track the lane target
		if (lanesPV.LLane_p[0] != 0 && lanesPV.tracked[0])
		{
			lanesPV.LLane = 0.5*LLane + 0.5*lanesPV.LLane_p;
		}
		else
		{
			lanesPV.LLane = LLane;
		}

		if (lanesPV.RLane_p[0] != 0 && lanesPV.confirmed[1])
		{
			lanesPV.RLane = 0.5*RLane + 0.5*lanesPV.RLane_p;
		}
		else
		{
			lanesPV.RLane = RLane;
		}

		calcDepartDirection(dDirection);

		//cout << "LLane: " << lanesPV.LLane << endl;
		//cout << "RLane: " << lanesPV.RLane << endl;

		// update the parameter: tracked
		lanesPV.LLane_p = lanesPV.LLane;
		lanesPV.RLane_p = lanesPV.RLane;
		if (leftP < 3 || leftP == 4)
		{
			lanesPV.tracked[0] = true;
		}
		else
		{
			lanesPV.tracked[0] = false;
		}
		if (rightP < 3 || rightP == 4)
		{
			lanesPV.tracked[1] = true;
		}
		else
		{
			lanesPV.tracked[1] = false;
		}
		break;
	}

	case -1:
	case 1:
	{
		Vec4f MLane;

		int nCols = SFrame.cols;
		int nRows = SFrame.rows;

		lanesPV.tracked[0] = false;

		int midP = parallelTest(lanesPV.MLane_p, fitLinesM, lanesPV.confirmed[0]);
		MLane = mergeLines(lanesPV.MLane_p, fitLinesM, midP, lanesPV.relia[0], lanesPV.relia[1]);

		if (MLane[0] != 0 && MLane[1] != 0)
		{
			lanesPV.tracked[0] = true;
		}

		if (lanesPV.MLane_p[0] != 0 && lanesPV.tracked[0])
		{
			lanesPV.MLane = mergeTwoLines(MLane, lanesPV.MLane_p, 0.8);
			//lanesPV.MLane = 0.8*MLane + 0.2*lanesPV.MLane_p;
		}
		else
		{
			lanesPV.MLane = MLane;
		}

		calcDepartDirection(dDirection);

		// update
		lanesPV.MLane_p = lanesPV.MLane;
		if (midP < 3 || midP == 4)
		{
			lanesPV.confirmed[0] = true;
		}
		else
		{
			lanesPV.confirmed[0] = false;
		}

		break;
	}
	}

	// bird's view
	Mat tFrame = Mat::zeros(Size(300, 720), CV_8UC3);
	vector<Point> projectedPoints;	
	getProjectedPoints(fitPoints, projectedPoints, transM);
	for (unsigned i = 0; i < projectedPoints.size(); i++)
	{
		circle(tFrame, projectedPoints[i], 3, Scalar(255, 0, 0), (-1));
	}
	imshow("BV", tFrame);
}

int LaneDetection::parallelTest(Vec4f line0, Vec4f(&lines)[2], bool tracked)
{
	// result
	// 0: 0、1线都与上一帧结果平行
	// 1: 0线（上）与上一帧结果不平行
	// 2: 1线（下）与上一帧结果不平行
	// 3: 0、1线与上一帧结果都不平行
	// 4: 无上一帧结果，0、1线相互平行
	// 5: 无上一帧结果，0、1线相互不平行

	int result = 0;

	if (tracked && line0[0] != 0 && line0[1] != 0)		//存在上一帧目标
	{
		bool p0 = false;
		bool p1 = false;
		if (lines[0][0] != 0 && lines[0][1] != 0)	//0线存在
		{
			p0 = parallelIndex(line0, lines[0]);		//0线与上一帧的平行关系
		}
		if (lines[1][0] != 0 && lines[1][1] != 0)	//1线存在
		{
			p1 = parallelIndex(line0, lines[1]);		//1线与上一帧的平行关系
		}

		if (!p0)
		{
			lines[0] = line0;
			result = 1;
		}
		if (!p1)
		{
			lines[1] = line0;
			result = 2;
		}
		if (!p0 && !p1)
		{
			result = 3;
		}
	}
	else if (lines[0][0] != 0 && lines[1][0] != 0)
	{
		result = 4;
		bool p = parallelIndex(lines[0], lines[1]);
		if (!p)
		{
			result = 5;
		}
	}

	return result;
}

bool LaneDetection::parallelIndex(Vec4f line0, Vec4f line1)
{
	double pIndex = 0;	// gradient
	double dIndex = 0;	// distance

	int dy = frameSize.height;
	Point p00 = pointInLine(line0, dy / 2);
	Point p01 = pointInLine(line0, dy);
	Point p10 = pointInLine(line1, dy / 2);
	Point p11 = pointInLine(line1, dy);
	int dist0 = p10.x - p00.x;
	int dist1 = p11.x - p01.x;

	float d1 = abs(line0[0] - line1[0]);
	float d2 = abs(line0[1] - line1[1]);
	float s1 = abs(line0[0] + line1[0]);
	float s2 = abs(line0[1] + line1[1]);

	pIndex = s1 > s2 ? d1 : d2;
	dIndex = abs(dist0 - dist1);

	bool isParallel = true;

	if (100 * pIndex + dIndex > 40)
	{
		isParallel = false;
	}

	return isParallel;
}

Vec4f LaneDetection::mergeLines(Vec4f line0, Vec4f(&lines)[2], int pTest, int relia0, int relia1)
{
	Vec4f outLine;

	switch (pTest)
	{
	case 0:
	{
		if (relia0 + relia1 > 50)
		{
			outLine = mergeTwoLines(lines[0], lines[1], 0.5);
			//outLine = 0.5*lines[0] + 0.5*lines[1];
		}
		break;
	}

	case 1:
	{
		if (relia1>30)
		{
			outLine = lines[1];
		}
		break;
	}

	case 2:
	{
		if (relia0 > 30)
		{
			outLine = lines[0];
		}
		break;
	}

	case 3:
	{
		break;
	}

	case 4:
	{
		if (relia0 + relia1 > 50)
		{
			outLine = mergeTwoLines(lines[0], lines[1], 0.5);
			//outLine = 0.5*lines[0] + 0.5*lines[1];
		}
		break;
	}

	case 5:
	{
		break;
	}

	default:
		break;
	}

	return outLine;
}

Vec4f LaneDetection::getOtherSideLane(Vec4f cLane, double deviation)
{
	int nRows = frameSize.height;
	int nCols = frameSize.width;

	Point p0 = pointInLine(cLane, 0.5*nRows, true);
	Point p1 = pointInLine(cLane, nRows, true);

	Point2f p0_BV = getProjectedPoint(p0, transM);
	Point2f p1_BV = getProjectedPoint(p1, transM);

	double dist = sqrt((p0_BV.x - p1_BV.x)*(p0_BV.x - p1_BV.x) + (p0_BV.y - p1_BV.y)*(p0_BV.y - p1_BV.y));
	double dx = deviation * (p1_BV.y - p0_BV.y) / dist;
	double dy = deviation * (p1_BV.x - p0_BV.x) / dist;

	Point2f p2_BV, p3_BV;
	p2_BV.x = p0_BV.x + dx;
	p2_BV.y = p0_BV.y + dy;
	p3_BV.x = p1_BV.x + dx;
	p3_BV.y = p1_BV.y + dy;

	Point2f p2 = getProjectedPoint(p2_BV, retransM);
	Point2f p3 = getProjectedPoint(p3_BV, retransM);

	//cout << "(" << p2_BV.x << ", " << p2_BV.y << "), （" << p3_BV.x << ", " << p3_BV.y << ")" << endl;

	dist = sqrt((p2.x - p3.x)*(p2.x - p3.x) + (p2.y - p3.y)*(p2.y - p3.y));
	Vec4f addLane;
	addLane[0] = (p3.x - p2.x) / dist;
	addLane[1] = (p3.y - p2.y) / dist;
	addLane[2] = p3.x;
	addLane[3] = p3.y;

	if (addLane[0] < 0)
	{
		addLane[0] = -addLane[0];
		addLane[1] = -addLane[1];
	}

	return addLane;
}

void LaneDetection::calcDepartDirection(deque<int>& dDirection)
{
	// dDirection: -1：向左偏， 0：不确定， 1：向右偏
	if (departure == 0)
	{
		float alpha, alpha_p;
		float dL = 0;
		float dR = 0;

		if (lanesPV.LLane[1] != 0 && lanesPV.LLane_p[1] != 0)
		{
			alpha = atan(lanesPV.LLane[0] / lanesPV.LLane[1]);
			alpha_p = atan(lanesPV.LLane_p[0] / lanesPV.LLane_p[1]);
			dL = alpha - alpha_p;
		}
		if (lanesPV.RLane[1] != 0 && lanesPV.RLane_p[1] != 0)
		{
			alpha = atan(lanesPV.RLane[0] / lanesPV.RLane[1]);
			alpha_p = atan(lanesPV.RLane_p[0] / lanesPV.RLane_p[1]);
			dR = alpha - alpha_p;
		}

		//cout << "dL = " << dL << endl;
		//cout << "dR = " << dR << endl;
		//cout << "dL+dR = " << dL + dR << endl;

		if (dL + dR > 0)
		{
			dDirection.push_back(-1);
			dDirection.pop_front();
		}
		else if (dL + dR < 0)
		{
			dDirection.push_back(1);
			dDirection.pop_front();
		}
		else
		{
			dDirection.push_back(0);
			dDirection.pop_front();
		}
	}
	else
	{
		float alpha, alpha_p;
		float dM = 0;
		if (lanesPV.MLane[1] != 0 && lanesPV.MLane_p[1] != 0)
		{
			alpha = atan(lanesPV.MLane[0] / lanesPV.MLane[1]);
			alpha_p = atan(lanesPV.MLane_p[0] / lanesPV.MLane_p[1]);
			dM = alpha - alpha_p;
		}

		//cout << "dM = " << dM << endl;

		if (dM > 0)
		{
			dDirection.push_back(-1);
			dDirection.pop_front();
		}
		else if (dM < 0)
		{
			dDirection.push_back(1);
			dDirection.pop_front();
		}
		else
		{
			dDirection.push_back(0);
			dDirection.pop_front();
		}
	}
}

void LaneDetection::nextFrame()
{
	//更新偏离状态的指示
	int sum = accumulate(begin(dDirection), end(dDirection), 0.0);
	//cout << "Sum = " << sum << endl; 

	if (departure == 0)				//当前为正常行驶
	{
		float Lx = abs(lanesPV.LLane[0]);
		float Rx = abs(lanesPV.RLane[0]);

		if (Lx > 0 && Lx < 0.60 && Rx > 0.60 && sum < -4)		// 向左偏离趋势大
		{
			departure = -1;
			lanesPV.MLane_p = lanesPV.LLane;
			//cout << "MLane_p = " << lanesPV.MLane_p << endl;
		}
		else if (Rx > 0 && Rx < 0.60 && Lx > 0.60 && sum > 4)	// 向右偏离趋势大
		{
			departure = 1;
			lanesPV.MLane_p = lanesPV.RLane;
		}
		else
		{
			departure = 0;
		}

		if (departure != 0)			// 从正常向偏离转变，重置参数
		{
			lanesPV.MLane = Vec4f(0, 0, 0, 0);
			//lanesPV.MLane_p = Vec4f(0, 0, 0, 0);

			if (departure == -1)
			{
				lanesPV.confirmed[0] = lanesPV.confirmed[1];
			}
		}
	}
	else if (departure == -1)			// 当前为向左偏离
	{
		float Mx = lanesPV.MLane[0];
		float My = lanesPV.MLane[1];

		if (Mx > 0.4)
		{
			if (My > 0 && sum < -4)			// 完成向左跨道
			{
				departure = 0;

				lanesPV.LLane = Vec4f(0, 0, 0, 0);
				lanesPV.LLane_p = Vec4f(0, 0, 0, 0);
				lanesPV.RLane = Vec4f(0, 0, 0, 0);
				lanesPV.RLane_p = lanesPV.MLane;
				lanesPV.confirmed[1] = lanesPV.confirmed[0];
				lanesPV.confirmed[0] = false;
			}
			else if (My < 0 && sum > 4)		// 退回当前车道
			{
				departure = 0;

				lanesPV.LLane = Vec4f(0, 0, 0, 0);
				lanesPV.LLane_p = lanesPV.MLane;
				lanesPV.RLane = Vec4f(0, 0, 0, 0);
				lanesPV.RLane_p = Vec4f(0, 0, 0, 0);
				lanesPV.confirmed[1] = false;
			}
		}
	}
	else if (departure == 1)			// 当前为向右偏离
	{
		float Mx = lanesPV.MLane[0];
		float My = lanesPV.MLane[1];

		if (Mx > 0.4)
		{
			if (My < 0 && sum > 4)			// 完成向右跨道
			{
				departure = 0;

				lanesPV.LLane = Vec4f(0, 0, 0, 0);
				lanesPV.LLane_p = lanesPV.MLane;
				lanesPV.RLane = Vec4f(0, 0, 0, 0);
				lanesPV.RLane_p = Vec4f(0, 0, 0, 0);
				lanesPV.confirmed[1] = false;

			}
			else if (My > 0 && sum < -4)		// 退回当前车道
			{
				departure = 0;

				lanesPV.LLane = Vec4f(0, 0, 0, 0);
				lanesPV.LLane_p = Vec4f(0, 0, 0, 0);
				lanesPV.RLane = Vec4f(0, 0, 0, 0);
				lanesPV.RLane_p = lanesPV.MLane;
				lanesPV.confirmed[1] = lanesPV.confirmed[0];
				lanesPV.confirmed[0] = false;
			}
		}

	}
	//cout << "Departure: " << departure << endl;
}

// 其他
Point2f LaneDetection::pointInLine(Vec4f cLine, int p, bool isY)
{
	Point2f cPoint;
	if (isY)
	{
		cPoint.x = (p - cLine[3]) / cLine[1] * cLine[0] + cLine[2];
		cPoint.y = p;
	}
	else
	{
		cPoint.x = p;
		cPoint.y = (p - cLine[2]) / cLine[0] * cLine[1] + cLine[3];
	}
	return cPoint;
}

int LaneDetection::dist2line(Point pt, Vec4f& lane)
{
	float x1 = lane[2];
	float y1 = lane[3];
	float x2 = x1 + 100 * lane[0];
	float y2 = y1 + 100 * lane[1];

	int A = y1 - y2;
	int B = x2 - x1;
	int C = x1*y2 - x2*y1;

	int num = A*pt.x + B*pt.y + C;
	int den = A*A + B*B;

	int dist2;
	if (den != 0)
	{
		dist2 = num*num / den;
	}
	else
	{
		dist2 = 0;
	}

	/*
	if (A*num < 0)
	{
		dist2 = -dist2;
	}
	*/

	return dist2;
}

Point LaneDetection::crossPoint(Vec4f line1, Vec4f line2)
{
	float k1 = line1[1] / line1[0];
	float k2 = line2[1] / line2[0];
	float b1 = line1[3] - k1 * line1[2];
	float b2 = line2[3] - k2 * line2[2];
	float x0 = (b2 - b1) / (k1 - k2);
	float y0 = k1 * x0 + b1;

	return Point(x0, y0);
}

void LaneDetection::fitLineWeighted(vector<Point>& points, Vec4f& curLine)
{
	double M00 = 0;
	double M01 = 0;
	double M10 = 0;
	double M11 = 0;
	double N0 = 0;
	double N1 = 0;

	for (unsigned i = 0; i < points.size(); i++)
	{
		double x = points[i].x;
		double y = points[i].y;
		M00 += x*x*y*y;
		M01 += x*y*y;
		M10 += x*y*y;
		M11 += y*y;
		N0 += x*y*y*y;
		N1 += y*y*y;
	}

	Mat MM = (Mat_<double>(2, 2) << M00, M01, M10, M11);
	Mat NN = (Mat_<double>(2, 1) << N0, N1);

	Mat fLine;
	solve(MM, NN, fLine, DECOMP_SVD);

	double a = fLine.at<double>(0, 0);
	double b = fLine.at<double>(1, 0);

	//转换直线表示方式
	double angle = atan(a);
	curLine[0] = cos(angle);
	curLine[1] = sin(angle);
	curLine[2] = -b / a;
	curLine[3] = 0;
}

Point2f LaneDetection::getProjectedPoint(Point2f pointPV, double transM[])
{
	double x = pointPV.x*transM[0] + pointPV.y*transM[1] + transM[2];
	double y = pointPV.x*transM[3] + pointPV.y*transM[4] + transM[5];
	double z = pointPV.x*transM[6] + pointPV.y*transM[7] + transM[8];
	x = x / z;
	y = y / z;

	return Point2f(x, y);
}

void LaneDetection::getProjectedPoints(vector<Point>& lanePoints, vector<Point>& projectedPoints, double transM[])
{
	projectedPoints.clear();

	for (unsigned i = 0; i < lanePoints.size(); i++)
	{
		double x = lanePoints[i].x*transM[0] + lanePoints[i].y*transM[1] + transM[2];
		double y = lanePoints[i].x*transM[3] + lanePoints[i].y*transM[4] + transM[5];
		double z = lanePoints[i].x*transM[6] + lanePoints[i].y*transM[7] + transM[8];
		x = x / z;
		y = y / z;
		projectedPoints.push_back(Point(x, y));
	}
}

Vec4f LaneDetection::mergeTwoLines(Vec4f line0, Vec4f line1, float p)
{
	int dy = frameSize.height;
	Point2f p00 = pointInLine(line0, dy / 2);
	Point2f p01 = pointInLine(line0, dy);
	Point2f p10 = pointInLine(line1, dy / 2);
	Point2f p11 = pointInLine(line1, dy);
	Point2f p0(p*p00.x + (1 - p)*p10.x, dy / 2);
	Point2f p1(p*p01.x + (1 - p)*p11.x, dy);
	float r = sqrt((p0.x - p1.x)*(p0.x - p1.x) + (p0.y - p1.y)*(p0.y - p1.y));
	float a = (p1.x - p0.x) / r;
	float b = (p1.y - p0.y) / r;
	if (a < 0)
	{
		a = -a;
		b = -b;
	}
	
	return Vec4f(a, b, p1.x, p1.y);
}

// 绘图显示
void LaneDetection::plotLanes(Mat& frame, Vec4f& leftLane, Vec4f& rightLane)
{
	//显示下半段
	float y1 = 0.3*frame.rows;
	float y2 = frame.rows;
	
	if (leftLane[0] != 0 && leftLane[1] != 0)
	{
		float xLeft1 = (y1 - leftLane[3]) / leftLane[1] * leftLane[0] + leftLane[2];
		float xLeft2 = (y2 - leftLane[3]) / leftLane[1] * leftLane[0] + leftLane[2];
		line(frame, Point(xLeft1, y1), Point(xLeft2, y2), Scalar(0, 0, 255), 2, 8, 0);
	}
	
	if (rightLane[0] != 0 && rightLane[1] != 0)
	{
		float xright1 = (y1 - rightLane[3]) / rightLane[1] * rightLane[0] + rightLane[2];
		float xright2 = (y2 - rightLane[3]) / rightLane[1] * rightLane[0] + rightLane[2];
		line(frame, Point(xright1, y1), Point(xright2, y2), Scalar(0, 255, 0), 2, 8, 0);
	}
}

void LaneDetection::plotLanes(Mat& frame, Vec4f& lane)
{
	//显示下半段
	float y1 = 0.3*frame.rows;
	float y2 = frame.rows;

	if (lane[0] != 0 && lane[1] != 0)
	{
		float x1 = (y1 - lane[3]) / lane[1] * lane[0] + lane[2];
		float x2 = (y2 - lane[3]) / lane[1] * lane[0] + lane[2];
		line(frame, Point(x1, y1), Point(x2, y2), Scalar(255, 0, 0), 2, 8, 0);
	}
}

void LaneDetection::plotSingleLine(Mat& frame, Vec4f& plotLine, double dL)
{
	Point p1, p2;
	p1.x = plotLine[2] - dL*plotLine[0];
	p1.y = plotLine[3] - dL*plotLine[1];
	p2.x = plotLine[2] + dL*plotLine[0];
	p2.y = plotLine[3] + dL*plotLine[1];
	line(frame, p1, p2, Scalar(255, 255, 255), 1);
}

// 换道偏离阶段
void LaneDetection::processMaskDepart(Mat& mask, vector<Point>& points)
{
	for (unsigned i = 0; i < points.size(); i++)
	{
		circle(mask, points[i], 15, 255, -1);
	}

	plotQuadMask(mask, lanesPV.MLane_p);
}

//void LaneDetection::getCandidateLinesDepart(Mat& frame, vector<Point>& ps, vector<Point>& fitPoints)
//{
//	//double t0;
//	//t0 = (double)cvGetTickCount();
//	
//	//计算灰度梯度
//	Mat frameGrey = frame(searchROI);
//	cvtColor(frameGrey, frameGrey, CV_RGB2GRAY);
//
//	//X方向
//	Mat maskX = (Mat_<char>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
//	filter2D(frameGrey, xFrame, CV_32F, maskX);
//	Mat xFrame2 = xFrame.mul(xFrame);
//
//	//Y方向
//	Mat maskY = (Mat_<char>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
//	filter2D(frameGrey, yFrame, CV_32F, maskY);
//	Mat yFrame2 = yFrame.mul(yFrame);
//
//	//梯度值
//	gFrame2 = xFrame2 + yFrame2;
//	sqrt(gFrame2, gFrame);
//
//	//方向值（正切值）
//	tanFrame = yFrame / xFrame;
//
//	//Canny图像预处理
//	Mat element(3, 3, CV_8U, Scalar(1));
//	dilate(edgeFrame, edge, element);
//
//	//显示梯度计算结果
//	Mat gradientL = Mat::zeros(frame.rows, frame.cols, CV_8U);
//	Mat gradientR = Mat::zeros(frame.rows, frame.cols, CV_8U);
//
//	//拟合点分组
//	vector<Point> fitPointsL;
//	vector<Point> fitPointsR;
//
//	Mat GL = gradientL(searchROI);
//	Mat GR = gradientR(searchROI);
//	Point move(searchROI.x, searchROI.y);
//
//	singleSideLinesDepart(fitLinesM, GL, GR,  fitPointsL, fitPointsR, move);
//
//	//cout << "MP: " << lanesPV.MLane_p << endl;
//	//cout << "ML: " << fitLinesM[0] << endl;
//	//cout << "MR: " << fitLinesM[1] << endl;
//
//	lanesPV.relia[0] = fitPointsL.size();
//	lanesPV.relia[1] = fitPointsR.size();
//
//	fitPoints.clear();
//	fitPoints.insert(fitPoints.end(), fitPointsL.begin(), fitPointsL.end());
//	fitPoints.insert(fitPoints.end(), fitPointsR.begin(), fitPointsR.end());
//
//	plotSingleLine(gradientL, fitLinesM[0], 500);
//	plotSingleLine(gradientR, fitLinesM[1], 500);
//
//	//t0 = (double)cvGetTickCount() - t0;
//	//t0 = t0 / ((double)cvGetTickFrequency()*1000.);
//	//cout << "时间消耗：FitLine Time = " << t0 << " ms" << endl;
//
//	imshow("GradientU", gradientL);
//	imshow("GradientD", gradientR);
//}
//
//void LaneDetection::singleSideLinesDepart(vector<Vec4f>& fitLines, Mat& GL, Mat& GR, vector<Point>& fitPointsL, vector<Point>& fitPointsR, Point move)
//{
//	fitPointsL.clear();
//	fitPointsR.clear();
//
//	//区域选取
//	Mat areaEdge = edge(searchROI);
//	Mat areaX = xFrame;
//	Mat areaG = gFrame;
//	Mat areaTan = tanFrame;
//
//	//统计方向梯度
//	float angleStep = CV_PI / 9;
//	float gHist[9] = { 0 };
//	int gNumHist[9] = { 0 };
//	Mat selectedPoints = Mat::zeros(areaTan.rows, areaTan.cols, CV_8U);
//	float* pTan;	// Angle
//	float* pX;		// X Gradient
//	float* pG;		// Gradient
//	uchar* pSP;		// Selected Points
//	uchar* pSPL;	// Selected Points Left
//	uchar* pSPR;	// Selected Points Right
//	uchar* pE;		// Edge frame
//	for (unsigned j = 0; j < areaTan.rows; j++)
//	{
//		pTan = areaTan.ptr<float>(j);
//		pG = areaG.ptr<float>(j);
//		pE = areaEdge.ptr<uchar>(j);
//		pSP = selectedPoints.ptr<uchar>(j);
//		for (unsigned i = 0; i < areaTan.cols; i++)
//		{
//			if (pE[i]>200)
//			{
//				pTan[i] = atan(pTan[i]) + CV_PI / 2;
//				int groupIndex = pTan[i] / angleStep;
//				gHist[groupIndex] += pG[i];
//				gNumHist[groupIndex] += 1;
//				pSP[i] = groupIndex;
//			}
//		}
//	}
//
//	//寻找梯度最大方向
//	float maxGradient = 0;
//	int maxIndex = 0;
//	for (unsigned i = 0; i < 9; i++)
//	{
//		if (gHist[i]>maxGradient)
//		{
//			maxGradient = gHist[i];
//			maxIndex = i;
//		}
//	}
//
//	//修正梯度结果图像
//	//float averGradient = maxGradient / float(gNumHist[maxIndex]);
//	if (maxGradient > 0)
//	{
//		for (unsigned j = 0; j < selectedPoints.rows; j++)
//		{
//			pX = areaX.ptr<float>(j);
//			pG = areaG.ptr<float>(j);
//			pSP = selectedPoints.ptr<uchar>(j);
//			pSPL = GL.ptr<uchar>(j);
//			pSPR = GR.ptr<uchar>(j);
//			for (unsigned i = 0; i < selectedPoints.cols; i++)
//			{
//				if (pSP[i] == maxIndex && pX[i] > 0)
//				{
//					pSP[i] = 255;
//					pSPL[i] = 255;
//					fitPointsL.push_back(Point(i, j) + move);
//				}
//				else if (pSP[i] == maxIndex && pX[i] < 0)
//				{
//					pSP[i] = 255;
//					pSPR[i] = 255;
//					fitPointsR.push_back(Point(i, j) + move);
//				}
//				else
//				{
//					pSP[i] = 0;
//				}
//			}
//		}
//	}
//
//	//计算拟合直线
//	Vec4f fitLinesL;
//	Vec4f fitLinesR;
//	Mat allBlack = Mat::zeros(GL.size(), GL.type());
//	if (fitPointsL.size() > 1000)
//	{
//		allBlack.copyTo(GL);
//		fitPointsL.clear();
//	}
//	if (fitPointsR.size() > 1000)
//	{
//		allBlack.copyTo(GR);
//		fitPointsR.clear();
//	}
//
//	if (fitPointsL.size() > 5)
//	{
//		//加权最小二乘法，越靠下的点权重越大
//		//fitLineWeighted(fitPointsL, fitLinesL);
//		fitLine(fitPointsL, fitLinesL, CV_DIST_L2, 0, 0.01, 0.01);
//	}
//	if (fitPointsR.size() > 5)
//	{
//		//加权最小二乘法，越靠下的点权重越大
//		//fitLineWeighted(fitPointsR, fitLinesR);
//		fitLine(fitPointsR, fitLinesR, CV_DIST_L2, 0, 0.01, 0.01);
//	}
//
//	fitLines.clear();
//	fitLines.push_back(fitLinesL);
//	fitLines.push_back(fitLinesR);
//}
//
//void LaneDetection::getTrackedLinesDepart(Mat& frame)
//{
//	//double t0;
//	//t0 = (double)cvGetTickCount();
//	
//	Vec4f MLane;
//
//	int nCols = frame.cols;
//	int nRows = frame.rows;
//
//	lanesPV.tracked[0] = false;
//
//	//检查结果之间的平行度
//	int midP = parallelTest(lanesPV.MLane_p, fitLinesM, frame.size(), lanesPV.confirmed[0]);
//
//	//输出初步结果
//	MLane = mergeLines(lanesPV.MLane_p, fitLinesM, midP, lanesPV.relia[0], lanesPV.relia[1]);
//
//	//cout << "Ptest:" << midP << endl;
//	//cout << "MLane: " << MLane << endl;
//
//	//更新tracked参数
//	if (MLane[0] != 0 && MLane[1] != 0)
//	{
//		lanesPV.tracked[0] = true;
//	}
//
//	//cout << "Tracked: " << lanesPV.tracked[0]  << endl;
//
//	//跟踪上一帧目标
//	if (lanesPV.MLane_p[0] != 0 && lanesPV.confirmed[0])
//	{
//		lanesPV.MLane = mergeTwoLines(MLane, lanesPV.MLane_p, 0.8);
//		//lanesPV.MLane = 0.8*MLane + 0.2*lanesPV.MLane_p;
//	}
//	else
//	{
//		lanesPV.MLane = MLane;
//	}
//
//	//cout << "MLane: " << lanesPV.MLane << endl;
//
//	//计算偏离趋势
//	calcDepartDirection(dDirection);
//
//	//更新跟踪参数
//	lanesPV.MLane_p = lanesPV.MLane;
//
//	if (midP < 3 || midP == 4)
//	{
//		lanesPV.confirmed[0] = true;
//	}
//	else
//	{
//		lanesPV.confirmed[0] = false;
//	}
//	//cout << "Confirmed: " << lanesPV.confirmed[0] << ", " << lanesPV.confirmed[1] << endl;
//
//	//鸟瞰视图
//	Mat tFrame = Mat::zeros(Size(300, 720), CV_8UC3);
//	vector<Point> projectedPoints;
//	getProjectedPoints(fitPoints, projectedPoints, transM);
//	for (unsigned i = 0; i < projectedPoints.size(); i++)
//	{
//		circle(tFrame, projectedPoints[i], 3, Scalar(255, 0, 0), (-1));
//	}
//	imshow("BV", tFrame);
//
//	//t0 = (double)cvGetTickCount() - t0;
//	//t0 = t0 / ((double)cvGetTickFrequency()*1000.);
//	//cout << "时间消耗：FitLine Time = " << t0 << " ms" << endl;
//
//}

