#pragma once

using namespace cv;
using namespace std;

struct LaneMarksBV		// Bird's view
{
	Vec2f laneDct;		// gradient
	Point LP;			// left lane position
	Point RP;			// right lane position
};

struct LaneMarksPV		// Perspective view
{
	// lane keeping
	Vec4f LLane;		// left lane
	Vec4f RLane;		// right lane
	Vec4f LLane_p;		// last left lane
	Vec4f RLane_p;		// last right lane
	Point vanish;		// vanish point	
	// lane changing
	Vec4f MLane;		// mid lane
	Vec4f MLane_p;		// last mid lane
	// index
	bool tracked[2];	// result of last-frame tracking
	bool confirmed[2];	// result of cur-frame fitting
	int relia[4];		// reliablity of line fitting
};

class LaneDetection
{
public:
	void initSys();
	void process(Mat& frame);
	void drawLanes(Mat& frame);

private:
	// Data Part
	CascadeClassifier cascade;		
	Mat OFrame;						// origin frame
	Mat SFrame;						// small frame
	
	Mat mask;
	Mat cannyFrame;
	Mat edgeFrame;
	Mat edge;

	Mat xFrame;
	Mat xFrame2;
	Mat yFrame;
	Mat yFrame2;
	Mat gFrame;
	Mat gFrame2;
	Mat tanFrame;
	Mat gradientU;
	Mat gradientD;
	Mat gradient;
	Mat selected;					// frame for line-fitting

	Size frameSize;					// frame size for processing
	Rect searchROI;					// searching area

	int departure;				    // departure condition. -1:left, 0:mid, 1:right
	deque<int> dDirection;		    // departure index
	
	double transM[9];				// projection matrix
	double retransM[9];				// reprojection matrix

	vector<Point> lanePoints;		// points of cascade result
	vector<Point> selectedPoints;   // points after deleting the wrong points
	vector<Point> fitPoints;		// points for line fitting

	Vec4f fitLinesL[2];				// left original targets
	Vec4f fitLinesR[2];				// right original targets
	Vec4f fitLinesM[2];				// mid original targets

	LaneMarksBV lanesBV;
	LaneMarksPV lanesPV;

	//
	// Method Part
	//
	// initSys()
	void initTransMatrix();
	
	// process()
	void frameInit();
	void getCandidatePoints();
	void getCannyArea();
	void getCandidateLines();
	void getTrackedLines();
	void nextFrame();

	// for getCannyArea()
	void processMask(Mat& mask, vector<Point>& points);		
	void plotQuadMask(Mat& mask, Vec4f line);
	void delErrorPoints(vector<Point>& ps0, vector<Point>& ps);
	
	// for getCandidateLines()
	void singleSideLines(Rect area, Vec4f (&fitLines)[2], vector<Point>& fitPointsU, vector<Point>& fitPointsD, Point move);
	int binSplitLaneArea();

	// for getTrackedLines()
	int parallelTest(Vec4f line0, Vec4f(&lines)[2], bool tracked);	// select the result according to the parallel test
	bool parallelIndex(Vec4f line0, Vec4f line1);	// calc the parallel index
	Vec4f mergeLines(Vec4f line0, Vec4f(&lines)[2], int pTest, int relia0, int relia1);
	Vec4f getOtherSideLane(Vec4f cLane, double deviation);
	void calcDepartDirection(deque<int>& dDirection);

	// for nextFrame();
	
	

	// other
	Point2f pointInLine(Vec4f cLine, int p, bool isY = true);		//求直线上的一点
	int dist2line(Point pt, Vec4f& lane);	//计算点到直线的距离
	Point LaneDetection::crossPoint(Vec4f line1, Vec4f line2);	//求交点
	void fitLineWeighted(vector<Point>& points, Vec4f& curLine);	// 加权最小二乘线性拟合
	Point2f getProjectedPoint(Point2f pointPV, double transM[]);	// 单个点的投影
	void getProjectedPoints(vector<Point>& lanePoints, vector<Point>& projectedPoints, double transM[]);	// 一组点的投影
	Vec4f mergeTwoLines(Vec4f line0, Vec4f line1, float p);

	// for drawLanes()
	void plotLanes(Mat& frame, Vec4f& leftLane, Vec4f& rightLane);	//双侧线
	void plotLanes(Mat& frame, Vec4f& lane);	//单侧线
	void plotSingleLine(Mat& frame, Vec4f& plotLine, double dL);

	// 换道偏离阶段
	void getCandidateLinesDepart(Mat& frame, vector<Point>& ps, vector<Point>& fitPoints);
	void processMaskDepart(Mat& mask, vector<Point>& points);
	void singleSideLinesDepart(vector<Vec4f>& fitLines, Mat& GU, Mat& GD, vector<Point>& fitPointsU, vector<Point>& fitPointsD, Point move);
	void getTrackedLinesDepart(Mat& frame);
	void delErrorPointsDepart(vector<Point>& ps0, vector<Point>& ps);
};

//Class Point for sorting
class PointSort : public Point
{
public:
	bool operator <(const PointSort& rhs) const
	{
		return y < rhs.y;
	}
	bool operator ==(const PointSort& rhs) const
	{
		return y == rhs.y;
	}
	bool operator >(const PointSort& rhs) const  
	{
		return y > rhs.y;
	}

	operator Point()
	{
		return Point(x, y);
	}

	PointSort(int x, int y) :Point(x, y)
	{
	}
};