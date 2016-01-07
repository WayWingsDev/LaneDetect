#pragma once

using namespace cv;
using namespace std;

// Data structure
struct LaneMarksBV		//俯视视图
{
	Vec2f laneDct;		//车道线斜率
	Point LP;			//左线位置
	Point RP;			//右线位置
};

struct LaneMarksPV		//透视视图
{
	//正常行驶中
	Vec4f LLane;		//左线
	Vec4f RLane;		//右线
	Vec4f LLane_p;		//左线 上一帧
	Vec4f RLane_p;		//右线 上一帧	
	Point vanish;		//车道线消失点
	bool tracked[2];	//本帧是否找到合适的车道线参数
	bool confirmed[2];	//上一帧的跟踪结果是否可信
	int reliablity[4];	//直线拟合的可信度
	//换道中
	Vec4f MLane;		//中线
	Vec4f MLane_p;		//中线 上一帧
};

// LaneDetection类
class LaneDetection
{
public:
	void initSys();
	void process(Mat& frame);
	void drawLanes(Mat& frame);
	void nextFrame();				//下帧预测

private:
	///////////////////////////
	// Data
	///////////////////////////
	CascadeClassifier cascade;		//分类器
	Mat OFrame;						//原始图像
	Mat SFrame;						//处理的缩小图像
	Mat edgeFrame;					//边缘图像
	Mat xFrame;
	Mat yFrame;
	Mat gFrame2;
	Mat gFrame;
	Mat tanFrame;
	Mat edge;

	Size frameSize;

	int departuring;				//进入偏离状态	-1:左偏，0：居中，1：右偏

	double transM[9];				//投影转换矩阵
	double retransM[9];

	vector<Point> lanePoints;		//分类器初选目标点
	vector<Point> fitPoints;		//用于拟合的目标点

	Rect searchROI;					//搜索区域

	vector<Vec4f> fitLinesL;		//左侧初选目标
	vector<Vec4f> fitLinesR;		//右侧初选目标
	vector<Vec4f> fitLinesM;		//中间初选目标

	deque<int> dDirection;		//偏离趋势指示

	LaneMarksBV lanesBV;
	LaneMarksPV lanesPV;

	///////////////////////////
	// method
	///////////////////////////
	// 初始化
	void initTransMatrix();

	// 换道状态初始化

	// 图像准备
	Rect frameInit(Mat& inFrame, Mat& outFrame);

	// 分类器搜索
	void getCandidatePoints(Mat& frame, vector<Point>& ps);

	// 边缘检测
	void getCannyArea(Mat& frame, Mat& eFrame, Rect searchROI, vector<Point>& ps);
	void processMask(Mat& mask, vector<Point>& points);		
	void plotQuadMask(Mat& mask, Vec4f line);
	void delErrorPoints(vector<Point>& ps0, vector<Point>& ps);
	
	// 车道初选
	void getCandidateLines(Mat& frame, vector<Point>& ps, vector<Point>& fitPoints);
	void singleSideLines(Rect area, vector<Vec4f>& fitLines, Mat& GU, Mat& GD, vector<Point>& fitPointsU, vector<Point>& fitPointsD, Point move);	//单侧车道线的计算
	int binSplitLaneArea(Mat& frame, vector<Point>& ps);	// 将图像和原始点集划分为左右两部分

	// 跟踪修正
	void getTrackedLines(Mat& frame);	
	int parallelTest(Vec4f line0, vector<Vec4f>& lines, Size frameSize, bool confirmed);	// 根据当前帧与上一帧结果之间的平行度进行筛选
	bool parallelIndex(Vec4f line0, Vec4f line1, Size frameSize);	// 计算两根直线之间的平行度
	Vec4f mergeLines(Vec4f line0, vector<Vec4f>& lines, int pTest, int relia0, int relia1);	// 根据当前帧与上一帧结果之间的平行度进行筛选
	Vec4f getOtherSideLane(Vec4f cLane, double deviation, Size frameSize);	// 补充单边缺失的情况
	void calcDepartDirection(deque<int>& dDirection);

	// 其他
	Point2f pointInLine(Vec4f cLine, int p, bool isY = true);		//求直线上的一点
	int dist2line(Point pt, Vec4f& lane);	//计算点到直线的距离
	Point LaneDetection::crossPoint(Vec4f line1, Vec4f line2);	//求交点
	void fitLineWeighted(vector<Point>& points, Vec4f& curLine);	// 加权最小二乘线性拟合
	Point2f getProjectedPoint(Point2f pointPV, double transM[]);	// 单个点的投影
	void getProjectedPoints(vector<Point>& lanePoints, vector<Point>& projectedPoints, double transM[]);	// 一组点的投影
	Vec4f mergeTwoLines(Vec4f line0, Vec4f line1, float p);

	// 绘图显示
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

//可用于排序的Point类
class PointSort : public Point
{
public:
	bool operator <(const PointSort& rhs) const // 升序排序时必须写的函数  
	{
		return y < rhs.y;
	}
	bool operator ==(const PointSort& rhs) const
	{
		return y == rhs.y;
	}
	bool operator >(const PointSort& rhs) const // 降序排序时必须写的函数  
	{
		return y > rhs.y;
	}

	operator Point()	//PointSort转化为Point
	{
		return Point(x, y);
	}

	PointSort(int x, int y) :Point(x, y)
	{
	}
};