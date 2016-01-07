#pragma once

using namespace cv;
using namespace std;

// Data structure
struct LaneMarksBV		//������ͼ
{
	Vec2f laneDct;		//������б��
	Point LP;			//����λ��
	Point RP;			//����λ��
};

struct LaneMarksPV		//͸����ͼ
{
	//������ʻ��
	Vec4f LLane;		//����
	Vec4f RLane;		//����
	Vec4f LLane_p;		//���� ��һ֡
	Vec4f RLane_p;		//���� ��һ֡	
	Point vanish;		//��������ʧ��
	bool tracked[2];	//��֡�Ƿ��ҵ����ʵĳ����߲���
	bool confirmed[2];	//��һ֡�ĸ��ٽ���Ƿ����
	int reliablity[4];	//ֱ����ϵĿ��Ŷ�
	//������
	Vec4f MLane;		//����
	Vec4f MLane_p;		//���� ��һ֡
};

// LaneDetection��
class LaneDetection
{
public:
	void initSys();
	void process(Mat& frame);
	void drawLanes(Mat& frame);
	void nextFrame();				//��֡Ԥ��

private:
	///////////////////////////
	// Data
	///////////////////////////
	CascadeClassifier cascade;		//������
	Mat OFrame;						//ԭʼͼ��
	Mat SFrame;						//�������Сͼ��
	Mat edgeFrame;					//��Եͼ��
	Mat xFrame;
	Mat yFrame;
	Mat gFrame2;
	Mat gFrame;
	Mat tanFrame;
	Mat edge;

	Size frameSize;

	int departuring;				//����ƫ��״̬	-1:��ƫ��0�����У�1����ƫ

	double transM[9];				//ͶӰת������
	double retransM[9];

	vector<Point> lanePoints;		//��������ѡĿ���
	vector<Point> fitPoints;		//������ϵ�Ŀ���

	Rect searchROI;					//��������

	vector<Vec4f> fitLinesL;		//����ѡĿ��
	vector<Vec4f> fitLinesR;		//�Ҳ��ѡĿ��
	vector<Vec4f> fitLinesM;		//�м��ѡĿ��

	deque<int> dDirection;		//ƫ������ָʾ

	LaneMarksBV lanesBV;
	LaneMarksPV lanesPV;

	///////////////////////////
	// method
	///////////////////////////
	// ��ʼ��
	void initTransMatrix();

	// ����״̬��ʼ��

	// ͼ��׼��
	Rect frameInit(Mat& inFrame, Mat& outFrame);

	// ����������
	void getCandidatePoints(Mat& frame, vector<Point>& ps);

	// ��Ե���
	void getCannyArea(Mat& frame, Mat& eFrame, Rect searchROI, vector<Point>& ps);
	void processMask(Mat& mask, vector<Point>& points);		
	void plotQuadMask(Mat& mask, Vec4f line);
	void delErrorPoints(vector<Point>& ps0, vector<Point>& ps);
	
	// ������ѡ
	void getCandidateLines(Mat& frame, vector<Point>& ps, vector<Point>& fitPoints);
	void singleSideLines(Rect area, vector<Vec4f>& fitLines, Mat& GU, Mat& GD, vector<Point>& fitPointsU, vector<Point>& fitPointsD, Point move);	//���೵���ߵļ���
	int binSplitLaneArea(Mat& frame, vector<Point>& ps);	// ��ͼ���ԭʼ�㼯����Ϊ����������

	// ��������
	void getTrackedLines(Mat& frame);	
	int parallelTest(Vec4f line0, vector<Vec4f>& lines, Size frameSize, bool confirmed);	// ���ݵ�ǰ֡����һ֡���֮���ƽ�жȽ���ɸѡ
	bool parallelIndex(Vec4f line0, Vec4f line1, Size frameSize);	// ��������ֱ��֮���ƽ�ж�
	Vec4f mergeLines(Vec4f line0, vector<Vec4f>& lines, int pTest, int relia0, int relia1);	// ���ݵ�ǰ֡����һ֡���֮���ƽ�жȽ���ɸѡ
	Vec4f getOtherSideLane(Vec4f cLane, double deviation, Size frameSize);	// ���䵥��ȱʧ�����
	void calcDepartDirection(deque<int>& dDirection);

	// ����
	Point2f pointInLine(Vec4f cLine, int p, bool isY = true);		//��ֱ���ϵ�һ��
	int dist2line(Point pt, Vec4f& lane);	//����㵽ֱ�ߵľ���
	Point LaneDetection::crossPoint(Vec4f line1, Vec4f line2);	//�󽻵�
	void fitLineWeighted(vector<Point>& points, Vec4f& curLine);	// ��Ȩ��С�����������
	Point2f getProjectedPoint(Point2f pointPV, double transM[]);	// �������ͶӰ
	void getProjectedPoints(vector<Point>& lanePoints, vector<Point>& projectedPoints, double transM[]);	// һ����ͶӰ
	Vec4f mergeTwoLines(Vec4f line0, Vec4f line1, float p);

	// ��ͼ��ʾ
	void plotLanes(Mat& frame, Vec4f& leftLane, Vec4f& rightLane);	//˫����
	void plotLanes(Mat& frame, Vec4f& lane);	//������
	void plotSingleLine(Mat& frame, Vec4f& plotLine, double dL);

	// ����ƫ��׶�
	void getCandidateLinesDepart(Mat& frame, vector<Point>& ps, vector<Point>& fitPoints);
	void processMaskDepart(Mat& mask, vector<Point>& points);
	void singleSideLinesDepart(vector<Vec4f>& fitLines, Mat& GU, Mat& GD, vector<Point>& fitPointsU, vector<Point>& fitPointsD, Point move);
	void getTrackedLinesDepart(Mat& frame);
	void delErrorPointsDepart(vector<Point>& ps0, vector<Point>& ps);
};

//�����������Point��
class PointSort : public Point
{
public:
	bool operator <(const PointSort& rhs) const // ��������ʱ����д�ĺ���  
	{
		return y < rhs.y;
	}
	bool operator ==(const PointSort& rhs) const
	{
		return y == rhs.y;
	}
	bool operator >(const PointSort& rhs) const // ��������ʱ����д�ĺ���  
	{
		return y > rhs.y;
	}

	operator Point()	//PointSortת��ΪPoint
	{
		return Point(x, y);
	}

	PointSort(int x, int y) :Point(x, y)
	{
	}
};