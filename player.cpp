#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace std;
using namespace cv;
int pmain(void) {

	//打开视频文件
	VideoCapture capture("test.avi");

	//isOpen判断视频是否打开成功
	if (!capture.isOpened())
	{
		cout << "Movie open Error" << endl;
		return -1;
	}
	//获取视频帧频
	double rate = capture.get(CV_CAP_PROP_FPS);
	cout << "帧率为:" << " " << rate << endl;
	cout << "总帧数为:" << " " << capture.get(CV_CAP_PROP_FRAME_COUNT) << endl;//输出帧总数
	Mat frame;
	namedWindow("Movie Player");

	double position = 0.0;
	//设置播放到哪一帧，这里设置为第0帧
	capture.set(CV_CAP_PROP_POS_FRAMES, position);
	while (1)
	{
		//读取视频帧
		if (!capture.read(frame))
			break;

		imshow("Movie Player", frame);
		//获取按键值
		char c = waitKey(33);
		if (c == 27)
			break;
	}
	capture.release();
	destroyWindow("Movie Player");
	return 0;
}