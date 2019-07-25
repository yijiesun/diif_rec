#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace std;
using namespace cv;
int pmain(void) {

	//����Ƶ�ļ�
	VideoCapture capture("test.avi");

	//isOpen�ж���Ƶ�Ƿ�򿪳ɹ�
	if (!capture.isOpened())
	{
		cout << "Movie open Error" << endl;
		return -1;
	}
	//��ȡ��Ƶ֡Ƶ
	double rate = capture.get(CV_CAP_PROP_FPS);
	cout << "֡��Ϊ:" << " " << rate << endl;
	cout << "��֡��Ϊ:" << " " << capture.get(CV_CAP_PROP_FRAME_COUNT) << endl;//���֡����
	Mat frame;
	namedWindow("Movie Player");

	double position = 0.0;
	//���ò��ŵ���һ֡����������Ϊ��0֡
	capture.set(CV_CAP_PROP_POS_FRAMES, position);
	while (1)
	{
		//��ȡ��Ƶ֡
		if (!capture.read(frame))
			break;

		imshow("Movie Player", frame);
		//��ȡ����ֵ
		char c = waitKey(33);
		if (c == 27)
			break;
	}
	capture.release();
	destroyWindow("Movie Player");
	return 0;
}