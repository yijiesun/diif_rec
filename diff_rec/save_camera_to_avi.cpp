#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <conio.h>
#include <iostream>
using namespace std;
using namespace cv;


int qmain()
{
	VideoCapture cap(0);

	if (!cap.isOpened())
	{
		printf("open video failed!\n");
		return 1;
	}

	Mat Frame;

	//������Ƶ��·��
	string outputVideoPath = "..\\test.avi";
	//��ȡ��ǰ����ͷ����Ƶ��Ϣ
	cv::Size sWH = cv::Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	VideoWriter outputVideo;
	outputVideo.open(outputVideoPath, CV_FOURCC('M', 'P', '4', '2'), 25.0, sWH);

	bool begin = false;
	while (cap.isOpened())
	{
		cap >> Frame;
		imshow("img", Frame);

		int ch;
		if (_kbhit()) {
			ch = _getch();

			if (ch == 32)
			{
				if (begin)
					break;
				else
				{
					std::cout << "��ʼ¼��" << endl;
					begin = true;
				}

			}
		}

		if (begin)
			outputVideo << Frame;

		waitKey(10);

	}
	if (begin)
		outputVideo.release();

	return 0;
	//system("pause");
}