#if 1
#include <conio.h>
#include <iostream>
#include <sstream>
#include <string>
#include<opencv2/opencv.hpp>
#include<opencv2/video/background_segm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include<vector>
#include<stdexcept>
#define IPUT_AVI 1
#define EXTEND_PIXEL 5
#define SMALL_CONTOR_SIZE_NEED_CLEAR 500
#define OVER_PERCENT 0.0001f
#define INSIDE_DILATE_WIN_SIZE 50
#define IDWS_SCALE 1
#define NORMAL_DIFF 0.1f
using namespace cv;
using namespace std;
float DecideOverlap(const Rect &r1, const Rect &r2, Rect &r3);
int buildAndClearSmallContors(vector<vector<Point>> &rect, vector<Rect> &rects, int size);
void mergeRecs(vector<Rect> &rects, float percent);
void paddingRecs(vector<Rect> &rects, int size);
void insideDilate(Mat & bimg, Mat & bout, int win_size,int scale);
void normalDiff(vector<Rect> &rects, Mat & bk, Mat & src);
void drawRecs(Mat & img, vector<Rect> &rects, const Scalar& color, string &txt);
bool keyEvent(Mat & img,  Mat & diff, vector<vector<Point>> &cont, vector<Vec4i> &h);
int IMG_WID, IMG_HGT;

int main(int argc, char* argv[])
{
#if IPUT_AVI
	//打开视频文件
	VideoCapture capture("test.avi");
	//isOpen判断视频是否打开成功
	if (!capture.isOpened())
	{
		cout << "Movie open Error" << endl;
		return -1;
	}
	double position = 70;
	//设置播放到哪一帧，这里设置为第0帧
	capture.set(CV_CAP_PROP_POS_FRAMES, position);
#else
	//读取视频或摄像头
	VideoCapture capture(0);
	if (!capture.isOpened()) { //判断能够打开摄像头
		cout << "can not open the camera" << endl;
		cin.get();
		exit(1);
	}
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
#endif
	cv::namedWindow("image");
	cv::Mat src, src_gray, cv_image_bgr, bk, bk_gray;
	cv::Mat diff1,diff2,diff3;

	Mat resultImg = Mat(src.rows, src.cols, CV_8UC3);
	Mat combine = Mat(src.rows, 2*src.cols, CV_8UC3);
	
	Mat pDes, kernel, kernel1, combine1;

	capture >> bk;

	cv::cvtColor(bk, bk_gray, CV_BGR2GRAY);
	//equalizeHist(bk_gray, bk_gray);

	IMG_WID = bk_gray.size().width;
	IMG_HGT = bk_gray.size().height;

	string txt1 = "small";
	string txt2 = "combine";
	string txt3 = "needclear";
	while (true)
	{
		cout << position++ << endl;
		cv::Mat diff(cv::Size(IMG_WID, IMG_HGT), CV_8UC1);
		capture >> src;

		cvtColor(src, src_gray, CV_BGR2GRAY);

		cvtColor(src_gray, resultImg, CV_GRAY2BGR);

		kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
		
		absdiff(src_gray, bk_gray, diff);

		threshold(diff, diff, 50, 255, CV_THRESH_BINARY);
		blur(diff, diff, Size(15, 15), Point(-1, -1));

		kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
		
		morphologyEx(diff, diff, CV_MOP_ERODE, kernel);//腐蚀
		
		//kernel1 = getStructuringElement(MORPH_RECT, Size(5, 5));
		//morphologyEx(diff, diff, MORPH_DILATE, kernel1); //膨胀

		threshold(diff, diff, 50, 255, CV_THRESH_BINARY);

		Mat diff_inside = diff.clone();
		diff2 = diff.clone();
		insideDilate(diff, diff_inside,INSIDE_DILATE_WIN_SIZE,IDWS_SCALE);
		diff1 = diff_inside.clone();

		vector<vector<Point>> contours;
		vector<Vec4i> hierarcy;
		findContours(diff_inside, contours, hierarcy, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE); //查找轮廓
		vector<Rect> boundRect; //定义外接矩形集合

		int rec_nums = buildAndClearSmallContors(contours, boundRect, SMALL_CONTOR_SIZE_NEED_CLEAR);
		//paddingRecs(boundRect, EXTEND_PIXEL);
		drawRecs(resultImg, boundRect, Scalar(0, 255, 0, 0),txt1);

		mergeRecs(boundRect,OVER_PERCENT);
		mergeRecs(boundRect, OVER_PERCENT);
		drawRecs(resultImg, boundRect, Scalar(255, 0, 0, 0), txt3);
		normalDiff(boundRect, bk_gray, src_gray);
		drawRecs(resultImg, boundRect, Scalar(0, 0, 255, 0), txt2);

		Mat diff2c = Mat(resultImg.rows, resultImg.cols, CV_8UC3);
		cvtColor(diff1, diff2c, CV_GRAY2BGR);
		hconcat(resultImg, diff2c, combine);

		imshow("image", combine);
		//
		//bool key = keyEvent(src_gray, diff1, contours, hierarcy);
		//if(key)
		//	return 0;

		waitKey(30);
	}
	capture.release();
	return 0;
}

void mergeRecs(vector<Rect> &rects, float percent)
{
	int len = rects.size();
	int new_len = 0;
	int ptr = 0;
	Rect tmp;

	for (;;)
	{
		if (ptr >= len)
			break;
		
		for (int i = 0; i < len; i++)
		{
			if (ptr < 0 || ptr >= rects.size() || i < 0 || i >= rects.size())
				break;
			if (i == ptr)
				continue;
			if (DecideOverlap(rects[ptr], rects[i], tmp) >= percent)
			{
				rects[ptr] = tmp;
				if (rects.begin() + i <= rects.end())
				{
					rects.erase(rects.begin() + i);
					i--;
				}
				else
					break;
			}

		}
		ptr++;
		len = rects.size();
	}

}

int buildAndClearSmallContors(vector<vector<Point>> &contours, vector<Rect> &rects, int size)
{
	int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
	int  cnt = 0;
	for (int i = 0; i<contours.size(); i++)
	{
		Rect rect_tmp = boundingRect((Mat)contours[i]);
		x0 = rect_tmp.x;
		y0 = rect_tmp.y;
		w0 = rect_tmp.width;
		h0 = rect_tmp.height;
		if (w0 * h0 >= size)
		{
			rects.push_back(rect_tmp);
			cnt++;
		}
	}
	return cnt;
}

void paddingRecs(vector<Rect> &rects, int size)
{
	for (int i = 0; i<rects.size(); i++)
	{
		rects[i].x = min(max(rects[i].x - EXTEND_PIXEL, 0), IMG_WID);
		rects[i].y = min(max(rects[i].y - EXTEND_PIXEL, 0), IMG_HGT);
		rects[i].width = rects[i].x + rects[i].width + EXTEND_PIXEL > IMG_WID ? IMG_WID - rects[i].x : rects[i].width + EXTEND_PIXEL;
		rects[i].height = rects[i].y + rects[i].height + EXTEND_PIXEL > IMG_HGT ? IMG_HGT - rects[i].y : rects[i].height + EXTEND_PIXEL;
	}
}

float DecideOverlap(const Rect &r1, const Rect &r2, Rect &r3)
{
	//r3 = r1;
	int x1 = r1.x;
	int y1 = r1.y;
	int width1 = r1.width;
	int height1 = r1.height;

	int x2 = r2.x;
	int y2 = r2.y;
	int width2 = r2.width;
	int height2 = r2.height;

	int endx = max(x1 + width1, x2 + width2);
	int startx = min(x1, x2);
	int width = width1 + width2 - (endx - startx);

	int endy = max(y1 + height1, y2 + height2);
	int starty = min(y1, y2);
	int height = height1 + height2 - (endy - starty);

	float ratio = 0.0f;
	float Area, Area1, Area2;

	if (width <= 0 || height <= 0)
		return 0.0f;
	else
	{
		Area = width*height;
		Area1 = width1*height1;
		Area2 = width2*height2;
		ratio = max(Area/ (float)Area1, Area/ (float)Area2);
		r3.x = startx;
		r3.y = starty;
		r3.width = endx - startx;
		r3.height = endy - starty;
	}

	return ratio;
}

/*insideDilate:内膨胀函数
*像素点上下或者左右两边同时有白像素点夹逼时向内膨胀
*bimg:二值图像
*win_size:搜索窗口
*scale:夹逼有效像素数量
*/
void insideDilate(Mat & bimg, Mat & bout, int win_size, int scale)
{
	for (int w = 0 + win_size; w < IMG_WID - win_size; w++)
	{
		for (int h = 0 + win_size; h < IMG_HGT - win_size; h++)
		{
			Point curr(w,h);
			Point refer;
			Mat tmp;
			int l=0, r=0, u=0, d=0;
			Mat roi_u(bimg, Rect(w, h - win_size, 1, win_size));
			Mat roi_d(bimg, Rect(w , h, 1, win_size));
			Mat roi_l(bimg, Rect(w - win_size, h, win_size, 1));
			Mat roi_r(bimg, Rect(w, h, win_size, 1));
			u = countNonZero(roi_u);
			d = countNonZero(roi_d);
			l = countNonZero(roi_l);
			r = countNonZero(roi_r);
			uchar* data = bout.ptr<uchar>(h);

			if ((u >= scale && d >= scale) || (l >= scale && r >= scale))
				data[w] = 255;
	/*		else
				data[w] = 0;*/
		}
	}
}

void normalDiff(vector<Rect> &rects, Mat & bk, Mat & src)
{
	int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
	for (int i = 0; i< rects.size(); i++)
	{
		x0 = rects[i].x;  //获得第i个外接矩形的左上角的x坐标
		y0 = rects[i].y; //获得第i个外接矩形的左上角的y坐标
		w0 = rects[i].width; //获得第i个外接矩形的宽度
		h0 = rects[i].height; //获得第i个外接矩形的高度
		if (w0*h0 > SMALL_CONTOR_SIZE_NEED_CLEAR * 2)
			continue;
		Mat bk_roi(bk, Rect(x0, y0, w0, h0));
		Mat src_roi(src, Rect(x0, y0, w0, h0));
		Mat equal_src,equal_bk;
		Mat dif_roi;
		Mat nor_roi;
		equalizeHist(bk_roi, equal_bk);
		equalizeHist(src_roi, equal_src);
		absdiff(equal_src, equal_bk, dif_roi);
		normalize(dif_roi, nor_roi, 0, 1, cv::NORM_MINMAX);
		double sum_diff_roi = countNonZero(nor_roi);
		if (sum_diff_roi > w0 * h0 * NORMAL_DIFF)
		{
			rects.erase(rects.begin() + i);
			i--;
		}
	}
}

void drawRecs(Mat & img, vector<Rect> &rects, const Scalar& color,string &txt)
{
	int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
	for (int i = 0; i< rects.size(); i++)
	{
		x0 = rects[i].x;  //获得第i个外接矩形的左上角的x坐标
		y0 = rects[i].y; //获得第i个外接矩形的左上角的y坐标
		w0 = rects[i].width; //获得第i个外接矩形的宽度
		h0 = rects[i].height; //获得第i个外接矩形的高度

		rectangle(img, Point(x0, y0), Point(x0 + w0, y0 + h0), color, 2, 8); //绘制第i个外接矩形
		std::stringstream ss;
		ss << txt ;
		std::string s = ss.str();
		putText(img, s.c_str(), Point(x0, y0), FONT_HERSHEY_PLAIN, 1.0, color, 1);
	
	}
}

bool keyEvent(Mat & img, Mat & diff, vector<vector<Point>> &cont, vector<Vec4i> &h)
{

	int ch;
	if (_kbhit()) {//如果有按键按下，则_kbhit()函数返回真
		ch = _getch();//使用_getch()函数获取按下的键值
		cout << "save img!" << endl;
		if (ch == 32)
		{

			Mat dstImage = Mat::zeros(img.rows, img.cols, CV_8UC3);
			int index = 0;
			for (; index >= 0; index = h[index][0]) {
				Scalar color(rand() & 255, rand() & 255, rand() & 255);
				drawContours(dstImage, cont, index, color, CV_FILLED, 8, h);
			}

			vector<Rect> boundRectTmp;
			int rec_nums = buildAndClearSmallContors(cont, boundRectTmp, SMALL_CONTOR_SIZE_NEED_CLEAR);
			//paddingRecs(boundRectTmp, EXTEND_PIXEL);
			Mat dstImage1 = dstImage.clone();
			int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
			int  cnt = 0;
			for (int i = 0; i < boundRectTmp.size(); i++)
			{
				x0 = boundRectTmp[i].x;  //获得第i个外接矩形的左上角的x坐标
				y0 = boundRectTmp[i].y; //获得第i个外接矩形的左上角的y坐标
				w0 = boundRectTmp[i].width; //获得第i个外接矩形的宽度
				h0 = boundRectTmp[i].height; //获得第i个外接矩形的高度
				rectangle(dstImage1, Point(x0, y0), Point(x0 + w0, y0 + h0), Scalar(0, 255, 0), 1, 8); //绘制第i个外接矩形
				std::stringstream ss;
				ss << i << ":" << x0 << "," << y0 << "-" << w0 << "," << h0;
				std::string s = ss.str();
				putText(dstImage1, s.c_str(), Point(x0, y0), FONT_HERSHEY_PLAIN, 1.0, Scalar(255, 0, 0), 1);
			}


			drawContours(dstImage, cont, -1, Scalar(0, 0, 255), 1, 8);
			imwrite("diff.png", diff);
			imwrite("dst.png", img);
			imwrite("contours.png", dstImage);
			imwrite("allrec.png", dstImage1);
		}
		else if (ch == 27)
			return 1;
		return 0;
	}

}
#endif