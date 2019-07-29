#if 1
#include <fstream>
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
#define SAVE_AVI 1
#define EXTEND_PIXEL 20
#define SMALL_CONTOR_SIZE_NEED_CLEAR 200
#define OVER_PERCENT 0.0001f
#define INSIDE_DILATE_WIN_SIZE 10
#define IDWS_SCALE 1
#define NORMAL_DIFF 0.1f
#define CLEAR_BIG_THAN_THIS_HAMMING_RADIO 0.8f  //匹配矩形进行汉明距离计算，距离大于BK的该倍视为误匹配
using namespace cv;
using namespace std;
float DecideOverlap(const Rect &r1, const Rect &r2, Rect &r3);
int buildAndClearSmallContors(vector<vector<Point>> &rect, vector<Rect> &rects, int size);
void mergeRecs(vector<Rect> &rects, float percent);
void paddingRecs(vector<Rect> &rects, int size);
void insideDilate(Mat & bimg, Mat & bout, int win_size, int scale);
void normalDiff(vector<Rect> &rects, Mat & bk, Mat & src);
void hammingClear(vector<Rect> &rects, Mat & bk, Mat & src);
void hammingBiImgClear(vector<Rect> &rects, Mat & bk, Mat & src, Mat & diff);
void drawRecs(Mat & img, vector<Rect> &rects, const Scalar& color, string &txt);
bool keyEvent(Mat & img, Mat & diff, vector<vector<Point>> &cont, vector<Vec4i> &h);
void writeMatToFile(cv::Mat& m, const char* filename);
void svm_hog_detector(vector<Rect> &rects, Mat & src);
void denoise(Mat &src, Mat &dst);
int dealMat(Mat &img, Rect &r1);
int IMG_WID, IMG_HGT;
Mat resultImg;
double position;
Mat aa;
int main(int argc, char* argv[])
{
#if IPUT_AVI
	//打开视频文件
	VideoCapture capture("test1.avi");
	//isOpen判断视频是否打开成功
	if (!capture.isOpened())
	{
		cout << "Movie open Error" << endl;
		return -1;
	}
	position = 115;
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
	cv::Mat diff1, diff2, diff3;
	resultImg = Mat(src.rows, src.cols, CV_8UC3);
	Mat combine = Mat(src.rows, 2 * src.cols, CV_8UC3);

	Mat pDes, kernel, kernel1, combine1;

	capture >> bk;

	cv::cvtColor(bk, bk_gray, CV_BGR2GRAY);
	//equalizeHist(bk_gray, bk_gray);
	imwrite("bk.png", bk_gray);
	IMG_WID = bk_gray.size().width;
	IMG_HGT = bk_gray.size().height;

	string txt1 = "small";
	string txt2 = "combine";
	string txt3 = "ROI";
#if SAVE_AVI
	string outputVideoPath = "..\\result_temp.avi";
	//Size sWH = Size(2*IMG_WID, IMG_HGT);
	Size sWH = Size(IMG_WID, IMG_HGT);
	VideoWriter outputVideo;
	outputVideo.open(outputVideoPath, CV_FOURCC('M', 'P', '4', '2'), 25.0, sWH);
#endif

	position = 0;//555 649
				 //设置播放到哪一帧，这里设置为第0帧
	capture.set(CV_CAP_PROP_POS_FRAMES, position);
	while (true)
	{
		cout << position++ << endl;
		cv::Mat diff(cv::Size(IMG_WID, IMG_HGT), CV_8UC1);
		capture >> src;
		//imwrite("src.png", src);
		cvtColor(src, src_gray, CV_BGR2GRAY);

		cvtColor(src_gray, resultImg, CV_GRAY2BGR);

		kernel = getStructuringElement(MORPH_RECT, Size(5, 5));

		absdiff(src_gray, bk_gray, diff);

		threshold(diff, diff, 50, 255, CV_THRESH_BINARY);
		denoise(diff, diff);
		threshold(diff, diff, 50, 255, CV_THRESH_BINARY);

		diff2 = diff.clone();

		vector<vector<Point>> contours;
		vector<Vec4i> hierarcy;
		findContours(diff2, contours, hierarcy, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE); //查找轮廓
		vector<Rect> boundRect; //定义外接矩形集合

		int rec_nums = buildAndClearSmallContors(contours, boundRect, SMALL_CONTOR_SIZE_NEED_CLEAR);

		//drawRecs(resultImg, boundRect, Scalar(0, 255, 0, 0),txt1);
		hammingBiImgClear(boundRect, bk_gray, src_gray, diff);
		paddingRecs(boundRect, EXTEND_PIXEL);
		mergeRecs(boundRect, OVER_PERCENT);
		mergeRecs(boundRect, OVER_PERCENT);

		for (int i = 0; i < boundRect.size(); i++)
		{
			Mat imageROI = src_gray(Rect(boundRect[i].x, boundRect[i].y, boundRect[i].width, boundRect[i].height));
			Mat postimageROI;
			resize(imageROI, postimageROI, Size(100, 200), (0, 0), (0, 0), 3);
	
			dealMat(postimageROI, boundRect[i]);
			
		}

		drawRecs(resultImg, boundRect, Scalar(0, 255, 0, 0), txt3);

		//svm_hog_detector(boundRect, src_gray);
		imshow("image", diff);

		//normalDiff(boundRect, bk_gray, src_gray);
		//drawRecs(resultImg, boundRect, Scalar(0, 0, 255, 0), txt2);
#if 0
		Mat diff2c = Mat(resultImg.rows, resultImg.cols, CV_8UC3);
		cvtColor(diff1, diff2c, CV_GRAY2BGR);
		hconcat(resultImg, diff2c, combine);

		imshow("image", combine);
#endif
#if SAVE_AVI
		//outputVideo << combine;
		outputVideo << resultImg;
#endif

		bool key = keyEvent(src_gray, diff1, contours, hierarcy);
		if (key)
			break;

		waitKey(30);
	}
	capture.release();
#if SAVE_AVI
	outputVideo.release();
#endif
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
		rects[i].width = rects[i].x + rects[i].width + 2 * EXTEND_PIXEL > IMG_WID ? IMG_WID - rects[i].x : rects[i].width + 2 * EXTEND_PIXEL;
		rects[i].height = rects[i].y + rects[i].height + 2 * EXTEND_PIXEL > IMG_HGT ? IMG_HGT - rects[i].y : rects[i].height + 2 * EXTEND_PIXEL;
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
		ratio = max(Area / (float)Area1, Area / (float)Area2);
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
			Point curr(w, h);
			Point refer;
			Mat tmp;
			int l = 0, r = 0, u = 0, d = 0;
			Mat roi_u(bimg, Rect(w, h - win_size, 1, win_size));
			Mat roi_d(bimg, Rect(w, h, 1, win_size));
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
		if (i < 0 || i >= rects.size())
			break;

		x0 = rects[i].x;  //获得第i个外接矩形的左上角的x坐标
		y0 = rects[i].y; //获得第i个外接矩形的左上角的y坐标
		w0 = rects[i].width; //获得第i个外接矩形的宽度
		h0 = rects[i].height; //获得第i个外接矩形的高度
							  //if (w0*h0 > SMALL_CONTOR_SIZE_NEED_CLEAR * 2)
							  //	continue;
		Mat bk_roi(bk, Rect(x0, y0, w0, h0));
		Mat src_roi(src, Rect(x0, y0, w0, h0));
		Mat equal_src, equal_bk;
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
void hammingClear(vector<Rect> &rects, Mat & bk, Mat & src)
{
	int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
	for (int i = 0; i< rects.size(); i++)
	{
		if (i < 0 || i >= rects.size())
			break;
		x0 = rects[i].x;  //获得第i个外接矩形的左上角的x坐标
		y0 = rects[i].y; //获得第i个外接矩形的左上角的y坐标
		w0 = rects[i].width; //获得第i个外接矩形的宽度
		h0 = rects[i].height; //获得第i个外接矩形的高度
							  //if (w0*h0 > SMALL_CONTOR_SIZE_NEED_CLEAR * 2)
							  //	continue;
		cv::Mat bk_roi = bk(cv::Rect(x0, y0, w0, h0));
		cv::Mat src_roi = src(cv::Rect(x0, y0, w0, h0));

		Mat bk_comp, src_comp;

		//imwrite("roi.bmp", bk_roi);
		//bk_roi = imread("roi.bmp",0);
		//imwrite("roi.bmp", src_roi);
		//src_roi = imread("roi.bmp", 0);

		Scalar bk_sum = sum(bk_roi);
		int bk_avg = cvRound(bk_sum.val[0] / (double)(w0*h0));
		Mat bk_mask = Mat::ones(Size(w0, h0), CV_8UC1);
		bk_mask *= bk_avg;
		compare(bk_roi, bk_mask, bk_comp, CMP_GE);
		//bk_comp /= 255;

		Scalar src_sum = sum(src_roi);
		int src_avg = cvRound(src_sum.val[0] / (double)(w0*h0));
		Mat src_mask = Mat::ones(Size(w0, h0), CV_8UC1);
		src_mask *= src_avg;
		compare(src_roi, src_mask, src_comp, CMP_GE);
		//src_comp /= 255;

		Mat src_bk_comp = Mat::ones(Size(w0, h0), CV_8UC1);
		compare(bk_comp, src_comp, src_bk_comp, CMP_EQ);
		src_bk_comp /= 255;

		int hammingDist = countNonZero(src_bk_comp);
		double radio = hammingDist / (double)(w0*h0);
		if (radio > CLEAR_BIG_THAN_THIS_HAMMING_RADIO)
		{
			rects.erase(rects.begin() + i);
			i--;
			rectangle(resultImg, Point(x0, y0), Point(x0 + w0, y0 + h0), Scalar(255, 0, 0), 1, 8); //绘制第i个外接矩形
			std::stringstream ss;
			ss << "bad:" << position << "," << radio;
			std::string s = ss.str();
			putText(resultImg, s.c_str(), Point(x0, y0 + 5), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 1);
		}
		else
		{
			rectangle(resultImg, Point(x0, y0), Point(x0 + w0, y0 + h0), Scalar(0, 255, 0), 1, 8); //绘制第i个外接矩形
			std::stringstream ss;
			ss << "good:" << position << "," << radio;
			std::string s = ss.str();
			putText(resultImg, s.c_str(), Point(x0, y0 + 5), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 1);
		}

	}
}

void hammingBiImgClear(vector<Rect> &rects, Mat & bk, Mat & src, Mat & diff)
{
	int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
	for (int i = 0; i< rects.size(); i++)
	{
		if (i < 0 || i >= rects.size())
			break;
		x0 = rects[i].x;  //获得第i个外接矩形的左上角的x坐标
		y0 = rects[i].y; //获得第i个外接矩形的左上角的y坐标
		w0 = rects[i].width; //获得第i个外接矩形的宽度
		h0 = rects[i].height; //获得第i个外接矩形的高度

		Mat StandardDeviationSrcRoi, src_mean, src_sd, bk_roi_255; // 用于统计src roi二值化遮罩里面的标准差
		Mat bk_comp, src_comp, binary_roi_mask_turn, binary_roi_mask_turn255;
		Mat roi_ones = Mat::ones(Size(w0, h0), CV_8UC1);
		Mat bk_mask = Mat::ones(Size(w0, h0), CV_8UC1);
		Mat src_mask = Mat::ones(Size(w0, h0), CV_8UC1);

		cv::Mat bk_roi = bk(cv::Rect(x0, y0, w0, h0)).clone();
		cv::Mat src_roi = src(cv::Rect(x0, y0, w0, h0)).clone();
		cv::Mat binary_roi_mask = diff(cv::Rect(x0, y0, w0, h0)).clone();//存放src二值图对应roi
		binary_roi_mask /= 255;//二值图遮罩
		compare(binary_roi_mask, roi_ones, binary_roi_mask_turn255, CMP_NE);
		binary_roi_mask_turn = binary_roi_mask_turn255 / 255;//01反转的二值图遮罩
		int maskArea = countNonZero(binary_roi_mask);

		imwrite("649src_roi.bmp", src_roi);

		bk_roi = bk_roi.mul(binary_roi_mask);  //套上mask
		Scalar bk_sum = sum(bk_roi);
		int bk_avg = cvRound(bk_sum.val[0] / (double)maskArea);
		bk_mask *= bk_avg;

		bk_roi_255 = bk_roi + binary_roi_mask_turn255; // bk_roi_255的二值图遮罩外面都是255
		compare(bk_roi_255, bk_mask, bk_comp, CMP_GE);// bk_comp的二值图遮罩外面都是1

		src_roi = src_roi.mul(binary_roi_mask); //src_roi的二值图遮罩外面都是0
		Scalar src_sum = sum(src_roi);
		int src_avg = cvRound(src_sum.val[0] / (double)maskArea);
		src_mask *= src_avg;
		compare(src_roi, src_mask, src_comp, CMP_GE); // src_comp的二值图遮罩外面都是0

		StandardDeviationSrcRoi = src_mask.mul(binary_roi_mask_turn);//遮罩外面都存放着src_roi的均值，这样可以不计入标准差的计算
		StandardDeviationSrcRoi += src_roi;//遮罩里面都是src_roi原数据
		meanStdDev(StandardDeviationSrcRoi, src_mean, src_sd);

		//计算汉明距离
		Mat src_bk_comp = Mat::ones(Size(w0, h0), CV_8UC1);
		compare(bk_comp, src_comp, src_bk_comp, CMP_EQ);//src_comp和bk_comp的二值图遮罩外面不相等所以都是0，不计入统计
		src_bk_comp /= 255;
		int hammingDist = countNonZero(src_bk_comp);
		double radio = hammingDist / (double)maskArea;

		//计算标准差
		meanStdDev(src_roi, src_mean, src_sd, binary_roi_mask);
		double m = src_mean.at<double>(0, 0);
		double sd = src_sd.at<double>(0, 0);

		imwrite("src_roi.bmp", src_roi);
		imwrite("bk_roi.bmp", bk_roi);
		imwrite("binary_roi_mask.bmp", binary_roi_mask);


		if (radio > CLEAR_BIG_THAN_THIS_HAMMING_RADIO || (radio > CLEAR_BIG_THAN_THIS_HAMMING_RADIO  && sd < 20))
		{
			rects.erase(rects.begin() + i);
			i--;
#if 0 
			rectangle(resultImg, Point(x0, y0), Point(x0 + w0, y0 + h0), Scalar(0, 0, 255), 1, 8); //绘制第i个外接矩形
			std::stringstream ss;
			ss << position << " hm:" << radio;
			std::string s = ss.str();
			putText(resultImg, s.c_str(), Point(x0, y0 + 25), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 1);
			std::stringstream sss;
			sss << "sd:" << sd;
			std::string s0 = sss.str();
			putText(resultImg, s0.c_str(), Point(x0, y0 + 45), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 1);
#endif 
		}
#if 0 
		else
		{
			rectangle(resultImg, Point(x0, y0), Point(x0 + w0, y0 + h0), Scalar(255, 0, 0), 1, 8); //绘制第i个外接矩形
			std::stringstream ss;
			ss << position << " hm:" << radio;
			std::string s = ss.str();
			putText(resultImg, s.c_str(), Point(x0, y0 + 25), FONT_HERSHEY_PLAIN, 1.0, Scalar(255, 0, 0), 1);
			std::stringstream sss;
			sss << "sd:" << sd;
			std::string s0 = sss.str();
			putText(resultImg, s0.c_str(), Point(x0, y0 + 45), FONT_HERSHEY_PLAIN, 1.0, Scalar(255, 0, 0), 1);
		}
#endif
	}
}
void drawRecs(Mat & img, vector<Rect> &rects, const Scalar& color, string &txt)
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
		ss << txt;
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
		if (ch == 27)
			return true;
	}
	return false;
}

void writeMatToFile(cv::Mat& m, const char* filename)
{
	std::ofstream fout(filename);

	if (!fout)
	{
		std::cout << "File Not Opened" << std::endl;
		return;
	}

	for (int i = 0; i<m.rows; i++)
	{
		for (int j = 0; j<m.cols; j++)
		{
			int xx = (int)m.at<uchar>(i, j);
			fout << xx << ",";
		}
		fout << std::endl;
	}

	fout.close();
}

void svm_hog_detector(vector<Rect> &rects, Mat & src)
{
	HOGDescriptor hog = HOGDescriptor();
	hog.setSVMDetector(hog.getDefaultPeopleDetector());
	int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
	for (int i = 0; i< rects.size(); i++)
	{
		x0 = rects[i].x;  //获得第i个外接矩形的左上角的x坐标
		y0 = rects[i].y; //获得第i个外接矩形的左上角的y坐标
		w0 = rects[i].width; //获得第i个外接矩形的宽度
		h0 = rects[i].height; //获得第i个外接矩形的高度
		cv::Mat roi(cv::Size(64 * 2, 128 * 2), CV_8UC1);
		cv::Mat src_roi = src(cv::Rect(x0, y0, w0, h0)).clone();

		int wid = 64 * 2;
		int hgt = 128 * 2;
		resize(src_roi, roi, Size(64 * 2, 128 * 2));
		imwrite("test.bmp", roi);
		HOGDescriptor detector(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
		vector<float> descriptor;
		vector<Point> location;
		detector.compute(roi, descriptor, Size(0, 0), Size(0, 0), location);
		//下面是调用opencv中集成的基于HOG的SVM行人检测数据集，进行行人检测

		vector<Rect> peopleLocation;
		hog.detectMultiScale(roi, peopleLocation, 0, Size(8, 8), Size(16, 16), 1.05, 2.0);
		for (int i = 0; i < peopleLocation.size(); ++i)
		{
			int xx = cvRound(peopleLocation[i].x * w0 / (double)wid);
			int yy = cvRound(peopleLocation[i].y * h0 / (double)hgt);
			int ww = cvRound(peopleLocation[i].width * w0 / (double)wid);
			int hh = cvRound(peopleLocation[i].height * h0 / (double)hgt);
			rectangle(roi, Point(peopleLocation[i].x, peopleLocation[i].y), Point(peopleLocation[i].width, peopleLocation[i].height),
				Scalar(200, 200, 0), 8);
			/*rectangle(resultImg, Point(xx + x0, yy + y0),Point(ww, hh), Scalar(200, 200, 0),8);
			std::stringstream ss;
			ss << xx<<","<<yy<<"--"<<ww<<","<<hh;
			std::string s = ss.str();
			putText(resultImg, s.c_str(), Point(xx, yy), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 200, 200), 1);*/
		}
		imshow("img", roi);
	}

}

void denoise(Mat &src, Mat &dst)
{
	/*Mat denoiseFigure, dstFigure, denoiseFigure1, out;
	Mat element = getStructuringElement(MORPH_RECT, Size(4, 4));
	medianBlur(src, denoiseFigure, 7);
	denoiseFigure1 = denoiseFigure.clone();
	GaussianBlur(denoiseFigure, denoiseFigure1, Size(3, 1), 0.0);
	erode(denoiseFigure1, dstFigure, element);
	dilate(dstFigure, out, element);
	return out;*/

	Mat element = getStructuringElement(MORPH_RECT, Size(4, 4));
	medianBlur(src, dst, 7);
	GaussianBlur(dst, dst, Size(3, 1), 0.0);
	erode(dst, dst, element);
	dilate(dst, dst, element);

}

int dealMat(Mat &img,Rect &r1)
{

	int flag = 0;
	//namedWindow("people detector");
	HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);//HOG检测器，用来计算HOG描述子的
																			  //hog.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector())
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	fflush(stdout);
	vector<Rect> found, found_filtered;
	//double t = (double)getTickCount();
	// run the detector with default parameters. to get a higher hit-rate  
	// (and more false alarms, respectively), decrease the hitThreshold and  
	// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).  
	//hog.svmDetector
	//hog.detectMultiScale(img, found, 0, Size(2, 1), Size(8, 8), 1.05, 2);
	hog.detectMultiScale(img, found, 0, Size(6, 3), Size(4, 4), 1.05, 2);
	//t = (double)getTickCount() - t;
	//printf("tdetection time = %gms\n", t*1000. / cv::getTickFrequency());
	size_t i, j;
	if (found.size() > 0)
		flag = 1;
	for (i = 0; i < found.size(); i++) {
		Rect r = found[i];
		for (j = 0; j < found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
		if (j == found.size())
			found_filtered.push_back(r);

	}
	for (i = 0; i < found_filtered.size(); i++) {
		Rect r = found_filtered[i];
		// the HOG detector returns slightly larger rectangles than the real objects.  
		// so we slightly shrink the rectangles to get a nicer output.  
		int xx = cvRound(((double)((double)r1.width * (double)r.x)) / 100.0) + r1.x;
		int yy = cvRound(((double)(r1.height * r.y)) / 200.0) + r1.y;
		int ww = cvRound(((double)(r1.width * r.width)) / 100.0);
		int hh = cvRound(((double)(r1.height * r.height)) / 200.0);
		rectangle(resultImg, Point(xx,yy), Point(xx + ww, yy + hh), cv::Scalar(0, 0, 255), 3);
		std::stringstream sss;
		sss << "SVM+HOG";
		std::string s0 = sss.str();
		putText(resultImg, s0.c_str(), Point(xx, yy), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 1);
	}
	imwrite("resultImg111.bmp", resultImg);
	//waitKey(20);
	return flag;
}

#endif