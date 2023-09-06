#include <stdio.h>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int, char**)
{
    cv::VideoCapture cap(0);
    if(!cap.isOpened()) {
        cout<<"error"<<endl;
        return -1;
    }
    Mat frame, edges;
    namedWindow("edges", WINDOW_AUTOSIZE);
    double* ptr_;
    for(;;)
    {
        cap >> frame;

        cv::Vec3b pixel = frame.at<Vec3b>(cv::Point (100, 100));
		int b, g, r;
		b = pixel[0];
		g = pixel[1];
		r = pixel[2];

		std::string rgbText = "[" + std::to_string(r) + ", " + std::to_string(g)
			+ ", " + std::to_string(b) + "]";

        cv::Scalar textColor = cv::Scalar(0,255,255);
        cv::putText(frame, rgbText, cv::Point2d(20, 120),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, textColor);
        imshow("edges", frame);
        if(waitKey(30) >= 0) break;
    }
    return 0;
}