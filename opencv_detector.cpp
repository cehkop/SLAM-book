#include <stdio.h>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iostream>


using namespace cv;
using namespace std;


void calculateCenterOfMass(
    const cv::Point2d& topRight, 
    const cv::Point2d& bottomLeft,
    const cv::Mat& image,
    cv::Point2d& centerOfMass);
std::vector<cv::Point2d> detectWhitePoints(const cv::Mat& image);


int main(int, char**)
{
    Mat image_cap = imread("img.png", IMREAD_COLOR);

    double* ptr_;

    cv::Vec3b pixel = image_cap.at<Vec3b>(cv::Point (535, 322));
    int b, g, r;
    b = pixel[0];
    g = pixel[1];
    r = pixel[2];

    std::string rgbText = "[" + std::to_string(r) + ", " + std::to_string(g)
        + ", " + std::to_string(b) + "]";

    cv::Scalar textColor = cv::Scalar(0,255,255);
    cv::putText(image_cap, rgbText, cv::Point2d(20, 120),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, textColor);

    std::shared_ptr<std::vector<cv::Point2d>> detect_point = 
            std::make_shared<std::vector<cv::Point2d>>(
                detectWhitePoints(image_cap)
                );

    imshow("frame", image_cap);
    waitKey(0);
    return 0;
}

void calculateCenterOfMass(
    const cv::Point2d& topRight, 
    const cv::Point2d& bottomLeft,
    const cv::Mat& image,
    cv::Point2d& centerOfMass) {


    int e = 3;
    int xmax = topRight.x;
    int ymax = topRight.y;
    int xmin = bottomLeft.x;
    int ymin = bottomLeft.y;
    // std::cout<<xmax<<std::endl;
    // std::cout<<ymax<<std::endl;
    // std::cout<<xmin<<std::endl;
    // std::cout<<ymin<<std::endl;
    cv::Mat gray_image;
	cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    imshow("gray", gray_image);
    waitKey(0);


    double totalMass = 0;
    double sumX = 0;
    double sumY = 0;

    for (int x = xmin; x < xmax; ++x)
    {
        for (int y = ymin; y < ymax; ++y)
        {
            cv::Vec3b point = gray_image.at<Vec3b>(x,y);
            double point_avg = (double)((point[0]+point[1]+point[2])/3);
            // std::cout<<point_avg<<std::endl;
    //         // if (point > 5)
    //         // {
                // std::cout<<point<<std::endl;
                // std::cout<<image.at<Vec3b>(x,y)<<std::endl;
                // std::cout<<gray_image.at<Vec3b>(x,y)<<std::endl;
    //         // }
            double mass = point_avg;
            totalMass += mass;
            sumX += mass * x;
            sumY += mass * y;

        }
    }
    if (totalMass == 0) {
        // return {0, 0};  // Return an arbitrary point for an empty matrix
    }

    double centerX = sumX / totalMass;
    double centerY = sumY / totalMass;

    std::cout<<"X = "<<centerX<<std::endl;
    std::cout<<"Y = "<<centerY<<std::endl;
}

std::vector<cv::Point2d> detectWhitePoints(const cv::Mat& image) {
    std::cout<<"start"<<endl;
    // std::vector<cv::Point2d> out;
    // return out;


	cv::Mat grayImage;
	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

	cv::Mat binaryImage;
	cv::threshold(grayImage, binaryImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	// ����� �������� �� �������������� �����������
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	std::vector<cv::Point2d> centers;

	for (const auto& contour : contours) {
        if (cv::contourArea(contour) < 5)
            continue;

        cv::Rect boundingRect = cv::boundingRect(contour);
        cv::Point2d topRight = { (double)boundingRect.x + (double)boundingRect.width, (double)boundingRect.y + (double)boundingRect.height };
        cv::Point2d bottomLeft = { (double)boundingRect.x, (double)boundingRect.y };

        // Вычисляем центр масс контура точки
        cv::Point2d centerOfMass;
        calculateCenterOfMass(topRight, bottomLeft, image, centerOfMass);
		std::cout<<centerOfMass<<std::endl;
		centers.push_back(centerOfMass);
	}

	std::vector<cv::Rect> boundingRectangles;
	for (const auto& contour : contours) {
		cv::Rect boundingRect = cv::boundingRect(contour);
		boundingRectangles.push_back(boundingRect);
	}

	std::vector<cv::Rect> mergedRectangles = boundingRectangles;

	std::vector<cv::Point2d> centers_;

	for (const auto& rect : mergedRectangles) {
		cv::Point2d center(rect.x + rect.width / 2.0f, rect.y + rect.height / 2.0f);
		if(
            center.x < 1400 and center.y < 1100
            and
            center.x > 200 and center.y > 200
        )
        {
			centers_.push_back(center);
		}
	}
	std::cout<<centers_<<std::endl;
	return centers;
}