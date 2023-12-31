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
    Mat image_cap = imread("img.png");

    double* ptr_;

    std::shared_ptr<std::vector<cv::Point2d>> detect_point = 
            std::make_shared<std::vector<cv::Point2d>>(
                detectWhitePoints(image_cap)
                );
    std::vector<cv::Point2d>* points_ptr = detect_point.get();
    for (const auto& point : *points_ptr) {
		cv::circle(image_cap, point, 2, cv::Scalar(0, 0, 255), cv::FILLED);
	}
    imshow("frame", image_cap);
    waitKey(0);
    return 0;
}

void calculateCenterOfMass(
    const cv::Point2d& topRight, 
    const cv::Point2d& bottomLeft,
    const cv::Mat& image,
    cv::Point2d& centerOfMass) {


    int e = 2;
    int xmax = topRight.x + e;
    int ymax = topRight.y + e;
    int xmin = bottomLeft.x - e;
    int ymin = bottomLeft.y - e;
    // std::cout<<xmax<<std::endl;
    // std::cout<<ymax<<std::endl;
    // std::cout<<xmin<<std::endl;
    // std::cout<<ymin<<std::endl;
    // cv::Mat gray_image;
	// cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    // imshow("gray", gray_image);
    // waitKey(0);
    // cv::drawContours(image, {xmax, ymax, xmin, ymin});
    cv::circle(image, cv::Point2d(xmax, ymax), 1, cv::Scalar(0, 255, 0), cv::FILLED);
    cv::circle(image, cv::Point2d(xmin, ymax), 1, cv::Scalar(0, 255, 0), cv::FILLED);
    cv::circle(image, cv::Point2d(xmax, ymin), 1, cv::Scalar(0, 255, 0), cv::FILLED);
    cv::circle(image, cv::Point2d(xmin, ymin), 1, cv::Scalar(0, 255, 0), cv::FILLED);




    double totalMass = 0;
    double sumX = 0;
    double sumY = 0;


    for (int x = xmin; x < xmax; ++x)
    {
        for (int y = ymin; y < ymax; ++y)
        {

            // std::cout<<"x = "<<x<<"    y = "<<y<<endl;
            // cv::Mat new_img;
            // image.copyTo(new_img);

            // cv::circle(new_img, cv::Point2d(x, y), 1, cv::Scalar(0, 0, 255), cv::FILLED);
            cv::Vec3b point = image.at<Vec3b>(y,x);
            // Point3_<uchar>& p = *image.ptr<Point3_<uchar> >(y,x);

            // std::cout << "Test_point = " << point << std::endl;
            double point_avg = (double)((point[0]+point[1]+point[2])/3);
            // image.at<Vec3b>(x,y) = {0,0,255};
            double mass = point_avg;
            totalMass += mass;
            sumX += mass * x;
            sumY += mass * y;
            // cv::imshow("test_point", new_img);
            // cv::waitKey(0);

        }
    }
    if (totalMass == 0) {
        // return {0, 0};  // Return an arbitrary point for an empty matrix
    }
    std::cout<<"sumX = "<<sumX<<std::endl;
    std::cout<<"totalMass = "<<totalMass<<std::endl;

    double centerX = sumX / totalMass;
    double centerY = sumY / totalMass;

    // std::cout<<"X = "<<centerX<<std::endl;
    // std::cout<<"Y = "<<centerY<<std::endl;
    centerOfMass.x = centerX;
    centerOfMass.y = centerY;
    // std::cout<<"end"<<endl;
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
    // std::cout<<"start"<<endl;


	for (const auto& contour : contours) {
        if (cv::contourArea(contour) < 5)
            continue;

        cv::Rect boundingRect = cv::boundingRect(contour);
        cv::Point2d topRight = { (double)boundingRect.x + (double)boundingRect.width, 
                                 (double)boundingRect.y + (double)boundingRect.height };
        cv::Point2d bottomLeft = { (double)boundingRect.x, (double)boundingRect.y };
        // std::cout<<"start"<<endl;

        // Вычисляем центр масс контура точки
        cv::Point2d centerOfMass;
        calculateCenterOfMass(topRight, bottomLeft, image, centerOfMass);
		// std::cout<<centerOfMass<<std::endl;
		centers.push_back(centerOfMass);
	}
    // std::cout<<"end"<<endl;

	std::cout<<centers<<std::endl;

	std::vector<cv::Rect> boundingRectangles;
	for (const auto& contour : contours) {
		cv::Rect boundingRect = cv::boundingRect(contour);
		boundingRectangles.push_back(boundingRect);
	}
    // std::cout<<"start"<<endl;

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