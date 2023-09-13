#include <stdio.h>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

#include <eigen3/Eigen/Dense>
#include <nanoflann.hpp>


using namespace cv;
using namespace std;


void calculateCenterOfMass(
    const cv::Point2d& topRight, 
    const cv::Point2d& bottomLeft,
    const cv::Mat& image,
    cv::Point2d& centerOfMass);
std::vector<cv::Point2d> detectWhitePoints(const cv::Mat& image);
int point_matching(std::vector<cv::Point2d>& points_ptr, std::vector<cv::Point3d>& map);


int main(int, char**)
{
    Mat image_cap = imread("img4.png");
    std::vector<cv::Point3d> map {{0.2,0,3}, {0.35,0,3}, {0.75,0,3},
                                  {0,0.2,3}, {0,0.35,3}, {0,0.75,3}};

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

    point_matching(*points_ptr, map);

    cout<<"end"<<endl;
    return 0;
}

int point_matching(std::vector<cv::Point2d>& points_ptr, std::vector<cv::Point3d>& map)
{
    // for(const auto& point : map ){ cout<<point<<endl;}

    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat camera_matrix;
    cv::Mat dist;

    dist = cv::Mat::zeros(5, 1, CV_64F);
    rvec = cv::Mat::zeros(3, 1, CV_64F);
    // rvec.at<double>(1) = -atan(1)*4;
    tvec = cv::Mat::zeros(3, 1, CV_64F);
    camera_matrix = cv::Mat::zeros(3, 3, CV_64F);

    camera_matrix.at<double>(0,0) = 1169;
    camera_matrix.at<double>(0,2) = 800;
    camera_matrix.at<double>(1,1) = 1169;
    camera_matrix.at<double>(1,2) = 540;
    camera_matrix.at<double>(2,2) = 1;

    std::vector<cv::Point2d> projected;
    // cout<<camera_matrix<<endl;

    cv::projectPoints(map, rvec, tvec, 
                      camera_matrix, dist, projected);

    cout<<projected<<endl;


    cv::Mat mat1 = cv::Mat(projected.size(),2,CV_32F,projected.data());
    cv::Mat mat2 = cv::Mat(points_ptr.size(),2,CV_32F,points_ptr.data());


    // cv::BFMatcher matcher(cv::NORM_L2, true);
    // std::vector<cv::DMatch> matches;
    // matcher.match(mat1,mat2,matches);

    nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> mat_index(projected, 10);
    mat_index.index->buildIndex();

    vector<double> query_pt(points2D.data(), points2D.data() + points2D.size());
    vector<size_t> ret_index(1);
    vector<double> out_dist_sqr(1);

    KNNResultSet<double> resultSet(1);
    resultSet.init(&ret_index[0], &out_dist_sqr[0]);
    mat_index.index->findNeighbors(resultSet, &query_pt[0], SearchParams(10));

    double total_dist = accumulate(out_dist_sqr.begin(), out_dist_sqr.end(), 0.0) / out_dist_sqr.size();



    // for (const auto& match : matches) {
    //     cout<<matches<<endl;
	// }

    // cout<<matches<<endl;




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
    // cv::circle(image, cv::Point2d(xmax, ymax), 1, cv::Scalar(0, 255, 0), cv::FILLED);
    // cv::circle(image, cv::Point2d(xmin, ymax), 1, cv::Scalar(0, 255, 0), cv::FILLED);
    // cv::circle(image, cv::Point2d(xmax, ymin), 1, cv::Scalar(0, 255, 0), cv::FILLED);
    // cv::circle(image, cv::Point2d(xmin, ymin), 1, cv::Scalar(0, 255, 0), cv::FILLED);

    double totalMass = 0;
    double sumX = 0;
    double sumY = 0;


    for (int x = xmin; x < xmax; ++x)
    {
        for (int y = ymin; y < ymax; ++y)
        {
            cv::Vec3b point = image.at<Vec3b>(y,x);
            double point_avg = (double)((point[0]+point[1]+point[2])/3);
            double mass = point_avg;
            totalMass += mass;
            sumX += mass * x;
            sumY += mass * y;

        }
    }
    if (totalMass == 0) {
        // return {0, 0};  // Return an arbitrary point for an empty matrix
    }
    // std::cout<<"sumX = "<<sumX<<std::endl;
    // std::cout<<"totalMass = "<<totalMass<<std::endl;

    double centerX = sumX / totalMass;
    double centerY = sumY / totalMass;

    centerOfMass.x = centerX;
    centerOfMass.y = centerY;
}

std::vector<cv::Point2d> detectWhitePoints(const cv::Mat& image) {
    // std::cout<<"start"<<endl;
	cv::Mat grayImage;
	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

	cv::Mat binaryImage;
	cv::threshold(grayImage, binaryImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	std::vector<cv::Point2d> centers;

	for (const auto& contour : contours) {
        if (cv::contourArea(contour) < 5)
            continue;

        cv::Rect boundingRect = cv::boundingRect(contour);
        cv::Point2d topRight = { (double)boundingRect.x + (double)boundingRect.width, 
                                 (double)boundingRect.y + (double)boundingRect.height };
        cv::Point2d bottomLeft = { (double)boundingRect.x, (double)boundingRect.y };

        cv::Point2d centerOfMass;
        calculateCenterOfMass(topRight, bottomLeft, image, centerOfMass);
		// std::cout<<centerOfMass<<std::endl;
		centers.push_back(centerOfMass);
	}
	std::cout<<centers<<std::endl;
	return centers;
}