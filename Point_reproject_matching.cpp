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


template <typename T>
struct PointCloud
{
    struct Point
    {
        T x, y, z;
    };

    using coord_t = T;  //!< The type of each coordinate

    std::vector<Point> pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate
    // value, the
    //  "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return pts[idx].x;
        else if (dim == 1)
            return pts[idx].y;
        else
            return pts[idx].z;
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned
    //   in "bb" so it can be avoided to redo it again. Look at bb.size() to
    //   find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const
    {
        return false;
    }
};



template <typename num_t>
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

    
    using matrix_t = Eigen::Matrix<double, 6, 2>;
    // matrix_t mat1(6, 2);
    // matrix_t mat2(6, 2);
    Eigen::Matrix<double, 6, 2> mat1;
    Eigen::Matrix<double, 6, 2> mat2;
    
    // mat1 << projected.data();
    // Eigen::Matrix<double, 6, 2> mat2;
    // mat1 << projected;
    mat1(0,0) = projected.data()[0].x;
    mat1(0,1) = projected.data()[0].y;
    mat1(1,0) = projected.data()[1].x;
    mat1(1,1) = projected.data()[1].y;
    mat1(2,0) = projected.data()[2].x;
    mat1(2,1) = projected.data()[2].y;
    mat1(3,0) = projected.data()[3].x;
    mat1(3,1) = projected.data()[3].y;
    mat1(4,0) = projected.data()[4].x;
    mat1(4,1) = projected.data()[4].y;
    mat1(5,0) = projected.data()[5].x;
    mat1(5,1) = projected.data()[5].y;
    // mat1[1] << 2;
    // mat1[2] << 3;
    cout<<mat1<<endl;

    mat2(0,0) = points_ptr.data()[0].x;
    mat2(0,1) = points_ptr.data()[0].y;
    mat2(1,0) = points_ptr.data()[1].x;
    mat2(1,1) = points_ptr.data()[1].y;
    mat2(2,0) = points_ptr.data()[2].x;
    mat2(2,1) = points_ptr.data()[2].y;
    mat2(3,0) = points_ptr.data()[3].x;
    mat2(3,1) = points_ptr.data()[3].y;
    mat2(4,0) = points_ptr.data()[4].x;
    mat2(4,1) = points_ptr.data()[4].y;
    mat2(5,0) = points_ptr.data()[5].x;
    mat2(5,1) = points_ptr.data()[5].y;
    cout<<mat2<<endl;

    // const num_t max_range = 20;

    using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<
            num_t, Eigen::Matrix<double, 6, 2>>,
        Eigen::Matrix<double, 6, 2>, 2 /* dim */
        >;
    my_kd_tree_t index(2, std::cref(mat1), 10 /* max leaf */);

    // // do a knn search
    const size_t        num_results = 1;
    size_t                         ret_index;
    num_t                          out_dist_sqr;

    nanoflann::KNNResultSet<num_t> resultSet(num_results);

    resultSet.init(&ret_index, &out_dist_sqr);
    num_t query_pt[3] = {0.5, 0.5, 0.5};
    index.findNeighbors(resultSet, &query_pt[0]);

    std::cout << "knnSearch(nn=" << num_results << "): \n";




    // for (const auto& match : matches) {
    //     cout<<matches<<endl;
	// }

    // cout<<matches<<endl;




    return 0;
}


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
    // imshow("frame", image_cap);
    // waitKey(0);

    point_matching<float>(*points_ptr, map);

    cout<<"end"<<endl;
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