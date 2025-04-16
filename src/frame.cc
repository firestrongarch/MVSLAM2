#include "frame.h"
#include <opencv2/core/types.hpp>

namespace MVSLAM2 {

cv::Point2d Frame::Pixel2Camera(const cv::Point2d& p2d)
{
    return cv::Point2d(
        (p2d.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p2d.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

cv::Point2d Frame::Camera2Pixel(const cv::Point3d& p3d)
{
    // cv::Mat p3d_mat = (cv::Mat_<double>(3, 1) << p3d.x/p3d.z, p3d.y/p3d.z, 1);
    // cv::Mat p3d_pixel = K * p3d_mat;
    // return cv::Point2d(
    //     p3d_pixel.at<double>(0),
    //     p3d_pixel.at<double>(1)
    // );
    return cv::Point2d(
        K.at<double>(0, 0) * p3d.x / p3d.z + K.at<double>(0, 2),
        K.at<double>(1, 1) * p3d.y / p3d.z + K.at<double>(1, 2));
}

cv::Point2d Frame::World2Pixel(const cv::Point3d& p3d)
{
    cv::Mat p3d_mat = (cv::Mat_<double>(4, 1) << p3d.x, p3d.y, p3d.z, 1);
    cv::Mat P = T_wc.inv()(cv::Range(0, 3), cv::Range::all());
    cv::Mat p2d_mat = P * p3d_mat;
    cv::Mat p3d_cam = p2d_mat / p2d_mat.at<double>(2);
    cv::Mat p2d_pixel = K * p3d_cam;
    return cv::Point2d(p2d_pixel.at<double>(0) / p2d_pixel.at<double>(2), p2d_pixel.at<double>(1) / p2d_pixel.at<double>(2));
}

cv::Point3d Frame::Pixel2World(const cv::Point2d& p2d)
{
    cv::Mat p2d_mat = (cv::Mat_<double>(3, 1) << p2d.x, p2d.y, 1);
    cv::Mat P = T_wc(cv::Range(0, 3), cv::Range::all());
    cv::Mat p3d_mat = P * K * p2d_mat;
    return cv::Point3d(p3d_mat.at<double>(0) / p3d_mat.at<double>(3), p3d_mat.at<double>(1) / p3d_mat.at<double>(3), p3d_mat.at<double>(2) / p3d_mat.at<double>(3));
}

}