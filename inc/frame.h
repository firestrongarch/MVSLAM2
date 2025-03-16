#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

namespace MVSLAM2 {
struct Frame {
    using Ptr = std::shared_ptr<Frame>;

    const cv::Mat left_image_;
    const cv::Mat right_image_;
    const double timestamp_;
    const int id;

    std::vector<cv::KeyPoint> left_kps_;
    cv::Mat left_des_;
    // std::vector<cv::KeyPoint> right_kps_;
    // cv::Mat right_des_;

    std::vector<cv::Point3d> points3d_; // 世界坐标系
    std::vector<cv::Point2d> points2d_; // 左图像素坐标系
    cv::Mat pose_ = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat relative_pose_ = cv::Mat::eye(4, 4, CV_64F);

    static Ptr last_frame_;
};

}
