#pragma once
#include <memory>
#include <opencv2/core.hpp>

namespace MVSLAM2 {
struct Frame {
    using Ptr = std::shared_ptr<Frame>;

    const cv::Mat left_image_;
    const cv::Mat right_image_;
    const double timestamp_;

    std::vector<cv::KeyPoint> left_kps_;
    cv::Mat left_des_;
    std::vector<cv::KeyPoint> right_kps_;
    cv::Mat right_des_;

    cv::Mat relative_pose_;

    void SetPose(const cv::Mat& pose) {
        relative_pose_ = pose;
    }
};

}
