#pragma once
#include <memory>
#include <opencv2/core.hpp>

namespace MVSLAM2 {
struct Frame {
    using Ptr = std::shared_ptr<Frame>;

    cv::Mat left_image_;
    cv::Mat right_image_;
    double timestamp_;
    cv::Mat relative_pose_;

    void SetPose(const cv::Mat& pose) {
        relative_pose_ = pose;
    }
};

}
