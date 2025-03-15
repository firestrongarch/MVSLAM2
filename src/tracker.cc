#include "tracker.h"
#include <opencv2/opencv.hpp>

namespace MVSLAM2 {

std::unique_ptr<Tracker> Tracker::Create(const std::string& type) {
    if (type == "ORB") {
        return std::make_unique<ORBTracker>();
    }
    return nullptr;
}

bool ORBTracker::Track(Frame& frame) {
    try {
        auto detector = cv::ORB::create(
            params_.nfeatures,
            params_.scale_factor,
            params_.nlevels
        );
        
        // 检测特征点
        detector->detectAndCompute(frame.left_image_, cv::noArray(), 
                                 frame.left_kps_, frame.left_des_);
        detector->detectAndCompute(frame.right_image_, cv::noArray(), 
                                 frame.right_kps_, frame.right_des_);

        // 显示结果
        cv::Mat out1 = frame.left_image_.clone();
        cv::Mat out2 = frame.right_image_.clone();
        cv::cvtColor(out1, out1, cv::COLOR_GRAY2BGR);
        cv::cvtColor(out2, out2, cv::COLOR_GRAY2BGR);

        DrawFeatures(out1, frame.left_kps_);
        DrawFeatures(out2, frame.right_kps_);

        cv::imshow("left", out1);
        cv::imshow("right", out2);
        
        return true;
    } catch (const cv::Exception& e) {
        // 错误处理
        return false;
    }
}

void ORBTracker::DrawFeatures(cv::Mat& image, const std::vector<cv::KeyPoint>& kpts) {
    for (const auto& kp : kpts) {
        cv::Point2f pt1{kp.pt.x - 5, kp.pt.y - 5};
        cv::Point2f pt2{kp.pt.x + 5, kp.pt.y + 5};
        cv::rectangle(image, pt1, pt2, cv::Scalar(0, 255, 0));
        cv::circle(image, kp.pt, 2, cv::Scalar(0, 255, 0), cv::FILLED);
    }
}

bool ORBTracker::Configure(const TrackerParams& params) {
    params_ = params;
    return true;
}

}