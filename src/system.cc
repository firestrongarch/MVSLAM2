#include "system.h"
#include "frame.h"
#include "map.h"
#include <cmath>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace MVSLAM2 {

void System::Run(Frame::Ptr frame) {

    if (Frame::last_frame_ == nullptr) {
        tracker_->Extract3d(frame, map_);
        Frame::kf = frame;
        Frame::last_frame_ = frame;

        return;
    }

    // 特征点太少，不进行处理
    if (Frame::last_frame_->left_kps_.size() < 10) {
        return;
    }

    tracker_->Track(frame);

    viewer_->DrawMatches(frame);
    // cv::waitKey(30);

    // frame->pose_ = frame->relative_pose_ * Frame::last_frame_->pose_;
    tracker_->Pnp(frame);
    // frame->relative_pose_ = frame->pose_ * Frame::last_frame_->pose_.inv();

    viewer_->AddTrajectoryPose(frame->pose_);

    // 补充特征点
    if (frame->left_kps_.size() < 50) {
        tracker_->Extract3d(frame, map_);
    }

    std::cout << "Frame ID: " << frame->id << ", Timestamp: " << frame->timestamp_ << std::endl;
    // std::cout << "Pose: " << frame->pose_ << std::endl;
    Frame::last_frame_ = frame;
}

}