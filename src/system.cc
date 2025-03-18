#include "system.h"
#include "frame.h"
#include "map.h"
#include <cmath>
#include <iostream>
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

    tracker_->Track(frame);

    for (auto& kp: frame->left_kps_) {
        std::cout << "last point: " << kp.match->pt << std::endl;
        std::cout << "current point: " <<  kp.pt  << std::endl;
    }

    frame->pose_ = frame->relative_pose_ * Frame::last_frame_->pose_;
    tracker_->Pnp(frame);
    frame->relative_pose_ = frame->pose_ * Frame::last_frame_->pose_.inv();

    viewer_->AddTrajectoryPose(frame->pose_);

    // // 观察重投影误差
    // for (auto& kps : frame->left_kps_) {
    //     cv::Point3d p3d = *kps.map_point.lock();
    //     cv::Point2f p2d = frame->World2Pixel(p3d);

    //     std::cout << "Reprojection Error: " << cv::norm(p2d - kps.pt) << std::endl;

    // }

    std::cout << "Frame ID: " << frame->id << ", Timestamp: " << frame->timestamp_ << std::endl;
    // std::cout << "Pose: " << frame->pose_ << std::endl;
    Frame::last_frame_ = frame;
}

}