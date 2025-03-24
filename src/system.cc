#include "system.h"
#include "frame.h"
#include "map.h"
#include <cmath>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <print>

namespace MVSLAM2 {

void System::Run(Frame::Ptr frame) {

    if (Frame::last_frame_ == nullptr) {
        tracker_->Extract3d(frame, map_);
        Frame::kf = frame;
        Frame::last_frame_ = frame;

        return;
    }

    // 特征点太少，不进行处理
    if (Frame::last_frame_->kps.size() < 10) {
        return;
    }

    tracker_->Track(frame);

    viewer_->DrawMatches(frame);
    // viewer_->DrawReprojection(frame);
    // 打印重投影误差
    for (auto& kp : frame->kps) {
        cv::Point2f pt_rep = frame->World2Pixel(* kp.map_point.lock());
        std::println("Reprojection Error: {}", cv::norm(kp.pt - pt_rep));
    }
    cv::waitKey(0);

    // frame->T_wc = frame->T_ww * Frame::last_frame_->T_wc;
    // tracker_->Pnp(frame);
    // frame->T_ww = frame->T_wc * Frame::last_frame_->T_wc.inv();
    // viewer_->AddTrajectoryPose(frame->T_wc);

    // 补充特征点
    if (frame->kps.size() < 50) {
        tracker_->Extract3d(frame, map_);
    }

    std::println("Frame ID: {}, Timestamp: {}",frame->id,frame->timestamp_);

    Frame::last_frame_ = frame;
}

}