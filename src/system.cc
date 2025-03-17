#include "system.h"
#include "frame.h"
#include "map.h"
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

namespace MVSLAM2 {

void System::Run(Frame::Ptr frame) {

    if (Frame::last_frame_ == nullptr) {
        Triangulate(frame);
        Frame::kf = frame;
        Frame::last_frame_ = frame;

        return;
    }

    std::cout << "detect 3D Point "<< std::endl;
    // 循环显示3d点
    for (size_t i = 0; i < Frame::last_frame_->left_kps_.size(); ++i) {
        MapPoint::Ptr map_point = Frame::last_frame_->left_kps_[i].map_point.lock();
        std::cout << "3D Point " << i << ": " << *map_point << std::endl;
    }
    // // 跟踪上一帧
    // auto detector = cv::ORB::create(2000);
    // detector->detectAndCompute(frame->left_image_, cv::noArray(), frame->left_kps_, frame->left_des_);

    // cv::BFMatcher matcher(cv::NORM_HAMMING);
    // std::vector<cv::DMatch> matches;
    // matcher.match(frame->left_des_, Frame::last_frame_->left_des_, matches);
    
    // // 清空当前帧的2D和3D点
    // frame->points2d_.clear();
    // frame->points3d_.clear();
    
    // // 保存匹配的特征点和对应的3D点
    // for (auto match : matches) {
    //     // 确保索引有效
    //     if (match.trainIdx >= Frame::last_frame_->points3d_.size()) {
    //         continue;
    //     }
           
    //     // 当前帧的2D点
    //     frame->points2d_.push_back(frame->left_kps_[match.queryIdx].pt);
    //     // 对应的上一帧3D点
    //     frame->points3d_.push_back(Frame::last_frame_->points3d_[match.trainIdx]);
    // }

    // frame->pose_ = frame->relative_pose_ * Frame::last_frame_->pose_;
    // // 计算相机位姿
    // // 从上一帧的位姿矩阵中提取R和t
    // cv::Mat R = Frame::last_frame_->pose_(cv::Range(0,3), cv::Range(0,3));
    // cv::Mat tvec = Frame::last_frame_->pose_(cv::Range(0,3), cv::Range(3,4));
    // cv::Mat rvec;
    // cv::Rodrigues(R, rvec);  // 将旋转矩阵转换为旋转向量
    // std::vector<int> inliers;
    // cv::solvePnPRansac(
    //     frame->points3d_,
    //     frame->points2d_,
    //     frame->K,
    //     cv::Mat(),
    //     rvec,
    //     tvec,
    //     true,
    //     30, // max iterations
    //     3.0, // reprojection error
    //     0.95, // confidence
    //     inliers
    // );
    // cv::Rodrigues(rvec, R);
    
    // cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
    // R.copyTo(pose(cv::Rect(0, 0, 3, 3)));
    // tvec.copyTo(pose(cv::Rect(3, 0, 1, 3)));
    
    // frame->pose_ = pose;
    // frame->relative_pose_ = frame->pose_ * Frame::last_frame_->pose_.inv();


    // viewer_->AddTrajectoryPose(frame->pose_);

    std::cout << "Frame ID: " << frame->id << ", Timestamp: " << frame->timestamp_ << std::endl;
    // std::cout << "Pose: " << frame->pose_ << std::endl;
    Frame::last_frame_ = frame;
}

}