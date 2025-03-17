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

    // 跟踪上一帧
    auto detector = cv::ORB::create(2000);
    std::vector<cv::KeyPoint> kps1;
    for (auto kp : Frame::last_frame_->left_kps_) {
        kps1.push_back(kp);
    }
    cv::Mat des1;
    detector->compute(Frame::last_frame_->left_image_, kps1, des1);

    std::vector<cv::KeyPoint> kps2;
    cv::Mat des2;
    detector->detectAndCompute(frame->left_image_, cv::noArray(), kps2, des2);

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(des1, des2, matches);

    std::cout << "Matches: " << matches.size() << std::endl;
    // 匹配点对筛选
    double min_dist = 10000, max_dist = 0;
    // 找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < des1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    // 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < des1.rows; i++) {
        if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
            good_matches.push_back(matches[i]);
        }
    }
    std::cout << "GoodMatches: " << good_matches.size() << std::endl;

    // 保存匹配的特征点和对应的3D点
    for (const auto& match : good_matches) {
        MapPoint::Ptr map_point = Frame::last_frame_->left_kps_[match.queryIdx].map_point.lock();
        std::cout << "3D Point " << match.queryIdx << ": " << *map_point << std::endl;
    }

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