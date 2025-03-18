#include "system.h"
#include "frame.h"
#include "map.h"
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

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


    std::cout << "Matches: " << matches.size() << std::endl;
    std::cout << "GoodMatches: " << good_matches.size() << std::endl;
    // 保存匹配的特征点和对应的3D点
    for (const auto& match : good_matches) {
        KeyPoint kp = kps2[match.trainIdx];
        kp.map_point = Frame::last_frame_->left_kps_[match.queryIdx].map_point;
        frame->left_kps_.push_back(kp);

        // std::cout << "last point: " << Frame::last_frame_->left_kps_[match.queryIdx].pt << std::endl;
        // std::cout << "current point: " <<  kp.pt  << std::endl;
    }

    frame->pose_ = frame->relative_pose_ * Frame::last_frame_->pose_;
    // 计算相机位姿
    // 从上一帧的位姿矩阵中提取R和t
    cv::Mat R = Frame::last_frame_->pose_(cv::Range(0,3), cv::Range(0,3));
    cv::Mat tvec = Frame::last_frame_->pose_(cv::Range(0,3), cv::Range(3,4));
    cv::Mat rvec;
    cv::Rodrigues(R, rvec);  // 将旋转矩阵转换为旋转向量
    std::vector<int> inliers;
    std::vector<cv::Point3d> points3d;
    std::vector<cv::Point2d> points2d;

    for (const auto& kp : frame->left_kps_) {
        cv::Point3d p3d = *kp.map_point.lock();
        points3d.push_back(p3d);
        points2d.push_back(kp.pt);
    }

    cv::solvePnPRansac(
        points3d,
        points2d,
        frame->K,
        cv::Mat(),
        rvec,
        tvec,
        true,
        100, // max iterations
        8.0, // reprojection error
        0.99, // confidence
        inliers
    );
    cv::Rodrigues(rvec, R);
    cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(pose(cv::Rect(0, 0, 3, 3)));
    tvec.copyTo(pose(cv::Rect(3, 0, 1, 3)));
    frame->pose_ = pose;
    frame->relative_pose_ = frame->pose_ * Frame::last_frame_->pose_.inv();
    viewer_->AddTrajectoryPose(frame->pose_);

    // // 观察重投影误差
    // for (int i = 0; i < points2d.size(); i++) {
    //     cv::Point2d projected = frame->World2Pixel(points3d[i]);
    //     std::cout << "Point " << i << " - Original: (" << points2d[i].x << ", " << points2d[i].y 
    //              << "), Projected: (" << projected.x << ", " << projected.y << ")" << std::endl;
    //     std::cout << "Reprojection Error: " << cv::norm(points2d[i] - projected) << std::endl;
    // }

    
    // // 清除外点
    // for 

    std::cout << "Frame ID: " << frame->id << ", Timestamp: " << frame->timestamp_ << std::endl;
    // std::cout << "Pose: " << frame->pose_ << std::endl;
    Frame::last_frame_ = frame;
}

}