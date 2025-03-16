#include "system.h"
#include "frame.h"
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

namespace MVSLAM2 {

void System::Run(Frame::Ptr frame) {
    // cv::imshow("left", frame->left_image_);
    // cv::imshow("right", frame->right_image_);

    if (Frame::last_frame_ == nullptr) {
        Triangulate(frame);
        Frame::last_frame_ = frame;

        return;
    }

    frame->pose_ = frame->relative_pose_ * Frame::last_frame_->pose_;

    // 跟踪上一帧
    auto detector = cv::ORB::create();
    detector->detectAndCompute(frame->left_image_, cv::noArray(), frame->left_kps_, frame->left_des_);

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(frame->left_des_, Frame::last_frame_->left_des_, matches);
    
    // 清空当前帧的2D和3D点
    frame->points2d_.clear();
    frame->points3d_.clear();
    
    // 保存匹配的特征点和对应的3D点
    for (auto match : matches) {
        // 确保索引有效
        if (match.trainIdx >= Frame::last_frame_->points3d_.size()) {
            continue;
        }
        // 当前帧的2D点
        frame->points2d_.push_back(frame->left_kps_[match.queryIdx].pt);
        // 对应的上一帧3D点
        frame->points3d_.push_back(Frame::last_frame_->points3d_[match.trainIdx]);
    }

    // 验证3D-2D投影
    for(size_t i = 0; i < frame->points3d_.size() && i < 5; ++i) {  // 只显示前5个点
        cv::Point3d p3d = frame->points3d_[i];
        cv::Point2d p2d = frame->points2d_[i];
        
        // 计算投影点
        cv::Mat P = K_ * frame->pose_(cv::Range(0,3), cv::Range::all());
        cv::Mat p3d_mat = (cv::Mat_<double>(4,1) << p3d.x, p3d.y, p3d.z, 1);
        cv::Mat p2d_proj = P * p3d_mat;
        
        // 转换为像素坐标
        cv::Point2d p2d_projected(p2d_proj.at<double>(0)/p2d_proj.at<double>(2), 
                                p2d_proj.at<double>(1)/p2d_proj.at<double>(2));
        
        std::cout << "Point " << i << ":\n"
                  << "3D: " << p3d << "\n"
                  << "2D original: " << p2d << "\n"
                  << "2D projected: " << p2d_projected << "\n"
                  << "Reprojection error: " << cv::norm(p2d - p2d_projected) << "\n"
                  << "-------------------" << std::endl;
    }

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
    //     K_,
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

    // frame->points2d_.clear();
    // frame->points3d_.clear();
    // Triangulate(frame);

    // viewer_->AddTrajectoryPose(frame->pose_);

    // std::cout << "Frame ID: " << frame->id << ", Timestamp: " << frame->timestamp_ << std::endl;
    // std::cout << "Pose: " << frame->pose_ << std::endl;

    Frame::last_frame_ = frame;
}

}