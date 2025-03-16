#include "system.h"
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

namespace MVSLAM2 {

void System::Run(Frame frame) {
    // cv::imshow("left", frame.left_image_);
    // cv::imshow("right", frame.right_image_);

    auto detector = cv::ORB::create();
    detector->detectAndCompute(frame.left_image_, cv::noArray(), frame.left_kps_, frame.left_des_);
    detector->detectAndCompute(frame.right_image_, cv::noArray(), frame.right_kps_, frame.right_des_);

    cv::Mat out1 = frame.left_image_.clone();
    cv::Mat out2 = frame.right_image_.clone();
    cv::cvtColor(out1, out1, cv::COLOR_GRAY2BGR);
    cv::cvtColor(out2, out2, cv::COLOR_GRAY2BGR);

    for (auto kp : frame.left_kps_) {
        cv::Point2f pt1,pt2;
        pt1.x=kp.pt.x-5;
        pt1.y=kp.pt.y-5;
        pt2.x=kp.pt.x+5;
        pt2.y=kp.pt.y+5;
        cv::rectangle(out1, pt1, pt2, cv::Scalar(0, 255, 0));
        cv::circle(out1, kp.pt, 2, cv::Scalar(0, 255, 0), cv::FILLED);
    }
    for (auto kp : frame.right_kps_) {
        cv::Point2f pt1,pt2;
        pt1.x=kp.pt.x-5;
        pt1.y=kp.pt.y-5;
        pt2.x=kp.pt.x+5;
        pt2.y=kp.pt.y+5;
        cv::rectangle(out2, pt1, pt2, cv::Scalar(0, 255, 0));
        cv::circle(out2, kp.pt, 2, cv::Scalar(0, 255, 0), cv::FILLED);
    }

    cv::imshow("left", out1);
    cv::imshow("right", out2);

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(frame.left_des_, frame.right_des_, matches);
    std::vector<cv::Point2d> pts1, pts2;
    for (auto match : matches) {
        pts1.push_back(frame.left_kps_[match.queryIdx].pt);
        pts2.push_back(frame.right_kps_[match.trainIdx].pt);
    }

    frame.points2d_ = pts1; // 只保留左图的特征点

    std::vector<cv::Point2d> pts1_cam, pts2_cam;
    pts1_cam.resize(pts1.size());
    pts2_cam.resize(pts2.size());
  
    for(size_t i = 0; i < pts1.size(); i++) {
        pts1_cam[i].x = (pts1[i].x - K_.at<double>(0,2)) / K_.at<double>(0,0);
        pts1_cam[i].y = (pts1[i].y - K_.at<double>(1,2)) / K_.at<double>(1,1);
        pts2_cam[i].x = (pts2[i].x - K_.at<double>(0,2)) / K_.at<double>(0,0);
        pts2_cam[i].y = (pts2[i].y - K_.at<double>(1,2)) / K_.at<double>(1,1);
    }

    try {
        // 第一个相机的投影矩阵：从frame.pose提取前3行
        cv::Mat P1 = cv::Mat::zeros(3, 4, CV_32F);
        frame.pose_(cv::Range(0,3), cv::Range::all()).convertTo(P1, CV_32F);

        // 第二个相机的投影矩阵：T01.inv * frame.pose，取前3行
        cv::Mat T_inv = T_01_.inv();
        cv::Mat P2 = (T_inv * frame.pose_)(cv::Range(0,3), cv::Range::all());
        P2.convertTo(P2, CV_32F);

        cv::Mat point4d;
        cv::triangulatePoints(P1, P2, pts1_cam, pts2_cam, point4d);
        // 转换成非齐次坐标
        for ( int i=0; i<point4d.cols; i++ )
        {
            cv::Mat x = point4d.col(i);
            x /= x.at<float>(3,0); // 归一化
            cv::Point3d p (
                x.at<float>(0,0), 
                x.at<float>(1,0), 
                x.at<float>(2,0) 
            );
            frame.points3d_.push_back( p );
        }
    } catch (const cv::Exception& e) {
        std::cout << "OpenCV error: " << e.what() << std::endl;
    }

    if (Frame::last_frame_) {
        // 计算相机位姿
        // 从上一帧的位姿矩阵中提取R和t
        cv::Mat R = Frame::last_frame_->pose_(cv::Range(0,3), cv::Range(0,3));
        cv::Mat tvec = Frame::last_frame_->pose_(cv::Range(0,3), cv::Range(3,4));
        cv::Mat rvec;
        cv::Rodrigues(R, rvec);  // 将旋转矩阵转换为旋转向量

        std::vector<int> inliers;
        cv::solvePnPRansac(
            Frame::last_frame_->points3d_,
            frame.points2d_,
            K_,
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
        
        frame.SetPose(pose);
    }

    std::cout << "Frame ID: " << frame.id << ", Timestamp: " << frame.timestamp_ << std::endl;

    Frame::last_frame_ = std::make_shared<Frame>(frame);
}

}