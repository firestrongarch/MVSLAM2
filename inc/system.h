#pragma once
#include "opencv2/core/core.hpp"
#include "frame.h"
#include <opencv2/core/mat.hpp>
#include <thread>
#include "viewer.h"

namespace MVSLAM2 {

class System {
public:
    System(cv::Mat K, cv::Mat T_01_) : K_(K), T_01_(T_01_) {
        viewer_ = std::make_shared<Viewer>();
        viewer_thread_ = std::thread(&Viewer::Run, viewer_);
    }
    ~System() = default;

    void Run(Frame::Ptr frame);

    void Triangulate(const Frame::Ptr frame) {
        std::vector<cv::KeyPoint> left_kps;
        cv::Mat left_des;
        std::vector<cv::KeyPoint> right_kps;
        cv::Mat right_des;

        auto detector = cv::ORB::create();
        detector->detectAndCompute(frame->left_image_, cv::noArray(), left_kps, left_des);
        detector->detectAndCompute(frame->right_image_, cv::noArray(), right_kps, right_des);
    
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<cv::DMatch> matches;
        matcher.match(left_des, right_des, matches);
        std::vector<cv::Point2d> pts1, pts2;
        for (auto match : matches) {
            pts1.push_back(left_kps[match.queryIdx].pt);
            pts2.push_back(right_kps[match.trainIdx].pt);
        }

        // 对匹配点进行筛选和排序
        std::vector<cv::DMatch> good_matches;
        for (const auto& m : matches) {
            if (m.distance < 50.0) {  // 根据描述子距离筛选
                good_matches.push_back(m);
            }
        }
        matches = good_matches;

        // 清空并重新收集特征点
        pts1.clear();
        pts2.clear();
        std::vector<size_t> pts_idx;  // 存储特征点索引
        for (const auto& match : matches) {
            pts1.push_back(left_kps[match.queryIdx].pt);
            pts2.push_back(right_kps[match.trainIdx].pt);
            pts_idx.push_back(match.queryIdx);  // 保存左图特征点的索引
        }

        frame->points2d_ = pts1;
        frame->left_kps_ = left_kps;
        frame->left_des_ = left_des;

        // 转换到相机坐标系
        std::vector<cv::Point2d> pts1_cam, pts2_cam;
        pts1_cam.resize(pts1.size());
        pts2_cam.resize(pts2.size());
        
        for(size_t i = 0; i < pts1.size(); i++) {
            pts1_cam[i].x = (pts1[i].x - K_.at<double>(0,2)) / K_.at<double>(0,0);
            pts1_cam[i].y = (pts1[i].y - K_.at<double>(1,2)) / K_.at<double>(1,1);
            pts2_cam[i].x = (pts2[i].x - K_.at<double>(0,2)) / K_.at<double>(0,0);
            pts2_cam[i].y = (pts2[i].y - K_.at<double>(1,2)) / K_.at<double>(1,1);
        }

        // 构建投影矩阵：基于当前帧位姿
        // 第一个相机的投影矩阵：从frame->pose提取前3行
        cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
        frame->pose_(cv::Range(0,3), cv::Range::all()).copyTo(P1);

        // 第二个相机的投影矩阵：T01.inv * frame->pose，取前3行
        cv::Mat T_inv = T_01_.inv();
        cv::Mat P2 = (T_01_ * frame->pose_)(cv::Range(0,3), cv::Range::all());

        std::cout<<"P1: "<<P1<<std::endl;
        std::cout<<"P2: "<<P2<<std::endl;

        // 三角化
        cv::Mat points4d;
        cv::triangulatePoints(P1, P2, pts1_cam, pts2_cam, points4d);

        // 转换成非齐次坐标并进行有效性检查
        frame->points3d_.clear();
        frame->points2d_.clear();  // 同时清空2D点
        for (int i = 0; i < points4d.cols; i++) {
            double w = points4d.at<float>(3,i);
            if (std::abs(w) < 1e-10) continue;
            if (std::abs(w) > 1e10) continue;
            
            cv::Point3d p3d(
                points4d.at<float>(0,i) / w,
                points4d.at<float>(1,i) / w,
                points4d.at<float>(2,i) / w
            );
            
            // 深度和范围检查
            if (p3d.z < 0) continue;

            // 只有当3D点有效时，才保存对应的2D点
            frame->points3d_.push_back(p3d);
            frame->points2d_.push_back(pts1[i]);  // 保存对应的2D点
        }

        // 输出调试信息
        std::cout << "Triangulated " << frame->points3d_.size() 
                  << " valid points from " << points4d.cols << " matches" << std::endl;
        std::cout << "2D points size: " << frame->points2d_.size() << std::endl;
    }

private:
    const cv::Mat K_;
    const cv::Mat T_01_;
    Viewer::Ptr viewer_;
    std::thread viewer_thread_;
    // const cv::Mat D_;
};

}