#pragma once
#include "opencv2/core/core.hpp"
#include "frame.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <thread>
#include "viewer.h"
#include "map.h"

namespace MVSLAM2 {

class System {
public:
    System(cv::Mat K, cv::Mat T_01_) {
        Frame::K = K;
        Frame::T_01 = T_01_;
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

        auto detector = cv::ORB::create(2000);
        detector->detectAndCompute(frame->left_image_, cv::noArray(), left_kps, left_des);
        detector->detectAndCompute(frame->right_image_, cv::noArray(), right_kps, right_des);
    
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<cv::DMatch> matches;
        matcher.match(left_des, right_des, matches);

        // 对匹配点进行筛选和排序
        std::vector<cv::DMatch> good_matches;
        for (const auto& m : matches) {
            if (m.distance < 50.0) {  // 根据描述子距离筛选
                good_matches.push_back(m);
            }
        }
        matches = good_matches;

        std::vector<cv::Point2d> pts1, pts2;
        for (const auto& match : matches) {
            pts1.push_back(left_kps[match.queryIdx].pt);
            pts2.push_back(right_kps[match.trainIdx].pt);
        }

        // 转换到相机坐标系
        std::vector<cv::Point2d> pts1_cam, pts2_cam;
        frame->Pixel2Camera(pts1, pts2, pts1_cam, pts2_cam);

        // 构建投影矩阵：基于当前帧位姿
        // 第一个相机的投影矩阵：从frame->pose提取前3行
        cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
        frame->pose_(cv::Range(0,3), cv::Range::all()).copyTo(P1);

        // 第二个相机的投影矩阵：T01.inv * frame->pose，取前3行
        cv::Mat T_inv = frame->T_01.inv();
        cv::Mat P2 = (frame->T_01 * frame->pose_)(cv::Range(0,3), cv::Range::all());

        // 三角化
        cv::Mat points4d;
        cv::triangulatePoints(P1, P2, pts1_cam, pts2_cam, points4d);

        // 转换成非齐次坐标并进行有效性检查
        for (int i = 0; i < points4d.cols; i++) {
            double w = points4d.at<float>(3,i);

            cv::Point3d p3d(
                points4d.at<float>(0,i) / w,
                points4d.at<float>(1,i) / w,
                points4d.at<float>(2,i) / w
            );
            
            // 检查NaN值
            if (std::isnan(p3d.x) || std::isnan(p3d.y) || std::isnan(p3d.z)) {
                continue;
            }
            
            // 检查深度是否在合理范围内 (0.1米到100米)
            if (p3d.z <= 0.1 || p3d.z > 100) {
                continue;
            }
            
            // 检查坐标值是否过大
            if (std::abs(p3d.x) > 100 || std::abs(p3d.y) > 100) {
                continue;
            }

            // 只有当3D点有效时，才保存对应的2D点
            MapPoint::Ptr map_point = std::make_shared<MapPoint>(p3d, MapPoint::next_id++);
            KeyPoint kp {};
            kp.map_point = map_point;
            kp.pt = pts1[i];
            frame->left_kps_.push_back(kp);
            
            map_->InsertMapPoint(map_point);
        }
    }

private:
    Viewer::Ptr viewer_;
    std::thread viewer_thread_;
    Map::Ptr map_ = std::make_shared<Map>();
};

}