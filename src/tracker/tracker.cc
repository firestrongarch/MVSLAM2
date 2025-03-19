#include "tracker/tracker.h"
#include <ceres/manifold.h>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/opencv.hpp>

namespace MVSLAM2 {

void Tracker::Extract2d(Frame::Ptr frame) {
    auto detector = cv::ORB::create(2000);
    // detector->detectAndCompute(frame.left_image_, cv::noArray(), frame.left_kps_, frame.left_des_);
}

void Tracker::Extract3d(Frame::Ptr frame, Map::Ptr map) {
    std::vector<cv::KeyPoint> kps1;
    cv::Mat des1;
    std::vector<cv::KeyPoint> kps2;
    cv::Mat des2;

    // 屏蔽已有特征点的区域
    cv::Mat mask(frame->left_image_.size(), CV_8UC1, cv::Scalar::all(255));
    for (const auto &feat : frame->left_kps_){
        cv::rectangle(mask,
                    feat.pt - cv::Point2f(10, 10),
                    feat.pt + cv::Point2f(10, 10),
                    0,
                    cv::FILLED);
    }

    auto detector = cv::ORB::create(2000);
    detector->detectAndCompute(frame->left_image_, mask, kps1, des1);
    detector->detectAndCompute(frame->right_image_, cv::noArray(), kps2, des2);

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<cv::DMatch> matches;
    matcher->match(des1, des2, matches);

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


    std::vector<cv::Point2d> pts1, pts2;
    for (const auto& match : good_matches) {
        pts1.push_back(kps1[match.queryIdx].pt);
        pts2.push_back(kps2[match.trainIdx].pt);
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
        
        map->InsertMapPoint(map_point);
    }
}

void Tracker::Track(Frame::Ptr frame) {
    // 准备上一帧的特征点和描述子
    auto detector = cv::ORB::create(2000);
    std::vector<cv::KeyPoint> kps1;
    for (const auto& kp : Frame::last_frame_->left_kps_) {
        cv::KeyPoint cvkp;
        cvkp.pt = kp.pt;
        kps1.push_back(cvkp);
    }
    
    if (kps1.empty()) {
        std::cout << "Warning: No keypoints in last frame!" << std::endl;
        return;
    }

    cv::Mat des1;
    detector->compute(Frame::last_frame_->left_image_, kps1, des1);

    // 检测当前帧的特征点
    std::vector<cv::KeyPoint> kps2;
    cv::Mat des2;
    detector->detectAndCompute(frame->left_image_, cv::noArray(), kps2, des2);

    if (kps2.empty() || des1.empty() || des2.empty()) {
        std::cout << "Warning: No keypoints or descriptors!" << std::endl;
        return;
    }

    // 使用交叉匹配以提高匹配质量
    std::vector<std::vector<cv::DMatch>> matches1to2, matches2to1;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    matcher->knnMatch(des1, des2, matches1to2, 2);
    matcher->knnMatch(des2, des1, matches2to1, 2);

    // 交叉检查 + ratio test
    std::vector<cv::DMatch> good_matches;
    const float ratio_thresh = 0.8f;  // 放宽比率测试阈值

    for (size_t i = 0; i < matches1to2.size(); i++) {
        if (matches1to2[i].size() < 2) continue;
        
        const cv::DMatch& m = matches1to2[i][0];
        const cv::DMatch& n = matches1to2[i][1];
        
        // ratio test
        if (m.distance > ratio_thresh * n.distance) continue;

        // 交叉检查
        bool cross_check_ok = false;
        for (const auto& backward_matches : matches2to1[m.trainIdx]) {
            if (backward_matches.trainIdx == m.queryIdx) {
                cross_check_ok = true;
                break;
            }
        }
        
        if (cross_check_ok) {
            good_matches.push_back(m);
        }
    }

    if (good_matches.size() < 10) {
        std::cout << "Warning: Too few good matches: " << good_matches.size() << std::endl;
        return;
    }

    // 保存匹配结果
    frame->left_kps_.clear();
    for (const auto& match : good_matches) {
        KeyPoint kp;
        kp.pt = kps2[match.trainIdx].pt;
        kp.map_point = Frame::last_frame_->left_kps_[match.queryIdx].map_point;
        kp.match = Frame::last_frame_->left_kps_[match.queryIdx].pt;
        frame->left_kps_.push_back(kp);
    }

    std::cout << "Successfully matched " << frame->left_kps_.size() << " points" << std::endl;
}

void Tracker::Pnp(Frame::Ptr frame) {
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
    cv::solvePnP(
        points3d,
        points2d,
        frame->K,
        cv::Mat(),
        rvec,
        tvec,
        true
    );
    // cv::solvePnPRansac(
    //     points3d,
    //     points2d,
    //     frame->K,
    //     cv::Mat(),
    //     rvec,
    //     tvec,
    //     true,
    //     100, // max iterations
    //     8.0, // reprojection error
    //     0.99, // confidence
    //     inliers
    // );
    cv::Rodrigues(rvec, R);
    cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(pose(cv::Rect(0, 0, 3, 3)));
    tvec.copyTo(pose(cv::Rect(3, 0, 1, 3)));
    frame->pose_ = pose;
}

}