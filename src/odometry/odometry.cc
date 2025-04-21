#include "odometry/odometry.h"
#include <ceres/manifold.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace MVSLAM2 {

void Odometry::Extract2d(Frame::Ptr frame)
{
    auto detector = cv::ORB::create(2000);
    // detector->detectAndCompute(frame.left_image_, cv::noArray(), frame.left_kps_, frame.left_des_);
}

void Odometry::Extract3d(Frame::Ptr frame, Map::Ptr map)
{

    // 屏蔽已有特征点的区域
    cv::Mat mask(frame->left_image_.size(), CV_8UC1, cv::Scalar::all(255));
    for (const auto& feat : frame->kps) {
        cv::rectangle(mask,
            feat.pt - cv::Point2f(10, 10),
            feat.pt + cv::Point2f(10, 10),
            0,
            cv::FILLED);
    }

    std::vector<cv::KeyPoint> kps1, kps2;
    cv::Mat des1, des2;
    auto detector = cv::ORB::create(2000);
    detector->detectAndCompute(frame->left_image_, mask, kps1, des1);
    detector->detectAndCompute(frame->right_image_, cv::noArray(), kps2, des2);

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<cv::DMatch> matches;
    matcher->match(des1, des2, matches);

    // 匹配点对筛选
    auto min_max = std::minmax_element(matches.begin(), matches.end(),
        [](const cv::DMatch& a, const cv::DMatch& b) {
            return a.distance < b.distance;
        });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;
    // 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < des1.rows; i++) {
        if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
            good_matches.push_back(matches[i]);
        }
    }

    std::vector<cv::Point2d> pts1, pts2;
    std::vector<cv::KeyPoint> kps1_good;
    cv::Mat des1_good;
    for (const auto& match : good_matches) {
        pts1.push_back(kps1[match.queryIdx].pt);
        pts2.push_back(kps2[match.trainIdx].pt);
        kps1_good.push_back(kps1[match.queryIdx]);
        des1_good.push_back(des1.row(match.queryIdx));
    }

    // 转换到相机坐标系
    std::vector<cv::Point2f> pts1_cam, pts2_cam;
    for (size_t i = 0; i < pts1.size(); i++) {
        pts1_cam.push_back(frame->Pixel2Camera(pts1[i]));
        pts2_cam.push_back(frame->Pixel2Camera(pts2[i]));
    }

    // 构建投影矩阵：基于当前帧位姿
    // 第一个相机的投影矩阵：从frame->pose提取前3行
    // 第一个相机的投影矩阵：3x4单位矩阵
    cv::Mat P1 = cv::Mat::eye(3, 4, CV_32F);

    // 第二个相机的投影矩阵：右目相对左目的变换
    cv::Mat P2 = frame->T_01.inv()(cv::Range(0, 3), cv::Range::all());

    // 三角化
    cv::Mat points4d;
    cv::triangulatePoints(P1, P2, pts1_cam, pts2_cam, points4d);

    // 转换成非齐次坐标并进行有效性检查
    for (int i = 0; i < points4d.cols; i++) {
        double w = points4d.at<float>(3, i);

        cv::Point3d p3d(
            points4d.at<float>(0, i) / w,
            points4d.at<float>(1, i) / w,
            points4d.at<float>(2, i) / w);

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
        cv::Mat p3d_mat = (cv::Mat_<double>(4, 1) << p3d.x, p3d.y, p3d.z, 1);
        cv::Mat p3d_world_mat = frame->T_wc * p3d_mat;
        cv::Point3d p3d_world(p3d_world_mat.at<double>(0), p3d_world_mat.at<double>(1), p3d_world_mat.at<double>(2));
        MapPoint::Ptr map_point = std::make_shared<MapPoint>(p3d_world, MapPoint::next_id++);
        KeyPoint kp = kps1_good[i];
        kp.map_point = map_point;
        kp.des = des1_good.row(i);
        frame->kps.push_back(kp);

        map->InsertMapPoint(map_point);
    }
}

void Odometry::Track(Frame::Ptr frame)
{
    // 准备上一帧的特征点和描述子
    auto detector = cv::ORB::create(2000);
    // 检测当前帧的特征点
    std::vector<cv::KeyPoint> kps1, kps2;
    cv::Mat des1, des2;
    for (const auto& kp : Frame::last_frame_->kps) {
        des1.push_back(kp.des);
        kps1.push_back(kp);
    }
    detector->detectAndCompute(frame->left_image_, cv::noArray(), kps2, des2);

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<cv::DMatch> matches;
    matcher->match(des1, des2, matches);
    // 匹配点对筛选
    auto min_max = std::minmax_element(matches.begin(), matches.end(),
        [](const cv::DMatch& a, const cv::DMatch& b) {
            return a.distance < b.distance;
        });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;
    // 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < des1.rows; i++) {
        if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
            good_matches.push_back(matches[i]);
        }
    }

    // 保存匹配结果
    for (const auto& match : good_matches) {
        KeyPoint kp = kps2[match.trainIdx];
        kp.map_point = Frame::last_frame_->kps[match.queryIdx].map_point;
        kp.match = Frame::last_frame_->kps[match.queryIdx].pt;
        kp.des = Frame::last_frame_->kps[match.queryIdx].des;
        frame->kps.push_back(kp);
    }
}

void Odometry::Pnp(Frame::Ptr frame)
{
    cv::Mat T_cw = frame->T_wc.inv();
    cv::Mat R = T_cw(cv::Range(0, 3), cv::Range(0, 3));

    for (int i = 0; i < 4; i++) {
        cv::Mat tvec = T_cw(cv::Range(0, 3), cv::Range(3, 4));
        cv::Mat rvec;
        cv::Rodrigues(R, rvec); // 将旋转矩阵转换为旋转向量
        std::vector<int> inliers;
        std::vector<cv::Point3d> points3d;
        std::vector<cv::Point2f> points2f;
        for (const auto& kp : frame->kps) {
            cv::Point3d p3d = *kp.map_point.lock();
            points3d.push_back(p3d);
            points2f.push_back(kp.pt);
        }
        cv::solvePnPRansac(
            points3d,
            points2f,
            frame->K,
            cv::Mat(),
            rvec,
            tvec,
            true,
            10, // max iterations
            8.0, // reprojection error
            0.99, // confidence
            inliers);
        // 根据inliers筛选特征点
        std::vector<KeyPoint> filtered_kps;
        for (auto idx : inliers) {
            filtered_kps.push_back(frame->kps[idx]);
        }
        frame->kps = filtered_kps;

        cv::Rodrigues(rvec, R);
        cv::Mat T_cw_new = cv::Mat::eye(4, 4, CV_64F);
        R.copyTo(T_cw_new(cv::Rect(0, 0, 3, 3)));
        tvec.copyTo(T_cw_new(cv::Rect(3, 0, 1, 3)));
        frame->T_wc = T_cw_new.inv();
    }
}

}