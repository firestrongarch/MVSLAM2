#include "tracker/tracker.h"
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

    // 保存匹配的特征点和对应的3D点
    for (const auto& match : good_matches) {
        KeyPoint kp = kps2[match.trainIdx];
        kp.map_point = Frame::last_frame_->left_kps_[match.queryIdx].map_point;
        kp.match = std::make_shared<KeyPoint>(Frame::last_frame_->left_kps_[match.queryIdx]);
        frame->left_kps_.push_back(kp);
    }
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

void CeresTracker::Extract3d(Frame::Ptr frame, Map::Ptr map) {
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

    // 对每对匹配点进行优化三角化
    for (size_t i = 0; i < pts1_cam.size(); i++) {
        // 初始化3D点（使用简单的中点法）
        double depth = 5.0;  // 假设初始深度为5米
        cv::Point3d init_point(
            pts1_cam[i].x * depth,
            pts1_cam[i].y * depth,
            depth
        );

        double point[3] = {init_point.x, init_point.y, init_point.z};

        // 配置Ceres问题
        ceres::Problem problem;
        
        // 添加左右相机的观测
        problem.AddResidualBlock(
            TriangulationError::Create(pts1_cam[i], P1),
            new ceres::HuberLoss(1.0),
            point
        );
        problem.AddResidualBlock(
            TriangulationError::Create(pts2_cam[i], P2),
            new ceres::HuberLoss(1.0),
            point
        );

        // 配置求解器
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations = 5;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // 检查优化结果
        cv::Point3d p3d(point[0], point[1], point[2]);
        
        // 有效性检查
        if (std::isnan(p3d.x) || std::isnan(p3d.y) || std::isnan(p3d.z) ||
            p3d.z <= 0.1 || p3d.z > 100 ||
            std::abs(p3d.x) > 100 || std::abs(p3d.y) > 100) {
            continue;
        }

        // 创建地图点和关键点
        MapPoint::Ptr map_point = std::make_shared<MapPoint>(p3d, MapPoint::next_id++);
        KeyPoint kp{};
        kp.map_point = map_point;
        kp.pt = pts1[i];
        frame->left_kps_.push_back(kp);
        
        map->InsertMapPoint(map_point);
    }
}

void CeresTracker::Pnp(Frame::Ptr frame) {
    if (!Frame::last_frame_ || frame->left_kps_.empty()) {
        return;
    }

    // 初始化位姿估计
    cv::Mat R = Frame::last_frame_->pose_(cv::Range(0,3), cv::Range(0,3));
    cv::Mat tvec = Frame::last_frame_->pose_(cv::Range(0,3), cv::Range(3,4));
    cv::Mat rvec;
    cv::Rodrigues(R, rvec);

    // 配置Ceres求解器
    ceres::Problem problem;
    double pose[6];  // 前3个为旋转向量，后3个为平移向量
    pose[0] = rvec.at<double>(0);
    pose[1] = rvec.at<double>(1);
    pose[2] = rvec.at<double>(2);
    pose[3] = tvec.at<double>(0);
    pose[4] = tvec.at<double>(1);
    pose[5] = tvec.at<double>(2);

    // 构建有效特征点索引映射
    std::vector<size_t> valid_indices;
    std::vector<ceres::ResidualBlockId> residual_block_ids;
    
    for (size_t i = 0; i < frame->left_kps_.size(); i++) {
        if (auto mp = frame->left_kps_[i].map_point.lock()) {
            valid_indices.push_back(i);
            cv::Point3d p3d = *mp;
            ceres::CostFunction* cost_function = 
                ReprojectionError::Create(frame->left_kps_[i].pt, p3d, frame->K);
            residual_block_ids.push_back(
                problem.AddResidualBlock(
                    cost_function,
                    new ceres::HuberLoss(5.9915),
                    pose,
                    pose + 3
                )
            );
        }
    }

    if (valid_indices.empty()) {
        return;
    }

    // 迭代优化
    const double chi2_th = 5.991;
    int cnt_outliers = 0;
    int num_iterations = 4;
    std::vector<bool> outlier_flags(valid_indices.size(), false);
    
    for (int iter = 0; iter < num_iterations; iter++) {
        // 配置求解器
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // 改用更稳定的求解器
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations = 10;
        options.function_tolerance = 1e-4;     // 添加收敛条件
        options.gradient_tolerance = 1e-4;
        options.parameter_tolerance = 1e-4;

        // 求解优化问题
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (!summary.IsSolutionUsable()) {
            std::cout << "PnP optimization failed!" << std::endl;
            return;
        }

        // 检查每个观测的误差
        cnt_outliers = 0;
        for (size_t i = 0; i < residual_block_ids.size(); i++) {
            double residuals[2];
            if (!problem.EvaluateResidualBlock(residual_block_ids[i], false, nullptr, residuals, nullptr)) {
                outlier_flags[i] = true;
                cnt_outliers++;
                continue;
            }

            double chi2 = residuals[0] * residuals[0] + residuals[1] * residuals[1];
            outlier_flags[i] = (chi2 > chi2_th);
            if (outlier_flags[i]) cnt_outliers++;

            // 倒数第二轮时移除鲁棒核函数
            if (iter == num_iterations - 2) {
                auto mp = frame->left_kps_[valid_indices[i]].map_point.lock();
                if (!mp) continue;
                
                problem.RemoveResidualBlock(residual_block_ids[i]);
                residual_block_ids[i] = problem.AddResidualBlock(
                    ReprojectionError::Create(
                        frame->left_kps_[valid_indices[i]].pt, 
                        *mp, 
                        frame->K
                    ),
                    nullptr,
                    pose,
                    pose + 3
                );
            }
        }
    }

    // 更新位姿
    cv::Mat optimized_rvec = (cv::Mat_<double>(3,1) << pose[0], pose[1], pose[2]);
    cv::Rodrigues(optimized_rvec, R);
    cv::Mat optimized_tvec = (cv::Mat_<double>(3,1) << pose[3], pose[4], pose[5]);

    cv::Mat optimized_pose = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(optimized_pose(cv::Rect(0, 0, 3, 3)));
    optimized_tvec.copyTo(optimized_pose(cv::Rect(3, 0, 1, 3)));

    frame->pose_ = optimized_pose;

    // 只保留内点
    std::vector<KeyPoint> inlier_kps;
    for (size_t i = 0; i < outlier_flags.size(); i++) {
        if (!outlier_flags[i]) {
            inlier_kps.push_back(frame->left_kps_[valid_indices[i]]);
        }
    }
    frame->left_kps_ = inlier_kps;
}

}