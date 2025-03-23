#include "tracker/ceres_tracker.h"
#include "opencv2/opencv.hpp"
#include <opencv2/core/quaternion.hpp>

namespace MVSLAM2 {
// void CeresTracker::Extract3d(Frame::Ptr frame, Map::Ptr map) {
//     std::vector<cv::KeyPoint> kps1;
//     cv::Mat des1;
//     std::vector<cv::KeyPoint> kps2;
//     cv::Mat des2;

//     // 屏蔽已有特征点的区域
//     cv::Mat mask(frame->left_image_.size(), CV_8UC1, cv::Scalar::all(255));
//     for (const auto &feat : frame->left_kps_){
//         cv::rectangle(mask,
//                     feat.pt - cv::Point2f(10, 10),
//                     feat.pt + cv::Point2f(10, 10),
//                     0,
//                     cv::FILLED);
//     }

//     auto detector = cv::ORB::create(2000);
//     detector->detectAndCompute(frame->left_image_, mask, kps1, des1);
//     detector->detectAndCompute(frame->right_image_, cv::noArray(), kps2, des2);

//     cv::BFMatcher matcher(cv::NORM_HAMMING);
//     std::vector<cv::DMatch> matches;
//     matcher.match(des1, des2, matches);

//     // 匹配点对筛选
//     double min_dist = 10000, max_dist = 0;
//     // 找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
//     for (int i = 0; i < des1.rows; i++) {
//         double dist = matches[i].distance;
//         if (dist < min_dist) min_dist = dist;
//         if (dist > max_dist) max_dist = dist;
//     }
//     // 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
//     std::vector<cv::DMatch> good_matches;
//     for (int i = 0; i < des1.rows; i++) {
//         if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
//             good_matches.push_back(matches[i]);
//         }
//     }


//     std::vector<cv::Point2d> pts1, pts2;
//     for (const auto& match : good_matches) {
//         pts1.push_back(kps1[match.queryIdx].pt);
//         pts2.push_back(kps2[match.trainIdx].pt);
//     }
//     // 转换到相机坐标系
//     std::vector<cv::Point2d> pts1_cam, pts2_cam;
//     frame->Pixel2Camera(pts1, pts2, pts1_cam, pts2_cam);

//     // 构建投影矩阵：基于当前帧位姿
//     // 第一个相机的投影矩阵：从frame->pose提取前3行
//     cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
//     frame->pose_(cv::Range(0,3), cv::Range::all()).copyTo(P1);

//     // 第二个相机的投影矩阵：T01.inv * frame->pose，取前3行
//     cv::Mat T_inv = frame->T_01.inv();
//     cv::Mat P2 = (frame->T_01 * frame->pose_)(cv::Range(0,3), cv::Range::all());

//     // 对每对匹配点进行优化三角化
//     for (size_t i = 0; i < pts1_cam.size(); i++) {
//         // 初始化3D点（使用简单的中点法）
//         double depth = 5.0;  // 假设初始深度为5米
//         cv::Point3d init_point(
//             pts1_cam[i].x * depth,
//             pts1_cam[i].y * depth,
//             depth
//         );

//         double point[3] = {init_point.x, init_point.y, init_point.z};

//         // 配置Ceres问题
//         ceres::Problem problem;
        
//         // 添加左右相机的观测
//         problem.AddResidualBlock(
//             TriangulationError::Create(pts1_cam[i], P1),
//             new ceres::HuberLoss(1.0),
//             point
//         );
//         problem.AddResidualBlock(
//             TriangulationError::Create(pts2_cam[i], P2),
//             new ceres::HuberLoss(1.0),
//             point
//         );

//         // 配置求解器
//         ceres::Solver::Options options;
//         options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
//         options.minimizer_progress_to_stdout = false;
//         options.max_num_iterations = 5;

//         ceres::Solver::Summary summary;
//         ceres::Solve(options, &problem, &summary);

//         // 检查优化结果
//         cv::Point3d p3d(point[0], point[1], point[2]);
        
//         // 有效性检查
//         if (std::isnan(p3d.x) || std::isnan(p3d.y) || std::isnan(p3d.z) ||
//             p3d.z <= 0.1 || p3d.z > 100 ||
//             std::abs(p3d.x) > 100 || std::abs(p3d.y) > 100) {
//             continue;
//         }

//         // 创建地图点和关键点
//         MapPoint::Ptr map_point = std::make_shared<MapPoint>(p3d, MapPoint::next_id++);
//         KeyPoint kp{};
//         kp.map_point = map_point;
//         kp.pt = pts1[i];
//         frame->left_kps_.push_back(kp);
        
//         map->InsertMapPoint(map_point);
//     }
// }

void CeresTracker::Pnp(Frame::Ptr frame) {
    if (!Frame::last_frame_ || frame->left_kps_.empty()) {
        return;
    }

    // 初始化位姿估计：将旋转矩阵转换为四元数
    cv::Mat R = Frame::last_frame_->T_wc(cv::Range(0,3), cv::Range(0,3));
    cv::Mat tvec = Frame::last_frame_->T_wc(cv::Range(0,3), cv::Range(3,4));
    
    // 使用OpenCV的四元数类
    cv::Quatd q = cv::Quatd::createFromRotMat(R);
    
    double pose[7];  // [qw, qx, qy, qz, tx, ty, tz]
    pose[0] = q.w;
    pose[1] = q.x;
    pose[2] = q.y;
    pose[3] = q.z;
    pose[4] = tvec.at<double>(0);
    pose[5] = tvec.at<double>(1);
    pose[6] = tvec.at<double>(2);

    // 配置Ceres求解器
    ceres::Problem problem;

    // 先添加一个参数块，然后再设置流形
    problem.AddParameterBlock(pose, 7);
    
    // 设置流形
    auto pose_manifold = new ceres::ProductManifold<ceres::QuaternionManifold, ceres::EuclideanManifold<3>>();
    problem.SetManifold(pose, pose_manifold);

    // 构建有效特征点索引映射
    std::vector<size_t> valid_indices;
    std::vector<ceres::ResidualBlockId> residual_block_ids;
    
    for (size_t i = 0; i < frame->left_kps_.size(); i++) {
        if (auto mp = frame->left_kps_[i].map_point.lock()) {
            valid_indices.push_back(i);
            cv::Point3d p3d = *mp;
            ceres::CostFunction* cost_function = 
                ReprojectionErrorQuat::Create(frame->left_kps_[i].pt, p3d, frame->K);
            residual_block_ids.push_back(
                problem.AddResidualBlock(
                    cost_function,
                    new ceres::HuberLoss(5.9915),
                    pose  // 现在pose包含了四元数和平移向量
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
                    ReprojectionErrorQuat::Create(
                        frame->left_kps_[valid_indices[i]].pt, 
                        *mp, 
                        frame->K
                    ),
                    nullptr,
                    pose
                );
            }
        }
    }

    // 更新位姿：将四元数转换回旋转矩阵
    cv::Quatd final_q(pose[0], pose[1], pose[2], pose[3]);
    cv::Matx33d rot_mat = final_q.toRotMat3x3();
    // 确保正确转换回cv::Mat
    R = cv::Mat(rot_mat);
    cv::Mat optimized_tvec = (cv::Mat_<double>(3,1) << pose[4], pose[5], pose[6]);

    cv::Mat optimized_pose = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(optimized_pose(cv::Rect(0, 0, 3, 3)));
    optimized_tvec.copyTo(optimized_pose(cv::Rect(3, 0, 1, 3)));

    frame->T_wc = optimized_pose;

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