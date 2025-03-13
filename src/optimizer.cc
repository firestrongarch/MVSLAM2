#include "optimizer.h"
#include <ceres/rotation.h>

namespace MVSLAM2 {

template <typename T>
bool PnPSolver::ReprojectionError::operator()(const T* const q, const T* const t, T* residuals) const {
    // 将3D点从世界坐标系转到相机坐标系
    T p[3];
    T point[3] = {T(point3d_.x), T(point3d_.y), T(point3d_.z)};
    ceres::QuaternionRotatePoint(q, point, p);
    p[0] += t[0];
    p[1] += t[1];
    p[2] += t[2];

    // 投影到像素平面
    T px = p[0] / p[2];
    T py = p[1] / p[2];
    
    const T predicted_x = T(K_.at<double>(0,0)) * px + T(K_.at<double>(0,2));
    const T predicted_y = T(K_.at<double>(1,1)) * py + T(K_.at<double>(1,2));

    // 计算残差
    residuals[0] = predicted_x - T(point2d_.x);
    residuals[1] = predicted_y - T(point2d_.y);
    
    return true;
}

bool PnPSolver::Solve(
    const std::vector<cv::Point3d>& points3d,
    const std::vector<cv::Point2d>& points2d,
    const cv::Mat& K,
    cv::Quatd& q_cw,
    cv::Point3d& t_cw) {    // 从 Vec3d 改为 Point3d
    
    CV_Assert(points3d.size() == points2d.size());

    // 配置求解器
    ceres::Problem problem;
    
    // 初始化位姿估计，注意cv::Quatd的存储顺序是[w,x,y,z]
    double q[4] = {q_cw.w, q_cw.x, q_cw.y, q_cw.z};
    double t[3] = {t_cw.x, t_cw.y, t_cw.z};    // 使用.x .y .z访问
    
    // 添加残差块
    for (size_t i = 0; i < points3d.size(); ++i) {
        auto* cost_function = new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3>(
            new ReprojectionError(points3d[i], points2d[i], K));
            
        problem.AddResidualBlock(
            cost_function,
            new ceres::HuberLoss(1.0),
            q,
            t
        );
    }
    
    // 保持四元数单位长度
    problem.SetManifold(q, new ceres::QuaternionManifold);
    
    // 求解
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = 100;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    // 更新结果
    q_cw = cv::Quatd(q[0], q[1], q[2], q[3]); // w,x,y,z顺序
    t_cw = cv::Point3d(t[0], t[1], t[2]);    // 构造 Point3d
    
    return summary.IsSolutionUsable();
}

}  // namespace MVSLAM2