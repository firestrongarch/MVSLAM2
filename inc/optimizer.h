#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/core/quaternion.hpp>
#include <ceres/ceres.h>

namespace MVSLAM2 {

class PnPSolver {
public:
    PnPSolver() = default;
    
    /**
     * @brief 求解PnP问题
     * @param points3d 3D点集
     * @param points2d 对应的2D投影点
     * @param K 相机内参矩阵
     * @param q_cw 输出四元数 (camera to world)
     * @param t_cw 输出平移向量
     * @return 是否求解成功
     */
    bool Solve(
        const std::vector<cv::Point3d>& points3d,
        const std::vector<cv::Point2d>& points2d,
        const cv::Mat& K,
        cv::Quatd& q_cw,
        cv::Point3d& t_cw    // 从 Vec3d 改为 Point3d
    );

private:
    struct ReprojectionError {
        ReprojectionError(const cv::Point3d& point3d, 
                         const cv::Point2d& point2d,
                         const cv::Mat& K)
            : point3d_(point3d), point2d_(point2d) {
            K.copyTo(K_);
        }

        template <typename T>
        bool operator()(const T* const q, const T* const t, T* residuals) const;

        const cv::Point3d point3d_;
        const cv::Point2d point2d_;
        cv::Mat K_;
    };
};

}  // namespace MVSLAM2