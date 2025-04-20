#pragma once
#include "ceres/ceres.h"
#include "odometry/odometry.h"
#include <ceres/rotation.h>

namespace MVSLAM2 {
class CeresOdom : public Odometry {
public:
    // void Extract3d(Frame::Ptr frame, Map::Ptr map) override;
    void Pnp(Frame::Ptr frame) override;

private:
    // 定义重投影误差代价函数
    struct ReprojectionError {
        ReprojectionError(const cv::Point2d& point2d, const cv::Point3d& point3d, const cv::Mat& K)
            : point2d_(point2d)
            , point3d_(point3d)
        {
            fx = K.at<double>(0, 0);
            fy = K.at<double>(1, 1);
            cx = K.at<double>(0, 2);
            cy = K.at<double>(1, 2);
        }

        template <typename T>
        bool operator()(const T* const rvec, const T* const tvec, T* residuals) const
        {
            // 旋转向量转旋转矩阵
            T R[9];
            ceres::AngleAxisToRotationMatrix(rvec, R);

            // 3D点投影
            T p[3];
            p[0] = R[0] * T(point3d_.x) + R[1] * T(point3d_.y) + R[2] * T(point3d_.z) + tvec[0];
            p[1] = R[3] * T(point3d_.x) + R[4] * T(point3d_.y) + R[5] * T(point3d_.z) + tvec[1];
            p[2] = R[6] * T(point3d_.x) + R[7] * T(point3d_.y) + R[8] * T(point3d_.z) + tvec[2];

            // 投影到像素平面
            T u = fx * p[0] / p[2] + cx;
            T v = fy * p[1] / p[2] + cy;

            // 计算重投影误差
            residuals[0] = u - T(point2d_.x);
            residuals[1] = v - T(point2d_.y);

            return true;
        }

        static ceres::CostFunction* Create(const cv::Point2d& point2d, const cv::Point3d& point3d, const cv::Mat& K)
        {
            return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3>(
                new ReprojectionError(point2d, point3d, K));
        }

        cv::Point2d point2d_;
        cv::Point3d point3d_;
        double fx, fy, cx, cy;
    };

    // 添加三角化误差模型
    struct TriangulationError {
        TriangulationError(const cv::Point2d& point2d, const cv::Mat& P)
            : point2d_(point2d)
        {
            memcpy(P_, P.ptr<double>(), 12 * sizeof(double));
        }

        template <typename T>
        bool operator()(const T* const point3d, T* residuals) const
        {
            // 投影
            T p[3];
            p[0] = P_[0] * point3d[0] + P_[1] * point3d[1] + P_[2] * point3d[2] + P_[3];
            p[1] = P_[4] * point3d[0] + P_[5] * point3d[1] + P_[6] * point3d[2] + P_[7];
            p[2] = P_[8] * point3d[0] + P_[9] * point3d[1] + P_[10] * point3d[2] + P_[11];

            // 计算重投影误差
            T u = p[0] / p[2];
            T v = p[1] / p[2];
            residuals[0] = u - T(point2d_.x);
            residuals[1] = v - T(point2d_.y);
            return true;
        }

        static ceres::CostFunction* Create(const cv::Point2d& point2d, const cv::Mat& P)
        {
            return new ceres::AutoDiffCostFunction<TriangulationError, 2, 3>(
                new TriangulationError(point2d, P));
        }

        cv::Point2d point2d_;
        double P_[12]; // 投影矩阵
    };

    // 四元数版本的重投影误差
    struct ReprojectionErrorQuat {
        ReprojectionErrorQuat(const cv::Point2d& point2d, const cv::Point3d& point3d, const cv::Mat& K)
            : point2d_(point2d)
            , point3d_(point3d)
        {
            fx = K.at<double>(0, 0);
            fy = K.at<double>(1, 1);
            cx = K.at<double>(0, 2);
            cy = K.at<double>(1, 2);
        }

        template <typename T>
        bool operator()(const T* const pose, T* residuals) const
        {
            const T* q = pose; // 四元数 [qw, qx, qy, qz]
            const T* t = pose + 4; // 平移向量 [tx, ty, tz]

            // 使用四元数进行旋转
            T p[3];
            T pt[3] = { T(point3d_.x), T(point3d_.y), T(point3d_.z) };
            ceres::QuaternionRotatePoint(q, pt, p);

            // 添加平移
            p[0] += t[0];
            p[1] += t[1];
            p[2] += t[2];

            // 投影到像素平面
            T u = fx * p[0] / p[2] + cx;
            T v = fy * p[1] / p[2] + cy;

            residuals[0] = u - T(point2d_.x);
            residuals[1] = v - T(point2d_.y);

            return true;
        }

        static ceres::CostFunction* Create(const cv::Point2d& point2d, const cv::Point3d& point3d, const cv::Mat& K)
        {
            return new ceres::AutoDiffCostFunction<ReprojectionErrorQuat, 2, 7>(
                new ReprojectionErrorQuat(point2d, point3d, K));
        }

        cv::Point2d point2d_;
        cv::Point3d point3d_;
        double fx, fy, cx, cy;
    };
};

}