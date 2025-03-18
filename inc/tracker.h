#pragma once
#include <memory>
#include "frame.h"
#include "map.h"
#include "ceres/ceres.h"
#include <ceres/rotation.h>

namespace MVSLAM2 {

// struct TrackerParams {
//     int nfeatures{500};
//     float scale_factor{1.2f};
//     int nlevels{8};
//     // ...其他参数
// };

class Tracker {
public:
    using Ptr = std::shared_ptr<Tracker>;
    virtual ~Tracker() = default;
    virtual void Extract2d(Frame::Ptr frame);
    virtual void Extract3d(Frame::Ptr frame, Map::Ptr map);
    virtual void Track(Frame::Ptr frame);
    virtual void Pnp(Frame::Ptr frame);
    virtual bool Icp(Frame& frame) { return false; };
    // virtual bool Configure(const TrackerParams& params) { return false; };
    
    // // 使用智能指针替代原始指针
    // static std::unique_ptr<Tracker> Create(const std::string& type);
};

class CeresTracker : public Tracker {
public:
    void Pnp(Frame::Ptr frame) override;
private:
    // 定义重投影误差代价函数
    struct ReprojectionError {
        ReprojectionError(const cv::Point2d& point2d, const cv::Point3d& point3d, const cv::Mat& K)
            : point2d_(point2d), point3d_(point3d) {
            fx = K.at<double>(0, 0);
            fy = K.at<double>(1, 1);
            cx = K.at<double>(0, 2);
            cy = K.at<double>(1, 2);
        }

        template <typename T>
        bool operator()(const T* const rvec, const T* const tvec, T* residuals) const {
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

        static ceres::CostFunction* Create(const cv::Point2d& point2d, const cv::Point3d& point3d, const cv::Mat& K) {
            return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3>(
                new ReprojectionError(point2d, point3d, K));
        }

        cv::Point2d point2d_;
        cv::Point3d point3d_;
        double fx, fy, cx, cy;
    };
};

}