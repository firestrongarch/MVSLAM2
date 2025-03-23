#pragma once
#include <memory>
#include <opencv2/core/core.hpp>
#include <vector>

namespace MVSLAM2 {
class MapPoint : public cv::Point3d {
public:
    using Ptr = std::shared_ptr<MapPoint>;
    MapPoint() = delete;
    MapPoint(const cv::Point3d& p , const int& id) : cv::Point3d(p), id(id) {}

    const int id;
    static int next_id;
    bool is_outlier = false;
};

class KeyPoint : public cv::KeyPoint {
public:
    using Ptr = std::shared_ptr<KeyPoint>;
    KeyPoint() = default;
    KeyPoint(const cv::KeyPoint& kp) : cv::KeyPoint(kp) {}


    std::weak_ptr<MapPoint> map_point;
    cv::Mat des;
    cv::Point2f match;
    
    bool is_outlier = false;
};

struct Frame {
    using Ptr = std::shared_ptr<Frame>;

    const cv::Mat left_image_;
    const cv::Mat right_image_;
    const double timestamp_;
    const int id;

    std::vector<KeyPoint> left_kps_;

    cv::Mat T_wc = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat T_ww = cv::Mat::eye(4, 4, CV_64F);

    static Ptr last_frame_;
    static Ptr kf;
    static cv::Mat K;
    static cv::Mat T_01;

    static cv::Point2d Pixel2Camera(const cv::Point2d& p2d);

    cv::Point2d World2Pixel(const cv::Point3d& p3d);
    cv::Point3d Pixel2World(const cv::Point2d& p2d);
};

}
