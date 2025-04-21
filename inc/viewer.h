#pragma once
#include "frame.h"
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <pangolin/pangolin.h>

namespace MVSLAM2 {

class Viewer {
public:
    using Ptr = std::shared_ptr<Viewer>;
    // Viewer() = default;
    void ShowWithViz();
    void Run();
    void AddTrajectoryPose(const cv::Mat& pose);
    // void AddMapPoint(const Eigen::Vector3d &point);
    void RenderMapPoint();
    // void SetMap(const Map::Ptr map);
    void DrawFrame(Frame::Ptr frame);
    void DrawMatches(Frame::Ptr frame);
    void DrawReprojection(Frame::Ptr frame);

private:
    void Render();
    void RenderKf();
    std::vector<cv::Vec3d> traj_VO_;
    std::vector<cv::Mat> poses_; /// pose
    std::vector<cv::Vec3d> point_cloud_;
    // Map::Ptr map_;
};

}
