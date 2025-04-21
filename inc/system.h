#pragma once
#include "frame.h"
#include "map.h"
#include "odometry/ceres_odometry.h"
#include "opencv2/core/core.hpp"
#include "viewer.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <thread>

namespace MVSLAM2 {

class System {
public:
    System(cv::Mat K, cv::Mat T_01_)
    {
        Frame::K = K;
        Frame::T_01 = T_01_;
        viewer_ = std::make_shared<Viewer>();
        viewer_thread_ = std::thread(&Viewer::Run, viewer_);
    }
    ~System() = default;

    void Run(Frame::Ptr frame);

private:
    Viewer::Ptr viewer_;
    std::thread viewer_thread_;
    Map::Ptr map_ = std::make_shared<Map>();
    Odometry::Ptr odom_ = std::make_shared<Odometry>();
    // Tracker::Ptr tracker_ = std::make_shared<CeresTracker>();
};

}