#pragma once
#include "opencv2/core/core.hpp"
#include "frame.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <thread>
#include "viewer.h"
#include "map.h"
#include "tracker.h"

namespace MVSLAM2 {

class System {
public:
    System(cv::Mat K, cv::Mat T_01_) {
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
    // Tracker::Ptr tracker_ = std::make_shared<Tracker>();
    Tracker::Ptr tracker_ = std::make_shared<CeresTracker>();
};

}