#pragma once
#include "opencv2/core/core.hpp"
#include "frame.h"
#include <opencv2/core/mat.hpp>

namespace MVSLAM2 {

class System {
public:
    System(cv::Mat K, cv::Mat T_01_) : K_(K), T_01_(T_01_) {}
    ~System() = default;

    void Run(Frame frame);


private:
    const cv::Mat K_;
    const cv::Mat T_01_;
    // const cv::Mat D_;
};

}