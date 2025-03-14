#pragma once
#include "opencv2/core/core.hpp"
#include "frame.h"

namespace MVSLAM2 {

class System {

public:
    System(/* args */) = default;
    ~System() = default;

    void Run(Frame frame);

};

}