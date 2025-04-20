#pragma once
#include "frame.h"
#include "map.h"
#include <memory>

namespace MVSLAM2 {

class Odometry {
public:
    using Ptr = std::shared_ptr<Odometry>;
    virtual ~Odometry() = default;
    virtual void Extract2d(Frame::Ptr frame);
    virtual void Extract3d(Frame::Ptr frame, Map::Ptr map);
    virtual void Track(Frame::Ptr frame);
    virtual void Pnp(Frame::Ptr frame);
    virtual bool Icp(Frame& frame) { return false; };
};

}