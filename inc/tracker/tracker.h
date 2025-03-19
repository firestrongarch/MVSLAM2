#pragma once
#include <memory>
#include "frame.h"
#include "map.h"

namespace MVSLAM2 {

class Tracker {
public:
    using Ptr = std::shared_ptr<Tracker>;
    virtual ~Tracker() = default;
    virtual void Extract2d(Frame::Ptr frame);
    virtual void Extract3d(Frame::Ptr frame, Map::Ptr map);
    virtual void Track(Frame::Ptr frame);
    virtual void Pnp(Frame::Ptr frame);
    virtual bool Icp(Frame& frame) { return false; };
};

}