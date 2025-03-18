#pragma once
#include <memory>
#include "frame.h"
#include "map.h"

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

}