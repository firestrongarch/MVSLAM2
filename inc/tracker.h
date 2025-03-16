#pragma once
#include <memory>
#include "frame.h"

namespace MVSLAM2 {

struct TrackerParams {
    int nfeatures{500};
    float scale_factor{1.2f};
    int nlevels{8};
    // ...其他参数
};

class Tracker {
public:
    virtual ~Tracker() = default;
    virtual bool Extract2d(Frame& frame) { return false; };
    virtual bool Extract3d(Frame& frame) { return false; };
    virtual bool Track(Frame& frame) { return false; };
    virtual bool Pnp(Frame& frame) { return false; };
    virtual bool Icp(Frame& frame) { return false; };
    virtual bool Configure(const TrackerParams& params) { return false; };
    
    // 使用智能指针替代原始指针
    static std::unique_ptr<Tracker> Create(const std::string& type);
};

}