#pragma once
#include <memory>
#include <semaphore>
#include "frame.h"

namespace MVSLAM2 {

class Map {
public:
    using Ptr = std::shared_ptr<Map>;
    using KeyFrames = std::unordered_map<unsigned long, std::shared_ptr<Frame>>;
    using MapPoints = std::unordered_map<unsigned long, std::shared_ptr<MapPoint>>;
    Map() = default;

    void InsertKeyFrame(std::shared_ptr<Frame> key_frame);
    void InsertMapPoint(std::shared_ptr<MapPoint> map_point);
    MapPoints GetAllMapPoints();
    KeyFrames GetAllKeyFrames();

    void ShowCurrentKeyFrame();

    void RemoveOutliers();

    std::binary_semaphore semaphore_{0};
    std::binary_semaphore backend_finished_{1};

    std::shared_ptr<Frame> current_keyframe_{nullptr};

private:
    MapPoints all_map_points_;
    MapPoints active_map_point_;

    KeyFrames all_key_frames_;
    KeyFrames active_key_frames_;

    unsigned int num_active_key_frames_;
};

}