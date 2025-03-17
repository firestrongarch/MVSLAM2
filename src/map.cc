#include "map.h"

namespace MVSLAM2 {

void Map::InsertKeyFrame(std::shared_ptr<Frame> key_frame) {

    all_key_frames_.insert({key_frame->id, key_frame});
    // for(auto &kp: key_frame->left_kps_){
    //     kp->map_point.lock()->observers_.push_back({key_frame,kp});
    // }
}

void Map::InsertMapPoint(std::shared_ptr<MapPoint> map_point)
{
    if (all_map_points_.find(map_point->id) == all_map_points_.end()){
        all_map_points_.insert(make_pair(map_point->id, map_point));
    }
}
}