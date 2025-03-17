#include "frame.h"

namespace MVSLAM2 {

int MapPoint::next_id = 0;

Frame::Ptr Frame::last_frame_ = nullptr;
Frame::Ptr Frame::kf = nullptr;
cv::Mat Frame::K;
cv::Mat Frame::T_01;

// 2d = K * [R|t] * 3d(世界坐标系)
// 2d = K * 3d(相机坐标系)
void Frame::Pixel2Camera(const std::vector<cv::Point2d>& pts1, const std::vector<cv::Point2d>& pts2,
    std::vector<cv::Point2d>& pts1_cam, std::vector<cv::Point2d>& pts2_cam) {
    
    pts1_cam.resize(pts1.size());
    pts2_cam.resize(pts2.size());

    for(size_t i = 0; i < pts1.size(); i++) {
        pts1_cam[i].x = (pts1[i].x - K.at<double>(0,2)) / K.at<double>(0,0);
        pts1_cam[i].y = (pts1[i].y - K.at<double>(1,2)) / K.at<double>(1,1);
        pts2_cam[i].x = (pts2[i].x - K.at<double>(0,2)) / K.at<double>(0,0);
        pts2_cam[i].y = (pts2[i].y - K.at<double>(1,2)) / K.at<double>(1,1);
    }
}

}