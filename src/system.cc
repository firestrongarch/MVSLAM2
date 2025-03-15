#include "system.h"
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

namespace MVSLAM2 {

void System::Run(Frame frame) {
    // cv::imshow("left", frame.left_image_);
    // cv::imshow("right", frame.right_image_);

    auto detector = cv::ORB::create();
    detector->detectAndCompute(frame.left_image_, cv::noArray(), frame.left_kps_, frame.left_des_);
    detector->detectAndCompute(frame.right_image_, cv::noArray(), frame.right_kps_, frame.right_des_);

    cv::Mat out1 = frame.left_image_.clone();
    cv::Mat out2 = frame.right_image_.clone();
    cv::cvtColor(out1, out1, cv::COLOR_GRAY2BGR);
    cv::cvtColor(out2, out2, cv::COLOR_GRAY2BGR);

    for (auto kp : frame.left_kps_) {
        cv::Point2f pt1,pt2;
        pt1.x=kp.pt.x-5;
        pt1.y=kp.pt.y-5;
        pt2.x=kp.pt.x+5;
        pt2.y=kp.pt.y+5;
        cv::rectangle(out1, pt1, pt2, cv::Scalar(0, 255, 0));
        cv::circle(out1, kp.pt, 2, cv::Scalar(0, 255, 0), cv::FILLED);
    }
    for (auto kp : frame.right_kps_) {
        cv::Point2f pt1,pt2;
        pt1.x=kp.pt.x-5;
        pt1.y=kp.pt.y-5;
        pt2.x=kp.pt.x+5;
        pt2.y=kp.pt.y+5;
        cv::rectangle(out2, pt1, pt2, cv::Scalar(0, 255, 0));
        cv::circle(out2, kp.pt, 2, cv::Scalar(0, 255, 0), cv::FILLED);
    }

    cv::imshow("left", out1);
    cv::imshow("right", out2);

}

}