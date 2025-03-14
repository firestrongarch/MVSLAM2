#include "system.h"
#include "dataset_reader/factories.hpp"
#include "dataset_reader/kitti_dataset.hpp"
#include <Poco/Environment.h>
#include <Poco/Path.h>
#include <opencv2/opencv.hpp>
#include <thread>

int main (int argc, char** argv){
    MVSLAM2::System SLAM;
    // 注册 KittiDataset 类型
    fsa::DatasetFactory::register_type<fsa::KittiDataset>("kitti");
    auto dataset = fsa::DatasetFactory::create("kitti", "~/datasets/KITTI/"+std::string(argv[1]));
    cv::Mat imLeft, imRight;
    // sensor_msgs::msg::Image::SharedPtr imLeft_msg, imRight_msg;
    while (dataset->has_next()) {
        auto frame = dataset->load_next();
        imLeft = cv::imread(frame->left_image_path, cv::IMREAD_GRAYSCALE);  // 直接读取为灰度图
        imRight = cv::imread(frame->right_image_path, cv::IMREAD_GRAYSCALE);  // 直接读取为灰度图

        SLAM.Run({
            .left_image_ = imLeft,
            .right_image_ = imRight,
            .timestamp_ = frame->timestamp
        });

        using namespace std::chrono;
        std::this_thread::sleep_for(30ms);
    }
}