#include "viewer.h"
#include "frame.h"
#include <opencv2/core/types.hpp>
#include <opencv2/viz.hpp>
#include <print>

namespace MVSLAM2 {

// ...existing code...

void Viewer::ShowWithViz()
{
    // 创建3D可视化窗口
    // cv::viz::Viz3d viz_window("Visual Odometry");

    // 设置窗口背景颜色和初始相机视角
    viz_window.setBackgroundColor(cv::viz::Color::white());

    // 创建坐标系部件
    viz_window.showWidget("Coordinate", cv::viz::WCoordinateSystem());

    // 主循环
    while (!viz_window.wasStopped()) {
        // viz_window.setViewerPose(cv::Affine3d(
        //     cv::Mat::eye(3, 3, CV_64F),
        //     cv::Vec3d(x, y, z)));

        // 绘制轨迹
        if (!traj_VO_.empty()) {
            std::vector<cv::Point3d> points;
            for (const auto& p : traj_VO_) {
                points.emplace_back(p[0], p[1], p[2]);
            }

            cv::viz::WCloud cloud_widget(points, cv::viz::Color::blue());
            viz_window.showWidget("Trajectory_Points", cloud_widget);

            cv::viz::WPolyLine trajectory_widget(points, cv::viz::Color::red());
            viz_window.showWidget("Trajectory_Lines", trajectory_widget);

            cv::viz::WCameraPosition cpw_curr(0.5);
            viz_window.showWidget("Camera", cpw_curr,
                cv::Affine3d(cv::Mat::eye(3, 3, CV_64F),
                    cv::Vec3d(points.back().x, points.back().y, points.back().z)));
        }

        viz_window.spinOnce(30, true);
    }
}

void Viewer::Run()
{
    pangolin::CreateWindowAndBind("MVSLAM2", 1280, 720);
    glEnable(GL_DEPTH_TEST);

    /// Issue specific OpenGl we might need
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1280, 720, 5000, 5000, 640, 360, 1.0, 1e10),
        pangolin::ModelViewLookAt(0, 1000, 0, 0, 0, 0, pangolin::AxisZ));

    // Create Interactive View in window
    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                                .SetHandler(&handler);

    while (!pangolin::ShouldQuit()) {
        // Clear entire screen
        glClearColor(1, 1, 1, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);

        Render();
        // RenderKf();
        // RenderMapPoint();

        // Swap frames and Process Events
        pangolin::FinishFrame();
    }
}

void Viewer::Render()
{
    if (traj_VO_.empty()) {
        return;
    }

    // 绘制轨迹点
    glPointSize(5);
    glBegin(GL_POINTS);
    for (auto& p : traj_VO_) {
        glColor3f(0, 1, 0);
        glVertex3d(p[0], p[1], p[2]);
    }
    glEnd();

    // 绘制轨迹线
    glLineWidth(2);
    glBegin(GL_LINE_STRIP);
    for (auto& p : traj_VO_) {
        glColor3f(0, 0, 1); // 蓝色轨迹线
        glVertex3d(p[0], p[1], p[2]);
    }
    glEnd();
}

void Viewer::RenderKf()
{
    // glPointSize(5);
    // glBegin(GL_POINTS);
    // auto const Kfs = map_->GetAllKeyFrames();
    // for (auto &kf : Kfs){
    //     auto p = kf.second->Pose().inverse().translation();
    //     glColor3f(0, 1, 0);
    //     glVertex3d(p.x(), p.y(), p.z());
    // }
    // glEnd();
}

void Viewer::AddTrajectoryPose(const cv::Mat& pose)
{
    // static double add;
    // add += 5;
    cv::Mat t = pose(cv::Range(0, 3), cv::Range(3, 4));
    cv::Vec3d t_d;
    t_d[0] = static_cast<float>(t.at<double>(0, 0));
    t_d[1] = static_cast<float>(t.at<double>(1, 0));
    t_d[2] = static_cast<float>(t.at<double>(2, 0));
    // poses_.emplace_back(pose.cast<float>());
    traj_VO_.emplace_back(t_d);
}

void Viewer::DrawFrame(Frame::Ptr frame)
{
    cv::Mat out1 = frame->left_image_.clone();
    cv::Mat out2 = frame->right_image_.clone();
    cv::cvtColor(out1, out1, cv::COLOR_GRAY2BGR);
    cv::cvtColor(out2, out2, cv::COLOR_GRAY2BGR);

    for (auto kp : frame->kps) {
        cv::Point2f pt1, pt2;
        pt1.x = kp.pt.x - 5;
        pt1.y = kp.pt.y - 5;
        pt2.x = kp.pt.x + 5;
        pt2.y = kp.pt.y + 5;
        cv::rectangle(out1, pt1, pt2, cv::Scalar(0, 255, 0));
        cv::circle(out1, kp.pt, 2, cv::Scalar(0, 255, 0), cv::FILLED);
    }
    // for (auto kp : frame->right_kps_) {
    //     cv::Point2f pt1,pt2;
    //     pt1.x=kp.pt.x-5;
    //     pt1.y=kp.pt.y-5;
    //     pt2.x=kp.pt.x+5;
    //     pt2.y=kp.pt.y+5;
    //     cv::rectangle(out2, pt1, pt2, cv::Scalar(0, 255, 0));
    //     cv::circle(out2, kp.pt, 2, cv::Scalar(0, 255, 0), cv::FILLED);
    // }
}

void Viewer::DrawMatches(Frame::Ptr frame)
{
    cv::Mat out1 = frame->left_image_.clone();
    cv::cvtColor(out1, out1, cv::COLOR_GRAY2BGR);

    // 画出特征跟踪结果
    for (auto kp : frame->kps) {
        cv::circle(out1, kp.pt, 4, cv::Scalar(0, 255, 0), cv::FILLED);
        cv::line(out1, kp.pt, kp.match, cv::Scalar(0, 0, 255));
    }

    cv::imshow("matches", out1);
}

void Viewer::DrawReprojection(Frame::Ptr frame)
{
    cv::Mat out1 = frame->left_image_.clone();
    cv::cvtColor(out1, out1, cv::COLOR_GRAY2BGR);

    // 画出特征跟踪结果
    for (auto kp : frame->kps) {
        cv::circle(out1, kp.pt, 2, cv::Scalar(0, 255, 0), cv::FILLED);
        cv::Point2d pt_rep = frame->World2Pixel(*kp.map_point.lock());
        cv::circle(out1, pt_rep, 2, cv::Scalar(0, 0, 255), cv::FILLED);
    }

    cv::imshow("rep", out1);
}

}