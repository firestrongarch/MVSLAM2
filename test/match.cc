#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "gms.h"
#include <print>
#include <fstream>

void matchFeaturesBF(cv::Mat& descriptors1, cv::Mat& descriptors2, std::vector<cv::DMatch>& matches, bool crossCheck) {
    cv::BFMatcher matcher(cv::NORM_HAMMING, crossCheck);
    matcher.match(descriptors1, descriptors2, matches);
}

void matchFeaturesFLANN(cv::Mat& descriptors1, cv::Mat& descriptors2, std::vector<cv::DMatch>& matches) {
    cv::Ptr<cv::flann::IndexParams> indexParams = new cv::flann::LshIndexParams(6, 12, 2);
    cv::Ptr<cv::flann::SearchParams> searchParams = new cv::flann::SearchParams(50);
    cv::FlannBasedMatcher matcher(indexParams, searchParams);
    matcher.match(descriptors1, descriptors2, matches);
}

void matchFeaturesKNN(cv::Mat& descriptors1, cv::Mat& descriptors2, std::vector<cv::DMatch>& matches) {
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

    // 应用Lowe's ratio test筛选匹配点
    const float ratioThresh = 0.75f;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance) {
            matches.push_back(knnMatches[i][0]);
        }
    }
}

// GMS 匹配算法实现
void matchFeaturesGMS(cv::Mat& img1, cv::Mat& img2, 
    std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2,
    std::vector<cv::DMatch>& matches) {
    // 参数设置
    int gridX = 20; // 网格X方向划分数量
    int gridY = 20; // 网格Y方向划分数量
    int maxInliers = 0; // 最大内点数
    std::vector<cv::DMatch> refinedMatches; // 筛选后的匹配点

    // 计算图像尺寸
    int width = img1.cols;
    int height = img1.rows;

    // 计算每个网格的大小
    float cellWidth = static_cast<float>(width) / gridX;
    float cellHeight = static_cast<float>(height) / gridY;

    // 创建网格结构
    std::vector<std::vector<std::vector<cv::DMatch>>> grid(gridX, 
    std::vector<std::vector<cv::DMatch>>(gridY));

    // 将匹配点分配到网格中
    for (const auto& match : matches) {
        int pt1X = static_cast<int>(keypoints1[match.queryIdx].pt.x);
        int pt1Y = static_cast<int>(keypoints1[match.queryIdx].pt.y);
        int gridXIndex = std::min(gridX - 1, static_cast<int>(pt1X / cellWidth));
        int gridYIndex = std::min(gridY - 1, static_cast<int>(pt1Y / cellHeight));
        grid[gridXIndex][gridYIndex].push_back(match);
    }

    // 遍历网格，找到包含最多匹配点的网格
    for (int i = 0; i < gridX; ++i) {
        for (int j = 0; j < gridY; ++j) {
            if (grid[i][j].size() > maxInliers) {
                maxInliers = grid[i][j].size();
                refinedMatches = grid[i][j];
            }
        }
    }

    // 如果没有找到足够的匹配点，直接返回原始匹配点
    if (refinedMatches.empty()) {
        refinedMatches = matches;
    }

    // 将筛选后的匹配点赋值给输出参数
    matches = refinedMatches;
}

void matchFeaturesReprojection(const cv::Mat& img1, const cv::Mat& img2,
                              std::vector<cv::KeyPoint>& keypoints1, 
                              std::vector<cv::KeyPoint>& keypoints2,
                              std::vector<cv::DMatch>& matches) {
    if (matches.empty()) return;

    // 计算基础矩阵
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    std::vector<uchar> inlierMask;
    cv::Mat F = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3.0, 0.99, inlierMask);

    // 根据重投影误差筛选匹配点
    std::vector<cv::DMatch> refinedMatches;
    for (size_t i = 0; i < inlierMask.size(); i++) {
        if (inlierMask[i]) {
            refinedMatches.push_back(matches[i]);
        }
    }

    matches = refinedMatches;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::println("用法: {} <特征点数量> <图片1> <图片2> <保存目录>", argv[0]);
        return -1;
    }

    // 读取输入图像
    cv::Mat img1 = cv::imread(argv[2]);
    cv::Mat img2 = cv::imread(argv[3]);
    if (img1.empty() || img2.empty()) {
        std::cout << "无法加载图像！" << std::endl;
        return -1;
    }

    int size = std::stoi(argv[1]);
    // 初始化ORB特征检测器
    cv::Ptr<cv::ORB> orb = cv::ORB::create(size);

    // 检测关键点并计算描述符
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    orb->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);

    // 方法1：BFMatcher（汉明距离，交叉验证）
    std::vector<cv::DMatch> matches1;
    matchFeaturesBF(descriptors1, descriptors2, matches1, true);
    cv::Mat result1;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches1, result1,
                   cv::Scalar::all(-1), cv::Scalar::all(-1),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // 方法2：BFMatcher（汉明距离，非交叉验证）
    std::vector<cv::DMatch> matches2;
    matchFeaturesBF(descriptors1, descriptors2, matches2, false);
    cv::Mat result2;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches2, result2,
                   cv::Scalar::all(-1), cv::Scalar::all(-1),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // 方法3：FLANN匹配器
    std::vector<cv::DMatch> matches3;
    matchFeaturesFLANN(descriptors1, descriptors2, matches3);
    cv::Mat result3;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches3, result3,
                   cv::Scalar::all(-1), cv::Scalar::all(-1),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // 方法4：KNN匹配
    std::vector<cv::DMatch> matches4;
    matchFeaturesKNN(descriptors1, descriptors2, matches4);
    cv::Mat result4;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches4, result4,
        cv::Scalar::all(-1), cv::Scalar::all(-1),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    // 方法5：GMS匹配
    std::vector<cv::DMatch> matches5 = matches2; // 使用BF匹配结果作为输入
    matchFeaturesGMS(img1, img2, keypoints1, keypoints2, matches5);
    cv::Mat result5;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches5, result5,
                   cv::Scalar::all(-1), cv::Scalar::all(-1),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // 方法6：GMS匹配
	std::vector<bool> vbInliers;
    std::vector<cv::DMatch> matches6 = matches2; 
    std::vector<cv::DMatch> matches_gms; 
	gms_matcher gms(keypoints1, img1.size(), keypoints2, img2.size(), matches6);
	int num_inliers = gms.GetInlierMask(vbInliers, false, false);
    // collect matches
	for (size_t i = 0; i < vbInliers.size(); ++i) {
		if (vbInliers[i] == true) {
			matches_gms.push_back(matches6[i]);
		}
	}
    cv::Mat result6;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches_gms, result6,
        cv::Scalar::all(-1), cv::Scalar::all(-1),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // 方法7：重投影匹配
    std::vector<cv::DMatch> matches7 = matches2; // 使用BF匹配结果作为输入
    matchFeaturesReprojection(img1, img2, keypoints1, keypoints2, matches7);
    cv::Mat result7;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches7, result7,
        cv::Scalar::all(-1), cv::Scalar::all(-1),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    std::string save_dir = argv[4];
    std::ofstream file(save_dir + "/output.txt");
    // 在显示窗口之前添加以下代码
    std::println(file, "特征匹配结果统计:");
    std::println(file, "检测到的特征点数量: 图像1={}, 图像2={}", keypoints1.size(), keypoints2.size());
    std::println(file, "BFMatcher (Cross-Check) 匹配数量: {}", matches1.size());
    std::println(file, "BFMatcher (Non-Cross-Check) 匹配数量: {}", matches2.size());
    std::println(file, "FLANN Matcher 匹配数量: {}", matches3.size());
    std::println(file, "KNN Matcher 匹配数量: {}", matches4.size());
    std::println(file, "GMS Matcher 匹配数量: {}", matches5.size());
    std::println(file, "GMS Original 匹配数量: {}", matches_gms.size());
    std::println(file, "Reprojection Matcher 匹配数量: {}", matches7.size());
    file.close();
    
    // 保存匹配结果图片
    cv::imwrite(save_dir + "/01_bf_cross_check.png", result1);
    cv::imwrite(save_dir + "/02_bf_no_cross_check.png", result2);
    cv::imwrite(save_dir + "/03_flann.png", result3);
    cv::imwrite(save_dir + "/04_knn.png", result4);
    cv::imwrite(save_dir + "/05_gms.png", result5);
    cv::imwrite(save_dir + "/06_gms_ori.png", result6);
    cv::imwrite(save_dir + "/07_reprojection.png", result7);

    std::println("已将结果图片保存至目录: {}", save_dir);

    // 显示结果
    cv::namedWindow("BFMatcher (Cross-Check)", cv::WINDOW_NORMAL);
    cv::imshow("BFMatcher (Cross-Check)", result1);

    cv::namedWindow("BFMatcher (Non-Cross-Check)", cv::WINDOW_NORMAL);
    cv::imshow("BFMatcher (Non-Cross-Check)", result2);

    cv::namedWindow("FLANN Matcher", cv::WINDOW_NORMAL);
    cv::imshow("FLANN Matcher", result3);

    cv::namedWindow("KNN Matcher", cv::WINDOW_NORMAL);
    cv::imshow("KNN Matcher", result4);

    cv::namedWindow("GMS Matcher", cv::WINDOW_NORMAL);
    cv::imshow("GMS Matcher", result5);

    cv::namedWindow("GMS Ori", cv::WINDOW_NORMAL);
    cv::imshow("GMS Ori", result6);

    cv::namedWindow("Reprojection Matcher", cv::WINDOW_NORMAL);
    cv::imshow("Reprojection Matcher", result7);


    
    cv::waitKey(0);

    return 0;
}