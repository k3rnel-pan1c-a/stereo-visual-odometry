#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class Trajectory {
public:
    static bool loadKittiPoses(
        const std::string& path,
        std::vector<cv::Matx44d>& poses
    );

    static bool saveKittiPoses(
        const std::string& path,
        const std::vector<cv::Matx44d>& poses
    );

    static void drawTopDownTrajectory(
        const std::vector<cv::Matx44d>& estimated,
        const std::vector<cv::Matx44d>& ground_truth,
        const std::string& output_path
    );
};