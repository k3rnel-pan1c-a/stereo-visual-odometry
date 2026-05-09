#include "Trajectory.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>

bool Trajectory::loadKittiPoses(
    const std::string& path,
    std::vector<cv::Matx44d>& poses
) {
    std::ifstream file(path);

    if (!file.is_open()) {
        std::cerr << "Failed to open ground truth poses file: " << path << std::endl;
        return false;
    }

    poses.clear();

    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);

        cv::Matx44d T = cv::Matx44d::eye();

        for (int i = 0; i < 12; ++i) {
            double value;
            ss >> value;

            int row = i / 4;
            int col = i % 4;

            T(row, col) = value;
        }

        poses.push_back(T);
    }

    return true;
}

bool Trajectory::saveKittiPoses(
    const std::string& path,
    const std::vector<cv::Matx44d>& poses
) {
    std::ofstream file(path);

    if (!file.is_open()) {
        std::cerr << "Failed to write estimated poses file: " << path << std::endl;
        return false;
    }

    for (const auto& T : poses) {
        file << T(0, 0) << " " << T(0, 1) << " " << T(0, 2) << " " << T(0, 3) << " "
             << T(1, 0) << " " << T(1, 1) << " " << T(1, 2) << " " << T(1, 3) << " "
             << T(2, 0) << " " << T(2, 1) << " " << T(2, 2) << " " << T(2, 3) << "\n";
    }

    return true;
}

void Trajectory::drawTopDownTrajectory(
    const std::vector<cv::Matx44d>& estimated,
    const std::vector<cv::Matx44d>& ground_truth,
    const std::string& output_path
) {
    int width = 1000;
    int height = 1000;
    int margin = 50;

    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    if (estimated.empty()) {
        cv::imwrite(output_path, canvas);
        return;
    }

    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double min_z = std::numeric_limits<double>::max();
    double max_z = std::numeric_limits<double>::lowest();

    auto updateBounds = [&](const std::vector<cv::Matx44d>& poses) {
        for (const auto& T : poses) {
            double x = T(0, 3);
            double z = T(2, 3);

            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_z = std::min(min_z, z);
            max_z = std::max(max_z, z);
        }
    };

    updateBounds(estimated);
    updateBounds(ground_truth);

    double range_x = std::max(1.0, max_x - min_x);
    double range_z = std::max(1.0, max_z - min_z);

    auto project = [&](double x, double z) {
        int px = static_cast<int>(
            margin + (x - min_x) / range_x * (width - 2 * margin)
        );

        int py = static_cast<int>(
            height - margin - (z - min_z) / range_z * (height - 2 * margin)
        );

        return cv::Point(px, py);
    };

    for (size_t i = 1; i < ground_truth.size(); ++i) {
        cv::Point p1 = project(ground_truth[i - 1](0, 3), ground_truth[i - 1](2, 3));
        cv::Point p2 = project(ground_truth[i](0, 3), ground_truth[i](2, 3));
        cv::line(canvas, p1, p2, cv::Scalar(0, 200, 0), 2);
    }

    for (size_t i = 1; i < estimated.size(); ++i) {
        cv::Point p1 = project(estimated[i - 1](0, 3), estimated[i - 1](2, 3));
        cv::Point p2 = project(estimated[i](0, 3), estimated[i](2, 3));
        cv::line(canvas, p1, p2, cv::Scalar(0, 0, 255), 2);
    }

    cv::putText(
        canvas,
        "Green: Ground Truth, Red: Estimated",
        cv::Point(30, 40),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(0, 0, 0),
        2
    );

    cv::imwrite(output_path, canvas);
}