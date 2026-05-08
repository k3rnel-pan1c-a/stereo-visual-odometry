#include "Calibration.hpp"

#include <fstream>
#include <sstream>
#include <iostream>

bool Calibration::loadFromFile(const std::string& calib_path) {
    std::ifstream file(calib_path);

    if (!file.is_open()) {
        std::cerr << "Failed to open calibration file: " << calib_path << std::endl;
        return false;
    }

    std::string line;

    while (std::getline(file, line)) {
        if (line.rfind("P0:", 0) == 0) {
            if (!parseProjectionLine(line, P0)) {
                return false;
            }
        } else if (line.rfind("P1:", 0) == 0) {
            if (!parseProjectionLine(line, P1)) {
                return false;
            }
        }
    }

    if (P0.empty() || P1.empty()) {
        std::cerr << "Could not find P0 and P1 in calibration file." << std::endl;
        return false;
    }

    fx = P0.at<double>(0, 0);
    fy = P0.at<double>(1, 1);
    cx = P0.at<double>(0, 2);
    cy = P0.at<double>(1, 2);

    baseline = -P1.at<double>(0, 3) / P1.at<double>(0, 0);

    K = (cv::Mat_<double>(3, 3) <<
        fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0
    );

    return true;
}

bool Calibration::parseProjectionLine(const std::string& line, cv::Mat& P) {
    std::stringstream ss(line);

    std::string label;
    ss >> label;

    P = cv::Mat::zeros(3, 4, CV_64F);

    for (int i = 0; i < 12; ++i) {
        double value;
        if (!(ss >> value)) {
            std::cerr << "Failed to parse projection matrix line: " << line << std::endl;
            return false;
        }

        int row = i / 4;
        int col = i % 4;
        P.at<double>(row, col) = value;
    }

    return true;
}

void Calibration::print() const {
    std::cout << "Calibration:" << std::endl;
    std::cout << "fx = " << fx << std::endl;
    std::cout << "fy = " << fy << std::endl;
    std::cout << "cx = " << cx << std::endl;
    std::cout << "cy = " << cy << std::endl;
    std::cout << "baseline = " << baseline << " meters" << std::endl;

    std::cout << "\nK:\n" << K << std::endl;
    std::cout << "\nP0:\n" << P0 << std::endl;
    std::cout << "\nP1:\n" << P1 << std::endl;
}