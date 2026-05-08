#pragma once

#include <string>
#include <opencv2/opencv.hpp>

class Calibration {
public:
    bool loadFromFile(const std::string& calib_path);

    void print() const;

    cv::Mat K;
    cv::Mat P0;
    cv::Mat P1;

    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    double baseline = 0.0;

private:
    bool parseProjectionLine(const std::string& line, cv::Mat& P);
};