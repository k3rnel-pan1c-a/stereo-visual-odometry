#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "Calibration.hpp"

class Viewer {
public:
    Viewer(const Calibration& calib, std::size_t max_points = 200000);
    ~Viewer();

    void start();
    void requestStop();
    void join();

    void pushUpdate(
        const cv::Matx44d& T_world_cam,
        const std::vector<cv::Vec3f>& world_points,
        const cv::Mat& left_image
    );

private:
    void renderLoop();

    Calibration calib_;
    std::size_t max_points_;

    std::mutex mutex_;
    std::vector<cv::Matx44d> trajectory_;
    std::deque<cv::Vec3f> map_points_;
    cv::Mat current_image_;
    int image_width_  = 1241;
    int image_height_ = 376;

    std::atomic<bool> stop_requested_{false};
    std::thread thread_;
};
