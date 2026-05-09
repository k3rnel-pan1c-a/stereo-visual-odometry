#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "Frame.hpp"
#include "Calibration.hpp"
#include "Config.hpp"

struct MotionResult {
    bool success = false;

    int num_correspondences = 0;
    int num_inliers = 0;

    cv::Matx44d T_curr_prev = cv::Matx44d::eye();

    cv::Vec3d rvec;
    cv::Vec3d tvec;

    std::vector<cv::DMatch> temporal_matches;
    std::vector<cv::DMatch> inlier_matches;
};

class MotionEstimator {
public:
    explicit MotionEstimator(const Config& config);

    MotionResult estimateMotion(
        const Frame& prev,
        const Frame& curr,
        const Calibration& calib
    );

private:
    Config config_;
};