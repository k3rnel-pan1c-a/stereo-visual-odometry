#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "Frame.hpp"
#include "Calibration.hpp"
#include "Config.hpp"

class StereoMatcher {
public:
    explicit StereoMatcher(const Config& config);

    int computeSparseDepth(
        Frame& frame,
        const Calibration& calib,
        std::vector<cv::DMatch>* valid_matches_out = nullptr
    );

private:
    Config config_;
};