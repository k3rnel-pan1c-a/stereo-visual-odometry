#include "StereoMatcher.hpp"


StereoMatcher::StereoMatcher(const Config& config)
    : config_(config) {}

int StereoMatcher::computeSparseDepth(
    Frame& frame,
    const Calibration& calib,
    std::vector<cv::DMatch>* valid_matches_out
) {
    frame.points3d.assign(frame.keypoints_left.size(), cv::Point3f(0, 0, 0));
    frame.has_depth.assign(frame.keypoints_left.size(), false);

    if (frame.descriptors_left.empty() || frame.descriptors_right.empty()) {
        return 0;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING, true);

    std::vector<cv::DMatch> matches;
    matcher.match(frame.descriptors_left, frame.descriptors_right, matches);

    int valid_count = 0;
    std::vector<cv::DMatch> valid_matches;

    for (const auto& m : matches) {
        const cv::Point2f& pL = frame.keypoints_left[m.queryIdx].pt;
        const cv::Point2f& pR = frame.keypoints_right[m.trainIdx].pt;

        double y_diff = std::abs(pL.y - pR.y);
        if (y_diff > config_.stereo_max_y_diff) {
            continue;
        }

        double disparity = pL.x - pR.x;
        if (disparity < config_.min_disparity) {
            continue;
        }

        double Z = calib.fx * calib.baseline / disparity;

        if (Z < config_.min_depth || Z > config_.max_depth) {
            continue;
        }

        double X = (pL.x - calib.cx) * Z / calib.fx;
        double Y = (pL.y - calib.cy) * Z / calib.fy;

        frame.points3d[m.queryIdx] = cv::Point3f(
            static_cast<float>(X),
            static_cast<float>(Y),
            static_cast<float>(Z)
        );

        frame.has_depth[m.queryIdx] = true;

        valid_matches.push_back(m);
        valid_count++;
    }

    if (valid_matches_out != nullptr) {
        *valid_matches_out = valid_matches;
    }

    return valid_count;
}