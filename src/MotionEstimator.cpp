#include "MotionEstimator.hpp"


MotionEstimator::MotionEstimator(const Config& config)
    : config_(config) {}

MotionResult MotionEstimator::estimateMotion(
    const Frame& prev,
    const Frame& curr,
    const Calibration& calib
) {
    MotionResult result;

    if (prev.descriptors_left.empty() || curr.descriptors_left.empty()) {
        return result;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING, true);

    std::vector<cv::DMatch> matches;
    matcher.match(prev.descriptors_left, curr.descriptors_left, matches);

    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    std::vector<cv::DMatch> used_matches;

    for (const auto& m : matches) {
        if (m.distance > config_.max_match_distance) {
            continue;
        }

        int prev_idx = m.queryIdx;
        int curr_idx = m.trainIdx;

        if (prev_idx < 0 || prev_idx >= static_cast<int>(prev.has_depth.size())) {
            continue;
        }

        if (!prev.has_depth[prev_idx]) {
            continue;
        }

        object_points.push_back(prev.points3d[prev_idx]);
        image_points.push_back(curr.keypoints_left[curr_idx].pt);
        used_matches.push_back(m);
    }

    result.num_correspondences = static_cast<int>(object_points.size());
    result.temporal_matches = used_matches;

    if (result.num_correspondences < config_.min_pnp_points) {
        return result;
    }

    cv::Mat rvec, tvec;
    std::vector<int> inliers;

    bool ok = cv::solvePnPRansac(
        object_points,
        image_points,
        calib.K,
        cv::noArray(),
        rvec,
        tvec,
        false,
        config_.pnp_iterations,
        static_cast<float>(config_.pnp_reproj_error),
        config_.pnp_confidence,
        inliers,
        cv::SOLVEPNP_ITERATIVE
    );

    if (!ok) {
        return result;
    }

    result.num_inliers = static_cast<int>(inliers.size());

    if (result.num_inliers < config_.min_pnp_inliers) {
        return result;
    }

    std::vector<cv::Point3f> inlier_object_points;
    std::vector<cv::Point2f> inlier_image_points;
    std::vector<cv::DMatch> inlier_matches;

    for (int idx : inliers) {
        inlier_object_points.push_back(object_points[idx]);
        inlier_image_points.push_back(image_points[idx]);
        inlier_matches.push_back(used_matches[idx]);
    }

    if (static_cast<int>(inlier_object_points.size()) >= config_.min_pnp_points) {
        cv::solvePnPRefineLM(
            inlier_object_points,
            inlier_image_points,
            calib.K,
            cv::noArray(),
            rvec,
            tvec
        );
    }

    cv::Mat R;
    cv::Rodrigues(rvec, R);

    cv::Matx44d T = cv::Matx44d::eye();

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            T(r, c) = R.at<double>(r, c);
        }
    }

    T(0, 3) = tvec.at<double>(0);
    T(1, 3) = tvec.at<double>(1);
    T(2, 3) = tvec.at<double>(2);

    result.success = true;
    result.T_curr_prev = T;

    result.rvec = cv::Vec3d(
        rvec.at<double>(0),
        rvec.at<double>(1),
        rvec.at<double>(2)
    );

    result.tvec = cv::Vec3d(
        tvec.at<double>(0),
        tvec.at<double>(1),
        tvec.at<double>(2)
    );

    result.inlier_matches = inlier_matches;

    return result;
}