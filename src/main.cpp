#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "Config.hpp"
#include "KittiDataset.hpp"
#include "Calibration.hpp"
#include "Frame.hpp"
#include "FeatureExtractor.hpp"
#include "StereoMatcher.hpp"
#include "MotionEstimator.hpp"
#include "Trajectory.hpp"
#include "Viewer.hpp"

namespace fs = std::filesystem;

cv::Matx44d inverseRigidTransform(const cv::Matx44d& T) {
    cv::Matx33d R(
        T(0, 0), T(0, 1), T(0, 2),
        T(1, 0), T(1, 1), T(1, 2),
        T(2, 0), T(2, 1), T(2, 2)
    );

    cv::Vec3d t(T(0, 3), T(1, 3), T(2, 3));

    cv::Matx33d Rt = R.t();
    cv::Vec3d t_inv = -(Rt * t);

    cv::Matx44d T_inv = cv::Matx44d::eye();

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            T_inv(r, c) = Rt(r, c);
        }
    }

    T_inv(0, 3) = t_inv[0];
    T_inv(1, 3) = t_inv[1];
    T_inv(2, 3) = t_inv[2];

    return T_inv;
}

void saveMatchImage(
    const std::string& path,
    const cv::Mat& img1,
    const std::vector<cv::KeyPoint>& kp1,
    const cv::Mat& img2,
    const std::vector<cv::KeyPoint>& kp2,
    const std::vector<cv::DMatch>& matches,
    int max_matches = 100
) {
    std::vector<cv::DMatch> shown_matches = matches;

    if (static_cast<int>(shown_matches.size()) > max_matches) {
        shown_matches.resize(max_matches);
    }

    cv::Mat output;
    cv::drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        shown_matches,
        output
    );

    cv::imwrite(path, output);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage:\n";
        std::cerr << "  ./stereo_vo <sequence_path> <poses_path>\n\n";
        std::cerr << "Example:\n";
        std::cerr << "  ./stereo_vo ../data/KITTI/dataset/sequences/00 ../data/KITTI/dataset/poses/00.txt\n";
        return 1;
    }

    std::string sequence_path = argv[1];
    std::string poses_path = argv[2];
    std::string calib_path = sequence_path + "/calib.txt";

    fs::create_directories("../outputs");
    fs::create_directories("../outputs/stereo_matches");
    fs::create_directories("../outputs/temporal_matches");

    Config config;

    Calibration calib;
    if (!calib.loadFromFile(calib_path)) {
        std::cerr << "Failed to load calibration." << std::endl;
        return 1;
    }

    calib.print();

    KittiDataset dataset(sequence_path);

    std::cout << "\nNumber of stereo frames: " << dataset.size() << std::endl;

    if (dataset.size() == 0) {
        std::cerr << "No frames found." << std::endl;
        return 1;
    }

    FeatureExtractor feature_extractor(config.orb_features);
    StereoMatcher stereo_matcher(config);
    MotionEstimator motion_estimator(config);

    Viewer viewer(calib);
    viewer.start();

    constexpr float VIZ_MIN_DEPTH = 2.0f;
    constexpr float VIZ_MAX_DEPTH = 30.0f;

    auto buildWorldPoints = [](const Frame& f) {
        std::vector<cv::Vec3f> world_points;
        world_points.reserve(f.points3d.size());
        for (size_t j = 0; j < f.points3d.size(); ++j) {
            if (j < f.has_depth.size() && !f.has_depth[j]) continue;
            const auto& p = f.points3d[j];
            if (p.z < VIZ_MIN_DEPTH || p.z > VIZ_MAX_DEPTH) continue;
            cv::Vec4d ph(p.x, p.y, p.z, 1.0);
            cv::Vec4d pw = f.T_world_cam * ph;
            world_points.emplace_back((float)pw[0], (float)pw[1], (float)pw[2]);
        }
        return world_points;
    };

    auto makeStereoDisplay = [](const cv::Mat& left, const cv::Mat& right) {
        cv::Mat separator(left.rows, 4, left.type(), cv::Scalar::all(255));
        std::vector<cv::Mat> parts = {left, separator, right};
        cv::Mat out;
        cv::hconcat(parts, out);
        return out;
    };

    auto drawKeypointsOverlay = [](
        const cv::Mat& img,
        const std::vector<cv::KeyPoint>& kps,
        const std::vector<int>& highlight_idx
    ) {
        cv::Mat colored;
        if (img.channels() == 1) {
            cv::cvtColor(img, colored, cv::COLOR_GRAY2BGR);
        } else {
            colored = img.clone();
        }
        std::vector<bool> is_highlight(kps.size(), false);
        for (int idx : highlight_idx) {
            if (idx >= 0 && idx < (int)kps.size()) is_highlight[idx] = true;
        }
        for (size_t i = 0; i < kps.size(); ++i) {
            if (is_highlight[i]) {
                cv::circle(colored, kps[i].pt, 4, cv::Scalar(0, 255, 255), 2);
            } else {
                cv::circle(colored, kps[i].pt, 2, cv::Scalar(80, 200, 80), 1);
            }
        }
        return colored;
    };

    std::vector<cv::Matx44d> estimated_poses;
    std::vector<cv::Matx44d> ground_truth_poses;

    Trajectory::loadKittiPoses(poses_path, ground_truth_poses);
    std::cout << "Ground truth poses loaded: " << ground_truth_poses.size() << std::endl;
    if (ground_truth_poses.empty()) {
        std::cout << "  (path tried: " << poses_path << ")" << std::endl;
    }

    Frame prev_frame;
    cv::Matx44d T_world_prev = cv::Matx44d::eye();

    int total_frames = std::min((size_t)config.max_frames, dataset.size());

    for (int i = 0; i < total_frames; ++i) {
        Frame curr_frame;

        if (!dataset.loadFrame(i, curr_frame)) {
            std::cerr << "Failed to load frame " << i << std::endl;
            continue;
        }

        feature_extractor.extract(
            curr_frame.left_img,
            curr_frame.keypoints_left,
            curr_frame.descriptors_left
        );

        feature_extractor.extract(
            curr_frame.right_img,
            curr_frame.keypoints_right,
            curr_frame.descriptors_right
        );

        std::vector<cv::DMatch> stereo_valid_matches;

        int valid_depth_count = stereo_matcher.computeSparseDepth(
            curr_frame,
            calib,
            &stereo_valid_matches
        );

        std::cout << "\nFrame " << i << std::endl;
        std::cout << "Left keypoints: " << curr_frame.keypoints_left.size() << std::endl;
        std::cout << "Right keypoints: " << curr_frame.keypoints_right.size() << std::endl;
        std::cout << "Valid stereo 3D points: " << valid_depth_count << std::endl;

        if (i == 0) {
            curr_frame.T_world_cam = cv::Matx44d::eye();
            estimated_poses.push_back(curr_frame.T_world_cam);

            saveMatchImage(
                "../outputs/stereo_matches/frame_000000.png",
                curr_frame.left_img,
                curr_frame.keypoints_left,
                curr_frame.right_img,
                curr_frame.keypoints_right,
                stereo_valid_matches
            );

            cv::Mat left_overlay  = drawKeypointsOverlay(curr_frame.left_img,  curr_frame.keypoints_left,  {});
            cv::Mat right_overlay = drawKeypointsOverlay(curr_frame.right_img, curr_frame.keypoints_right, {});
            cv::Mat stereo_display = makeStereoDisplay(left_overlay, right_overlay);

            std::vector<cv::Vec3f> viz_points = buildWorldPoints(curr_frame);

            viewer.pushUpdate(
                curr_frame.T_world_cam,
                viz_points,
                stereo_display
            );

            prev_frame = curr_frame;
            T_world_prev = curr_frame.T_world_cam;
            continue;
        }

        MotionResult motion = motion_estimator.estimateMotion(
            prev_frame,
            curr_frame,
            calib
        );

        std::cout << "3D-2D correspondences: " << motion.num_correspondences << std::endl;
        std::cout << "PnP success: " << (motion.success ? "yes" : "no") << std::endl;
        std::cout << "PnP inliers: " << motion.num_inliers << std::endl;

        if (motion.success) {
            std::cout << "tvec: ["
                      << motion.tvec[0] << ", "
                      << motion.tvec[1] << ", "
                      << motion.tvec[2] << "]" << std::endl;

            std::cout << "rvec: ["
                      << motion.rvec[0] << ", "
                      << motion.rvec[1] << ", "
                      << motion.rvec[2] << "]" << std::endl;

            cv::Matx44d T_prev_curr = inverseRigidTransform(motion.T_curr_prev);

            curr_frame.T_world_cam = T_world_prev * T_prev_curr;
            T_world_prev = curr_frame.T_world_cam;
        } else {
            std::cout << "Using previous pose because PnP failed." << std::endl;
            curr_frame.T_world_cam = T_world_prev;
        }

        estimated_poses.push_back(curr_frame.T_world_cam);

        std::vector<int> inlier_train_idx;
        inlier_train_idx.reserve(motion.inlier_matches.size());
        for (const auto& m : motion.inlier_matches) {
            inlier_train_idx.push_back(m.trainIdx);
        }

        cv::Mat left_overlay  = drawKeypointsOverlay(curr_frame.left_img,  curr_frame.keypoints_left,  inlier_train_idx);
        cv::Mat right_overlay = drawKeypointsOverlay(curr_frame.right_img, curr_frame.keypoints_right, {});
        cv::Mat stereo_display = makeStereoDisplay(left_overlay, right_overlay);

        std::vector<cv::Vec3f> viz_points;
        if (i % 3 == 0) {
            viz_points = buildWorldPoints(curr_frame);
        }

        viewer.pushUpdate(
            curr_frame.T_world_cam,
            viz_points,
            stereo_display
        );

        if (i == 1) {
            saveMatchImage(
                "../outputs/temporal_matches/frame_000000_000001.png",
                prev_frame.left_img,
                prev_frame.keypoints_left,
                curr_frame.left_img,
                curr_frame.keypoints_left,
                motion.inlier_matches
            );
        }

        if (i % 50 == 0) {
            std::cout << "Current estimated position: x = "
                      << curr_frame.T_world_cam(0, 3)
                      << ", z = "
                      << curr_frame.T_world_cam(2, 3)
                      << std::endl;
        }

        prev_frame = curr_frame;
    }

    Trajectory::saveKittiPoses(
        "../outputs/trajectory_est.txt",
        estimated_poses
    );

    if (ground_truth_poses.size() > estimated_poses.size()) {
        ground_truth_poses.resize(estimated_poses.size());
    }

    Trajectory::drawTopDownTrajectory(
        estimated_poses,
        ground_truth_poses,
        "../outputs/trajectory_plot.png"
    );

    EvaluationResult eval;
    bool have_eval = false;
    if (!ground_truth_poses.empty()) {
        eval = Trajectory::evaluate(estimated_poses, ground_truth_poses, 10);
        Trajectory::saveEvaluation("../outputs/evaluation.txt", eval);
        have_eval = true;
    } else {
        std::cout << "\nWARNING: no ground-truth poses; ATE/RPE/drift evaluation skipped." << std::endl;
    }

    std::cout << "\nDone." << std::endl;
    std::cout << "Estimated trajectory:  " << fs::absolute("../outputs/trajectory_est.txt") << std::endl;
    std::cout << "Trajectory plot:       " << fs::absolute("../outputs/trajectory_plot.png") << std::endl;
    if (have_eval) {
        std::cout << "Evaluation file:       " << fs::absolute("../outputs/evaluation.txt") << std::endl;
    }
    std::cout << "Stereo match sample:   " << fs::absolute("../outputs/stereo_matches/frame_000000.png") << std::endl;
    std::cout << "Temporal match sample: " << fs::absolute("../outputs/temporal_matches/frame_000000_000001.png") << std::endl;

    std::cout << "\nViewer is still running. Press Enter in this terminal to exit." << std::endl;
    std::cin.get();

    viewer.requestStop();
    viewer.join();

    if (have_eval) {
        Trajectory::printEvaluation(eval);
    }

    return 0;
}