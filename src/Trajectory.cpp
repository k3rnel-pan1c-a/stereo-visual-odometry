#include "Trajectory.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

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

static cv::Matx44d invSE3(const cv::Matx44d& T) {
    cv::Matx33d R(
        T(0, 0), T(0, 1), T(0, 2),
        T(1, 0), T(1, 1), T(1, 2),
        T(2, 0), T(2, 1), T(2, 2)
    );
    cv::Vec3d t(T(0, 3), T(1, 3), T(2, 3));
    cv::Matx33d Rt = R.t();
    cv::Vec3d ti = -(Rt * t);
    cv::Matx44d Ti = cv::Matx44d::eye();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            Ti(r, c) = Rt(r, c);
        }
    }
    Ti(0, 3) = ti[0];
    Ti(1, 3) = ti[1];
    Ti(2, 3) = ti[2];
    return Ti;
}

static double rotationAngleDeg(const cv::Matx44d& T) {
    double trace = T(0, 0) + T(1, 1) + T(2, 2);
    double cos_theta = std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0));
    return std::acos(cos_theta) * 180.0 / CV_PI;
}

EvaluationResult Trajectory::evaluate(
    const std::vector<cv::Matx44d>& estimated,
    const std::vector<cv::Matx44d>& ground_truth,
    int rpe_delta_frames
) {
    EvaluationResult r;
    int n = static_cast<int>(std::min(estimated.size(), ground_truth.size()));
    r.num_compared = n;
    r.rpe_delta_frames = rpe_delta_frames;
    if (n < 2) return r;

    double sum_sq = 0.0;
    double sum = 0.0;
    double max_err = 0.0;
    for (int i = 0; i < n; ++i) {
        double dx = estimated[i](0, 3) - ground_truth[i](0, 3);
        double dy = estimated[i](1, 3) - ground_truth[i](1, 3);
        double dz = estimated[i](2, 3) - ground_truth[i](2, 3);
        double e2 = dx * dx + dy * dy + dz * dz;
        double e = std::sqrt(e2);
        sum_sq += e2;
        sum += e;
        max_err = std::max(max_err, e);
    }
    r.ate_translation_rmse = std::sqrt(sum_sq / n);
    r.ate_translation_mean = sum / n;
    r.ate_translation_max = max_err;

    double length = 0.0;
    for (int i = 1; i < n; ++i) {
        double dx = ground_truth[i](0, 3) - ground_truth[i - 1](0, 3);
        double dy = ground_truth[i](1, 3) - ground_truth[i - 1](1, 3);
        double dz = ground_truth[i](2, 3) - ground_truth[i - 1](2, 3);
        length += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    r.trajectory_length_gt = length;

    {
        int last = n - 1;
        double dx = estimated[last](0, 3) - ground_truth[last](0, 3);
        double dy = estimated[last](1, 3) - ground_truth[last](1, 3);
        double dz = estimated[last](2, 3) - ground_truth[last](2, 3);
        r.final_position_error = std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    r.drift_pct = (length > 1e-6)
        ? 100.0 * r.final_position_error / length
        : 0.0;

    if (rpe_delta_frames > 0 && n - rpe_delta_frames > 1) {
        double sum_t_sq = 0.0;
        double sum_r_sq = 0.0;
        int count = 0;
        for (int i = 0; i + rpe_delta_frames < n; ++i) {
            cv::Matx44d est_rel = invSE3(estimated[i]) * estimated[i + rpe_delta_frames];
            cv::Matx44d gt_rel  = invSE3(ground_truth[i]) * ground_truth[i + rpe_delta_frames];
            cv::Matx44d err     = invSE3(gt_rel) * est_rel;

            double tx = err(0, 3);
            double ty = err(1, 3);
            double tz = err(2, 3);
            sum_t_sq += tx * tx + ty * ty + tz * tz;

            double dr = rotationAngleDeg(err);
            sum_r_sq += dr * dr;

            count++;
        }
        if (count > 0) {
            r.rpe_translation_rmse = std::sqrt(sum_t_sq / count);
            r.rpe_rotation_deg_rmse = std::sqrt(sum_r_sq / count);
        }
    }

    return r;
}

void Trajectory::printEvaluation(const EvaluationResult& r) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=== Trajectory Evaluation ===\n";
    std::cout << "Frames compared:        " << r.num_compared << "\n";
    std::cout << "GT trajectory length:   " << r.trajectory_length_gt << " m\n";
    std::cout << "ATE translation RMSE:   " << r.ate_translation_rmse << " m\n";
    std::cout << "ATE translation mean:   " << r.ate_translation_mean << " m\n";
    std::cout << "ATE translation max:    " << r.ate_translation_max << " m\n";
    std::cout << "Final position error:   " << r.final_position_error << " m\n";
    std::cout << "Drift (final / length): " << r.drift_pct << " %\n";
    std::cout << "RPE delta:              " << r.rpe_delta_frames << " frames\n";
    std::cout << "RPE translation RMSE:   " << r.rpe_translation_rmse << " m / "
              << r.rpe_delta_frames << " frames\n";
    std::cout << "RPE rotation RMSE:      " << r.rpe_rotation_deg_rmse << " deg / "
              << r.rpe_delta_frames << " frames\n";
    std::cout << "=============================\n";
}

bool Trajectory::saveEvaluation(
    const std::string& path,
    const EvaluationResult& r
) {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to write evaluation file: " << path << std::endl;
        return false;
    }
    file << std::fixed << std::setprecision(6);
    file << "frames_compared "        << r.num_compared             << "\n";
    file << "trajectory_length_m "    << r.trajectory_length_gt     << "\n";
    file << "ate_translation_rmse_m " << r.ate_translation_rmse     << "\n";
    file << "ate_translation_mean_m " << r.ate_translation_mean     << "\n";
    file << "ate_translation_max_m "  << r.ate_translation_max      << "\n";
    file << "final_position_error_m " << r.final_position_error     << "\n";
    file << "drift_pct "              << r.drift_pct                << "\n";
    file << "rpe_delta_frames "       << r.rpe_delta_frames         << "\n";
    file << "rpe_translation_rmse_m " << r.rpe_translation_rmse     << "\n";
    file << "rpe_rotation_rmse_deg "  << r.rpe_rotation_deg_rmse    << "\n";
    return true;
}