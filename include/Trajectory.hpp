#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct EvaluationResult {
    int num_compared = 0;
    double trajectory_length_gt = 0.0;
    double final_position_error = 0.0;
    double drift_pct = 0.0;

    double ate_translation_rmse = 0.0;
    double ate_translation_mean = 0.0;
    double ate_translation_max = 0.0;

    int rpe_delta_frames = 0;
    double rpe_translation_rmse = 0.0;
    double rpe_rotation_deg_rmse = 0.0;
};

class Trajectory {
public:
    static bool loadKittiPoses(
        const std::string& path,
        std::vector<cv::Matx44d>& poses
    );

    static bool saveKittiPoses(
        const std::string& path,
        const std::vector<cv::Matx44d>& poses
    );

    static void drawTopDownTrajectory(
        const std::vector<cv::Matx44d>& estimated,
        const std::vector<cv::Matx44d>& ground_truth,
        const std::string& output_path
    );

    static EvaluationResult evaluate(
        const std::vector<cv::Matx44d>& estimated,
        const std::vector<cv::Matx44d>& ground_truth,
        int rpe_delta_frames = 10
    );

    static void printEvaluation(const EvaluationResult& result);

    static bool saveEvaluation(
        const std::string& path,
        const EvaluationResult& result
    );
};