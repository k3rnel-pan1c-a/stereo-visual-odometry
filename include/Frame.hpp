#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

struct Frame {
    int id = -1;

    cv::Mat left_img;
    cv::Mat right_img;

    std::vector<cv::KeyPoint> keypoints_left;
    std::vector<cv::KeyPoint> keypoints_right;

    cv::Mat descriptors_left;
    cv::Mat descriptors_right;

    std::vector<cv::Point3f> points3d;
    std::vector<bool> has_depth;

    cv::Matx44d T_world_cam = cv::Matx44d::eye();

    Frame() = default;

    Frame(int frame_id, const cv::Mat& left, const cv::Mat& right)
        : id(frame_id), left_img(left), right_img(right) {}
};