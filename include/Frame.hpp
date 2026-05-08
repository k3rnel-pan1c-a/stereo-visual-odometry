#pragma once

#include <opencv2/opencv.hpp>

struct Frame{
    int id = -1;
    cv::Mat left_img;
    cv::Mat right_img;
    
    Frame() = default;
    Frame(int frame_id, const cv::Mat& left, const cv::Mat& right)
    : id(frame_id), left_img(left), right_img(right) {}  
};