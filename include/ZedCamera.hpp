#pragma once

#include <string>

#include <opencv2/opencv.hpp>

#include "Calibration.hpp"
#include "Frame.hpp"

class ZedCamera {
public:
    enum class Resolution { VGA, HD, FHD, K2 };

    ZedCamera(
        const std::string& conf_path,
        int camera_index = 0,
        Resolution resolution = Resolution::HD
    );

    bool open();

    bool loadFrame(int frame_id, Frame& frame);

    const Calibration& calibration() const { return rectified_calib_; }

    int sideWidth() const { return per_side_size_.width; }
    int sideHeight() const { return per_side_size_.height; }

private:
    bool parseConf();
    bool buildRectification();

    static std::string sectionSuffix(Resolution r);
    static cv::Size perSideSize(Resolution r);

    std::string conf_path_;
    int camera_index_;
    Resolution resolution_;

    cv::Mat K_left_, K_right_;
    cv::Mat D_left_, D_right_;
    cv::Mat R_lr_;
    cv::Mat T_lr_;

    cv::Mat R1_, R2_, P1_, P2_, Q_;
    cv::Mat map_left_x_, map_left_y_;
    cv::Mat map_right_x_, map_right_y_;

    cv::Size per_side_size_;
    cv::Size full_frame_size_;

    cv::VideoCapture cap_;
    Calibration rectified_calib_;
};
