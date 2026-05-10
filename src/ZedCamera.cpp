#include "ZedCamera.hpp"

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

ZedCamera::ZedCamera(
    const std::string& conf_path,
    int camera_index,
    Resolution resolution
)
    : conf_path_(conf_path),
      camera_index_(camera_index),
      resolution_(resolution),
      per_side_size_(perSideSize(resolution)),
      full_frame_size_(per_side_size_.width * 2, per_side_size_.height) {}

cv::Size ZedCamera::perSideSize(Resolution r) {
    switch (r) {
        case Resolution::VGA: return cv::Size(672, 376);
        case Resolution::HD:  return cv::Size(1280, 720);
        case Resolution::FHD: return cv::Size(1920, 1080);
        case Resolution::K2:  return cv::Size(2208, 1242);
    }
    return cv::Size(1280, 720);
}

std::string ZedCamera::sectionSuffix(Resolution r) {
    switch (r) {
        case Resolution::VGA: return "VGA";
        case Resolution::HD:  return "HD";
        case Resolution::FHD: return "FHD";
        case Resolution::K2:  return "2K";
    }
    return "HD";
}

static std::string trim(const std::string& s) {
    auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

bool ZedCamera::parseConf() {
    std::ifstream file(conf_path_);
    if (!file.is_open()) {
        std::cerr << "ZedCamera: failed to open " << conf_path_ << std::endl;
        return false;
    }

    std::map<std::string, std::map<std::string, double>> sections;
    std::string current_section;

    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == ';' || line[0] == '#') continue;

        if (line.front() == '[' && line.back() == ']') {
            current_section = line.substr(1, line.size() - 2);
            continue;
        }

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        try {
            sections[current_section][key] = std::stod(val);
        } catch (...) {
            // skip non-numeric entries
        }
    }

    const std::string suffix = sectionSuffix(resolution_);
    const std::string left_key  = "LEFT_CAM_"  + suffix;
    const std::string right_key = "RIGHT_CAM_" + suffix;

    if (sections.count(left_key) == 0 || sections.count(right_key) == 0
        || sections.count("STEREO") == 0) {
        std::cerr << "ZedCamera: missing required sections in " << conf_path_
                  << " (need [" << left_key << "], [" << right_key << "], [STEREO])"
                  << std::endl;
        return false;
    }

    auto& L = sections[left_key];
    auto& R = sections[right_key];
    auto& S = sections["STEREO"];

    K_left_ = (cv::Mat_<double>(3, 3) <<
        L["fx"], 0,       L["cx"],
        0,       L["fy"], L["cy"],
        0,       0,       1);
    K_right_ = (cv::Mat_<double>(3, 3) <<
        R["fx"], 0,       R["cx"],
        0,       R["fy"], R["cy"],
        0,       0,       1);

    D_left_  = (cv::Mat_<double>(1, 5) <<
        L["k1"], L["k2"], L["p1"], L["p2"], L["k3"]);
    D_right_ = (cv::Mat_<double>(1, 5) <<
        R["k1"], R["k2"], R["p1"], R["p2"], R["k3"]);

    double rx = S.count("RX_" + suffix) ? S["RX_" + suffix] : 0.0;
    double cv_ = S.count("CV_" + suffix) ? S["CV_" + suffix] : 0.0;
    double rz = S.count("RZ_" + suffix) ? S["RZ_" + suffix] : 0.0;

    cv::Mat rvec = (cv::Mat_<double>(3, 1) << rx, cv_, rz);
    cv::Rodrigues(rvec, R_lr_);

    double baseline_mm = S.count("Baseline") ? S["Baseline"] : 120.0;
    double ty_mm = S.count("TY") ? S["TY"] : 0.0;
    double tz_mm = S.count("TZ") ? S["TZ"] : 0.0;

    T_lr_ = (cv::Mat_<double>(3, 1) <<
        baseline_mm / 1000.0,
        ty_mm       / 1000.0,
        tz_mm       / 1000.0);

    return true;
}

bool ZedCamera::buildRectification() {
    cv::stereoRectify(
        K_left_, D_left_, K_right_, D_right_,
        per_side_size_, R_lr_, T_lr_,
        R1_, R2_, P1_, P2_, Q_,
        cv::CALIB_ZERO_DISPARITY, 0, per_side_size_
    );

    cv::initUndistortRectifyMap(
        K_left_, D_left_, R1_, P1_,
        per_side_size_, CV_32FC1,
        map_left_x_, map_left_y_
    );
    cv::initUndistortRectifyMap(
        K_right_, D_right_, R2_, P2_,
        per_side_size_, CV_32FC1,
        map_right_x_, map_right_y_
    );

    double fx = P1_.at<double>(0, 0);
    double fy = P1_.at<double>(1, 1);
    double cx = P1_.at<double>(0, 2);
    double cy = P1_.at<double>(1, 2);
    double baseline = std::abs(P2_.at<double>(0, 3) / P2_.at<double>(0, 0));

    rectified_calib_.fx = fx;
    rectified_calib_.fy = fy;
    rectified_calib_.cx = cx;
    rectified_calib_.cy = cy;
    rectified_calib_.baseline = baseline;
    rectified_calib_.K = (cv::Mat_<double>(3, 3) <<
        fx, 0,  cx,
        0,  fy, cy,
        0,  0,  1);
    rectified_calib_.P0 = P1_.clone();
    rectified_calib_.P1 = P2_.clone();

    return true;
}

bool ZedCamera::open() {
    if (!parseConf()) return false;
    if (!buildRectification()) return false;

    cap_.open(camera_index_, cv::CAP_V4L2);
    if (!cap_.isOpened()) {
        cap_.open(camera_index_);
    }
    if (!cap_.isOpened()) {
        std::cerr << "ZedCamera: failed to open camera index " << camera_index_ << std::endl;
        return false;
    }

    cap_.set(cv::CAP_PROP_FOURCC,
             cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
    cap_.set(cv::CAP_PROP_FRAME_WIDTH,  full_frame_size_.width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, full_frame_size_.height);
    cap_.set(cv::CAP_PROP_FPS, 30);
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);

    int actual_w = (int)cap_.get(cv::CAP_PROP_FRAME_WIDTH);
    int actual_h = (int)cap_.get(cv::CAP_PROP_FRAME_HEIGHT);
    if (actual_w != full_frame_size_.width || actual_h != full_frame_size_.height) {
        std::cerr << "ZedCamera: WARNING requested "
                  << full_frame_size_.width << "x" << full_frame_size_.height
                  << " but driver returned "
                  << actual_w << "x" << actual_h
                  << ". Frames will be resized; rectification quality may suffer."
                  << std::endl;
    }

    std::cout << "ZedCamera: opened camera " << camera_index_
              << " at " << per_side_size_.width << "x" << per_side_size_.height
              << " per side, baseline " << rectified_calib_.baseline << " m"
              << std::endl;

    return true;
}

bool ZedCamera::loadFrame(int frame_id, Frame& frame) {
    cv::Mat raw;
    if (!cap_.read(raw) || raw.empty()) {
        return false;
    }

    if (raw.cols != full_frame_size_.width || raw.rows != full_frame_size_.height) {
        cv::resize(raw, raw, full_frame_size_);
    }

    int half_w = per_side_size_.width;
    cv::Mat left_raw  = raw(cv::Rect(0,      0, half_w, per_side_size_.height));
    cv::Mat right_raw = raw(cv::Rect(half_w, 0, half_w, per_side_size_.height));

    cv::Mat left_rect, right_rect;
    cv::remap(left_raw,  left_rect,  map_left_x_,  map_left_y_,  cv::INTER_LINEAR);
    cv::remap(right_raw, right_rect, map_right_x_, map_right_y_, cv::INTER_LINEAR);

    cv::Mat left_gray, right_gray;
    if (left_rect.channels() == 3) {
        cv::cvtColor(left_rect,  left_gray,  cv::COLOR_BGR2GRAY);
        cv::cvtColor(right_rect, right_gray, cv::COLOR_BGR2GRAY);
    } else {
        left_gray  = left_rect;
        right_gray = right_rect;
    }

    frame.id = frame_id;
    frame.left_img  = left_gray;
    frame.right_img = right_gray;
    return true;
}
