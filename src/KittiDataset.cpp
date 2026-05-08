#include "KittiDataset.hpp"

#include <filesystem>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

KittiDataset::KittiDataset(const std::string& sequence_path)
    : sequence_path_(sequence_path) {
    loadImagePaths();
}

void KittiDataset::loadImagePaths() {
    std::string left_dir = sequence_path_ + "/image_2";
    std::string right_dir = sequence_path_ + "/image_3";

    if (!fs::exists(left_dir)) {
        std::cerr << "Left image directory does not exist: " << left_dir << std::endl;
        return;
    }

    if (!fs::exists(right_dir)) {
        std::cerr << "Right image directory does not exist: " << right_dir << std::endl;
        return;
    }

    for (const auto& entry : fs::directory_iterator(left_dir)) {
        if (entry.path().extension() == ".png") {
            left_image_paths_.push_back(entry.path().string());
        }
    }

    for (const auto& entry : fs::directory_iterator(right_dir)) {
        if (entry.path().extension() == ".png") {
            right_image_paths_.push_back(entry.path().string());
        }
    }

    std::sort(left_image_paths_.begin(), left_image_paths_.end());
    std::sort(right_image_paths_.begin(), right_image_paths_.end());

    if (left_image_paths_.size() != right_image_paths_.size()) {
        std::cerr << "Warning: left/right image count mismatch." << std::endl;
    }
}

bool KittiDataset::loadFrame(int index, Frame& frame) const {
    if (index < 0 || index >= size()) {
        std::cerr << "Invalid frame index: " << index << std::endl;
        return false;
    }

    cv::Mat left = cv::imread(left_image_paths_[index], cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread(right_image_paths_[index], cv::IMREAD_GRAYSCALE);

    if (left.empty() || right.empty()) {
        std::cerr << "Failed to load image pair at index: " << index << std::endl;
        return false;
    }

    frame = Frame(index, left, right);
    return true;
}

std::size_t KittiDataset::size() const {
    return static_cast<int>(std::min(left_image_paths_.size(), right_image_paths_.size()));
}