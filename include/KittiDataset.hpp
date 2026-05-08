#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Frame.hpp"

class KittiDataset {
public:
    explicit KittiDataset(const std::string& sequence_path);

    bool loadFrame(int index, Frame& frame) const;

    std::size_t size() const;

private:
    std::string sequence_path_;
    std::vector<std::string> left_image_paths_;
    std::vector<std::string> right_image_paths_;

    void loadImagePaths();
};