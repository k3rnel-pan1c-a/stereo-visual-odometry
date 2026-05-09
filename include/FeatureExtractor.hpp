#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

class FeatureExtractor {
public:
    explicit FeatureExtractor(int nfeatures);

    void extract(
        const cv::Mat& image,
        std::vector<cv::KeyPoint>& keypoints,
        cv::Mat& descriptors
    );

private:
    cv::Ptr<cv::ORB> orb_;
};