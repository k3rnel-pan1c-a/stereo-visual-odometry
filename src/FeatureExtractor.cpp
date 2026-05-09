#include "FeatureExtractor.hpp"

FeatureExtractor::FeatureExtractor(int nfeatures) {
    orb_ = cv::ORB::create(nfeatures);
}

void FeatureExtractor::extract(
    const cv::Mat& image,
    std::vector<cv::KeyPoint>& keypoints,
    cv::Mat& descriptors
) {
    orb_->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}