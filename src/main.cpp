#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "KittiDataset.hpp"
#include "Calibration.hpp"
#include "Frame.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n";
        std::cerr << "  ./stereo_vo <sequence_path>\n\n";
        std::cerr << "Example:\n";
        std::cerr << "  ./stereo_vo ../data/KITTI/dataset/sequences/00\n";
        return 1;
    }

    std::string sequence_path = argv[1];
    std::string calib_path = sequence_path + "/calib.txt";

    Calibration calib;

    if (!calib.loadFromFile(calib_path)) {
        std::cerr << "Failed to load calibration." << std::endl;
        return 1;
    }

    calib.print();

    KittiDataset dataset(sequence_path);

    std::cout << "\nNumber of stereo frames: " << dataset.size() << std::endl;

    if (dataset.size() == 0) {
        std::cerr << "No frames found." << std::endl;
        return 1;
    }

    Frame frame0;

    if (!dataset.loadFrame(0, frame0)) {
        std::cerr << "Failed to load frame 0." << std::endl;
        return 1;
    }

    std::cout << "\nLoaded frame: " << frame0.id << std::endl;
    std::cout << "Left image size: "
              << frame0.left_img.cols << " x " << frame0.left_img.rows << std::endl;
    std::cout << "Right image size: "
              << frame0.right_img.cols << " x " << frame0.right_img.rows << std::endl;

    cv::Mat side_by_side;
    cv::hconcat(frame0.left_img, frame0.right_img, side_by_side);

    cv::imshow("KITTI Stereo Pair - Frame 0", side_by_side);
    cv::waitKey(0);

    return 0;
}