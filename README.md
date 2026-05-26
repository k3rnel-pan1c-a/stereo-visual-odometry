# Stereo Visual Odometry

An ORB-SLAM-inspired stereo visual odometry pipeline implemented from scratch in C++ using OpenCV and Pangolin.

Project page: https://k3rnel-pan1c-a.github.io/stereo-visual-odometry/

## Overview

This project implements a feature-based stereo visual odometry system that estimates the 6-DoF camera trajectory from a rectified stereo image sequence.

The system follows the main frontend ideas used in ORB-SLAM-style pipelines, but keeps the scope focused on visual odometry only. It does not include IMU integration, loop closure, keyframe-based local mapping, or full bundle adjustment. Instead, it focuses on the core multi-view geometry steps:

1. Extract ORB features from the left and right stereo images.
2. Match left-right ORB descriptors using Hamming distance.
3. Filter stereo matches using epipolar and disparity constraints.
4. Recover sparse metric 3D points from stereo disparity.
5. Match features temporally between consecutive left images.
6. Build 3D-2D correspondences.
7. Estimate relative motion using PnP with RANSAC.
8. Accumulate the camera pose into a global trajectory.
9. Visualize the trajectory, sparse map, camera frustums, and tracking state in real time using Pangolin.

## Demo / Project Page

A full write-up, figures, evaluation results, and qualitative visualizations are available here:

https://k3rnel-pan1c-a.github.io/stereo-visual-odometry/

## Features

- Stereo visual odometry from rectified image pairs
- ORB feature extraction using OpenCV
- Sparse stereo matching with epipolar and disparity filtering
- Metric depth recovery using stereo baseline and camera intrinsics
- Frame-to-frame temporal feature tracking
- 3D-2D motion estimation using `cv::solvePnPRansac`
- Global pose accumulation
- KITTI Odometry dataset support
- Quantitative evaluation against KITTI ground-truth poses
- ATE, RPE, and final drift computation
- Real-time Pangolin visualization
- Live ORB keypoint and PnP inlier overlay
- Docker build environment with X11 forwarding support
- Optional ZED stereo camera qualitative pipeline without using the ZED SDK

## Pipeline

```text
Stereo image pair
        |
        v
ORB feature extraction
(left image + right image)
        |
        v
Sparse stereo matching
(epipolar gate + disparity gate)
        |
        v
3D point triangulation
        |
        v
Temporal matching
left image k-1 <-> left image k
        |
        v
3D-2D correspondences
        |
        v
PnP + RANSAC
        |
        v
Relative pose estimate
        |
        v
Global pose accumulation
        |
        v
Trajectory + sparse map visualization
```

## Repository Structure

```text
.
├── include/              # Header files
├── src/                  # Main C++ implementation
├── docs/                 # GitHub Pages project report
├── zed_cam/              # ZED-related capture/utility code
├── CMakeLists.txt        # CMake build configuration
├── Dockerfile            # Reproducible Ubuntu build environment
├── docker-run.sh         # Docker runner with X11 forwarding
├── justfile              # Convenience build/run commands
└── outputs/              # Generated outputs, created at runtime
```

Main modules:

```text
Frame              # Stores images, keypoints, descriptors, depth, and pose
Calibration        # Loads KITTI calibration and camera intrinsics
KittiDataset       # Loads KITTI stereo image sequences
FeatureExtractor   # ORB feature extraction
StereoMatcher      # Left-right matching and sparse depth computation
MotionEstimator    # Temporal matching and PnP + RANSAC
Trajectory         # Pose saving, plotting, and evaluation
Viewer             # Pangolin real-time visualization
```

## Dependencies

Native build dependencies:

- C++17-compatible compiler
- CMake >= 3.10
- OpenCV
- Pangolin
- OpenGL / Mesa development libraries
- Eigen, if needed by your local Pangolin/OpenCV setup

On Ubuntu, most dependencies can be installed with:

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config \
    libopencv-dev \
    libeigen3-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libglew-dev \
    libglu1-mesa-dev \
    libepoxy-dev \
    libxkbcommon-dev \
    libwayland-dev \
    wayland-protocols \
    libx11-dev \
    libxext-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev
```

Pangolin may need to be built from source if it is not available through your package manager.

## Building

Clone the repository:

```bash
git clone https://github.com/k3rnel-pan1c-a/stereo-visual-odometry.git
cd stereo-visual-odometry
```

Configure and build:

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```

Or using the provided `justfile`:

```bash
just configure
just build
```

The main executable is:

```bash
./build/stereo_vo
```

## Dataset Format

The executable expects the KITTI Odometry dataset structure:

```text
dataset/
├── sequences/
│   └── 07/
│       ├── image_0/
│       ├── image_1/
│       └── calib.txt
└── poses/
    └── 07.txt
```

The first argument is the sequence directory, and the second argument is the ground-truth pose file.

## Running on KITTI

Example:

```bash
./build/stereo_vo \
    /path/to/KITTI/dataset/sequences/07 \
    /path/to/KITTI/dataset/poses/07.txt
```

Another example:

```bash
./build/stereo_vo \
    /path/to/KITTI/dataset/sequences/00 \
    /path/to/KITTI/dataset/poses/00.txt
```

The program will:

- Load stereo frames
- Extract ORB features
- Estimate frame-to-frame motion
- Display the live Pangolin viewer
- Save the estimated trajectory
- Save trajectory plots
- Save sample stereo and temporal match visualizations
- Save quantitative evaluation results when ground truth is available

Generated outputs are written to:

```text
outputs/
├── trajectory_est.txt
├── trajectory_plot.png
├── evaluation.txt
├── stereo_matches/
└── temporal_matches/
```

## Running with Docker

Build the Docker image:

```bash
docker build -t stereo_vo .
```

Run using the helper script:

```bash
./docker-run.sh /path/to/KITTI/dataset 07
```

For sequence 00:

```bash
./docker-run.sh /path/to/KITTI/dataset 00
```

To open an interactive shell inside the container:

```bash
./docker-run.sh /path/to/KITTI/dataset shell
```

The Docker runner mounts:

- KITTI dataset as read-only at `/data`
- Local `outputs/` directory to `/app/outputs`
- X11 socket so the Pangolin window appears on the host display

## Evaluation

The project evaluates the estimated trajectory against KITTI ground-truth poses using:

### Absolute Trajectory Error

ATE measures the Euclidean distance between the estimated and ground-truth camera positions at each frame.

### Relative Pose Error

RPE is computed over a fixed 10-frame window and reports both translational and rotational error.

### Final Drift

Final drift is computed as:

```text
Drift (%) = 100 * final_position_error / ground_truth_path_length
```

## Results

| Sequence | Frames | GT Length | ATE RMSE | ATE Mean | ATE Max | RPE Trans. | RPE Rot. | Final Error | Drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KITTI 07 | 1101 | 694.7 m | 9.09 m | 6.95 m | 17.61 m | 0.39 m | 0.39° | 9.99 m | 1.44% |
| KITTI 00 | 4541 | 3724.2 m | 22.96 m | 19.90 m | 48.67 m | 0.29 m | 0.66° | 38.07 m | 1.02% |

These results were achieved without loop closure, bundle adjustment, IMU fusion, or keyframe-based local mapping.

## Visualization

The Pangolin viewer shows:

- Estimated trajectory
- Current camera frustum
- Past camera frustums
- Sparse accumulated 3D map
- Current rectified stereo pair
- ORB keypoints
- PnP inlier keypoints

The keypoint overlay is useful for debugging tracking quality:

- Green points represent detected ORB keypoints.
- Yellow points represent PnP inliers used for motion estimation.

## ZED Stereo Camera Support

The project also includes a qualitative ZED stereo camera experiment.

The ZED camera is used as a generic USB stereo camera through OpenCV `VideoCapture`, without depending on the ZED SDK. Camera calibration parameters are used to rectify the stereo pair before passing it into the same VO pipeline used for KITTI.

This demonstrates that the pipeline can operate on non-KITTI stereo input, although no ground-truth trajectory is available for quantitative evaluation in this case.

## Limitations

This is a visual odometry frontend, not a full SLAM system.

Current limitations:

- No loop closure
- No global pose graph optimization
- No local bundle adjustment
- No keyframe-based local map
- No relocalization after tracking failure
- No IMU integration
- Drift accumulates over long trajectories
- Tracking can degrade in low-texture, blurred, or rapidly rotating frames

When PnP fails, the current implementation reuses the previous pose instead of performing relocalization.

## TODO / Future Work

- [ ] Add a constant-velocity motion model
- [ ] Refine PnP results using motion-only bundle adjustment
- [ ] Add keyframe selection
- [ ] Maintain a sliding-window local map
- [ ] Add local bundle adjustment using Ceres or g2o
- [ ] Add loop closure using DBoW3
- [ ] Add pose graph optimization
- [ ] Improve robustness during motion blur and low-texture scenes
- [ ] Add stereo camera live mode as a first-class runtime option

