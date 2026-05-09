# Stereo Visual Odometry — reproducible build environment.
#
# Build:
#   docker build -t stereo_vo .
#
# Run (Linux, with X11 forwarding so the Pangolin window appears on your host):
#   ./docker-run.sh /abs/path/to/kitti
# or manually:
#   xhost +local:docker
#   docker run --rm -it \
#       -e DISPLAY="$DISPLAY" \
#       -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
#       -v /abs/path/to/kitti:/data:ro \
#       -v "$PWD/outputs":/app/outputs:rw \
#       --device /dev/dri \
#       stereo_vo \
#       ./stereo_vo /data/sequences/07 /data/poses/07.txt

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# System build tools, OpenCV, Eigen, and Pangolin's prerequisites.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        pkg-config \
        ca-certificates \
        # OpenCV
        libopencv-dev \
        # Eigen
        libeigen3-dev \
        # Pangolin OpenGL stack
        libgl1-mesa-dev \
        libegl1-mesa-dev \
        libglew-dev \
        libglu1-mesa-dev \
        libepoxy-dev \
        # Wayland + X11 (so the window can be forwarded)
        libxkbcommon-dev \
        libwayland-dev \
        wayland-protocols \
        libx11-dev \
        libxext-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxi-dev \
    && rm -rf /var/lib/apt/lists/*

# Pangolin is not packaged in Ubuntu repos — build from source.
ARG PANGOLIN_VERSION=v0.9
RUN git clone --depth 1 --branch ${PANGOLIN_VERSION} \
        https://github.com/stevenlovegrove/Pangolin.git /tmp/Pangolin && \
    cmake -S /tmp/Pangolin -B /tmp/Pangolin/build \
        -G Ninja \
        -DBUILD_PANGOLIN_PYTHON=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF && \
    cmake --build /tmp/Pangolin/build --parallel && \
    cmake --install /tmp/Pangolin/build && \
    ldconfig && \
    rm -rf /tmp/Pangolin

WORKDIR /app

# Source comes last so dependency layers cache between code edits.
COPY . /app/

# Build the project (Release).
RUN cmake -S /app -B /app/build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /app/build --parallel

# Default working dir matches what the binary expects ("../outputs" is /app/outputs).
WORKDIR /app/build

# Default: print usage. Pass real args (sequence + poses) to actually run.
CMD ["./stereo_vo"]
