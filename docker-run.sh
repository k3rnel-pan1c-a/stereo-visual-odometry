#!/usr/bin/env bash
#
# Run the stereo_vo container with X11 forwarding so the Pangolin window
# appears on the host display (Linux).
#
# Usage:
#   ./docker-run.sh <kitti_dataset_root> [<sequence_id>]
#
# The first argument is the directory that contains "sequences/" and "poses/"
# (i.e. the KITTI Odometry "dataset" folder). Sequence id defaults to 07.
#
# Example:
#   ./docker-run.sh ~/datasets/kitti_odometry/dataset 07
#
# To get an interactive shell instead of running the binary:
#   ./docker-run.sh ~/datasets/kitti_odometry/dataset shell

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <kitti_dataset_root> [<sequence_id> | shell]" >&2
    echo "  <kitti_dataset_root> must contain 'sequences/' and 'poses/'." >&2
    exit 1
fi

DATASET="$(realpath "$1")"
SEQ="${2:-07}"

if [[ ! -d "$DATASET/sequences" || ! -d "$DATASET/poses" ]]; then
    echo "ERROR: $DATASET does not contain both 'sequences/' and 'poses/'." >&2
    exit 1
fi

OUTPUTS="$(pwd)/outputs"
mkdir -p "$OUTPUTS"

# Allow the local Docker daemon to talk to the X server.
# This is reverted on most setups when you log out; harmless to repeat.
xhost +local:docker > /dev/null

DOCKER_ARGS=(
    --rm -it
    -e DISPLAY="$DISPLAY"
    -e QT_X11_NO_MITSHM=1
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw
    -v "$DATASET":/data:ro
    -v "$OUTPUTS":/app/outputs:rw
    --device /dev/dri
)

if [[ "$SEQ" == "shell" ]]; then
    docker run "${DOCKER_ARGS[@]}" stereo_vo bash
else
    docker run "${DOCKER_ARGS[@]}" stereo_vo \
        ./stereo_vo "/data/sequences/$SEQ" "/data/poses/$SEQ.txt"
fi
