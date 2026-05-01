set shell := ["bash", "-cu"]

build:
    cmake --build build

configure:
    mkdir -p build
    cd build && cmake ..

run name:
    ./build/{{name}}
