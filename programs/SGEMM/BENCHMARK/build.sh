#!/bin/bash

CWD_DIR="$(pwd)"
SCRIPT_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
cd "${SCRIPT_DIR}"

rm -rf build
mkdir build
(
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Debug && \
    cmake --build . \
)

cd "${CWD_DIR}"