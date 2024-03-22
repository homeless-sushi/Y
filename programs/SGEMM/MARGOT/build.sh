#!/bin/bash

CWD_DIR="$(pwd)"
SCRIPT_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
cd "${SCRIPT_DIR}"

#rm -rf build
mkdir -p build
(
    cd build && \
    cmake .. && \
    cmake --build . \
)

cd "${CWD_DIR}"