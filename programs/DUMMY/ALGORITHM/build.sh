#!/bin/bash

CWD_DIR="$(pwd)"
SCRIPT_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
cd "${SCRIPT_DIR}"

rm -rf build
mkdir build
export CXX=/usr/bin/g++-8
(
    cd build && \
    cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBOOST_ROOT="~/Vivian/Thesis/margot/boost_1_60_0/install" \
    .. && \
    cmake --build . \
)

cd "${CWD_DIR}"
