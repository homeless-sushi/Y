#!/bin/bash

CWD_DIR="$(pwd)"
SCRIPT_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
cd "${SCRIPT_DIR}"

rm -rf build
mkdir build
cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBOOST_ROOT="~/Documents/Thesis/margot/boost_1_60_0/install" \
    -S . -B build
cmake --build build

cd "${CWD_DIR}"
