#!/bin/bash

CWD_DIR="$(pwd)"
SCRIPT_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
cd "${SCRIPT_DIR}"

sudo rm -rf agora/NBODY_Profiling
sudo rm -rf agora/observations
sudo rm -rf agora/configurations
sudo rm -rf agora/predictions
sudo rm -rf agora/descriptions
sudo rm -rf agora/clusters
sudo rm -rf agora/models

cd "${CWD_DIR}"
