#!/bin/bash

# this script should be executed using source

CWD_DIR="$(pwd)"
SCRIPT_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
cd "${SCRIPT_DIR}"

set -a
source .env
set +a

cd "${CWD_DIR}"