#!/usr/bin/env bash

# Copyright (c) 2025 HomomorphicEncryption.org
# All rights reserved.
#
# This software is licensed under the terms of the Apache v2 License.
# See the LICENSE.md file for details.

# ------------------------------------------------------------
# Usage: ./scripts/build_task.sh <TASK_DIR>
# Compiles the files in the source directory.
# ------------------------------------------------------------
set -euo pipefail

# Define core paths
ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
TASK_DIR="$1"
BUILD="$TASK_DIR/build"
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu || echo 4)

# --- 1. LibTorch (PyTorch C++ distribution) ---
LIBTORCH_DIR="$ROOT/third_party/libtorch"
LIBTORCH_ZIP_NAME="libtorch_temp.zip"
LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip"

if [ ! -d "$LIBTORCH_DIR" ]; then
    echo "Downloading LibTorch..."
    mkdir -p "$ROOT/third_party"
    cd "$ROOT/third_party"

    # Use -O to force the output filename and avoid ".zip.1" duplicates
    # We also remove any existing partial downloads first to be safe
    rm -f "$LIBTORCH_ZIP_NAME"
    wget -O "$LIBTORCH_ZIP_NAME" "$LIBTORCH_URL"
    
    echo "Unzipping LibTorch..."
    unzip -q "$LIBTORCH_ZIP_NAME"
    rm "$LIBTORCH_ZIP_NAME"
    
    cd "$ROOT"
    echo "LibTorch successfully set up at $LIBTORCH_DIR"
fi

# --- 2. nlohmann/json ---
NLOHMANN_DIR="$ROOT/third_party/nlohmann"
NLOHMANN_HEADER="$NLOHMANN_DIR/json.hpp"
NLOHMANN_URL="https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp"

if [[ ! -f "$NLOHMANN_HEADER" ]]; then
      echo "Downloading nlohmann/json..."
      mkdir -p "$NLOHMANN_DIR"
      curl -L -o "$NLOHMANN_HEADER" "$NLOHMANN_URL"
fi

# --- 3. Build Process ---
# We assume OpenFHE is in /third_party/openfhe or provided via CMAKE_PREFIX_PATH.
echo "Configuring project with CMake..."
cmake -S "$TASK_DIR" -B "$BUILD" \
      -DCMAKE_PREFIX_PATH="$ROOT/third_party/openfhe;$ROOT/third_party/libtorch"

echo "Compiling with $NPROC cores..."
cd "$BUILD"
make -j"$NPROC"

echo "Build complete."