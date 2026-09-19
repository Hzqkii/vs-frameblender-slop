#!/usr/bin/env bash
set -e

# Detect available C++ compiler
if command -v clang++ >/dev/null 2>&1; then
    CXX="clang++"
elif command -v g++ >/dev/null 2>&1; then
    CXX="g++"
elif command -v gcc >/dev/null 2>&1; then
    CXX="gcc -x c++"
else
    echo "Error: No suitable C++ compiler (clang++, g++, gcc) found."
    exit 1
fi

echo "Building with $CXX..."
$CXX ./src/blender.cpp -O3 -march=native -shared -fPIC -std=c++17 -I./vapoursynth -lstdc++ -lm -o ./blender.so

echo "Build success: $(pwd)/blender.so"

if [ "$1" = "--install" ]; then
    echo "Installing to /usr/lib/vapoursynth/blender.so..."
    sudo mkdir -p /usr/lib/vapoursynth
    sudo cp ./blender.so /usr/lib/vapoursynth/blender.so
    echo "Installation complete."
fi
