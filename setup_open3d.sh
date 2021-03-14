#!/bin/bash -x

# Expects appropriate virtualenv is activated and $(which python) point to it
# If the script breaks, install manually from http://www.open3d.org/docs/release/compilation.html

git clone --recursive https://github.com/intel-isl/Open3D

cd Open3D
mkdir build

if [ "$(uname)" == "Darwin" ]; then
    # Do something under Mac OS X platform
    cmake -DCMAKE_INSTALL_PREFIX=./build ..
    make -j$(sysctl -n hw.physicalcpu)
    make install-pip-package
    python -c "import open3d"

elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Do something under GNU/Linux platform
    util/install_deps_ubuntu.sh
    cmake -DCMAKE_INSTALL_PREFIX=./build ..
    make -j$(sysctl -n hw.physicalcpu)
    make install-pip-package
    python -c "import open3d"
fi
