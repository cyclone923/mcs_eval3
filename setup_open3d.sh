#!/bin/bash -x

# Expects appropriate virtualenv is activated and $(which python) point to it
# Only works with Python 3.6 on Linux & 3.8 on OSX
# If the script breaks, 
# 1. Check if you are using the right dev wheel for your corresponding Python version here:
# http://www.open3d.org/docs/latest/getting_started.html
# 2. Or install from source: http://www.open3d.org/docs/release/compilation.html

if [ "$(uname)" == "Darwin" ]; then
    # Do something under Mac OS X platform
    # For python 3.8
    pip install --user https://storage.googleapis.com/open3d-releases-master/python-wheels/open3d-0.12.0+50148ce-cp38-cp38-macosx_10_14_x86_64.whl

elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Do something under GNU/Linux platform
    # For python 3.6
    pip install --user https://storage.googleapis.com/open3d-releases-master/python-wheels/open3d-0.12.0+50148ce-cp36-cp36m-linux_x86_64.whl
fi

python -c "import open3d"