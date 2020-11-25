#!/bin/bash -x

if [ "$(uname)" == "Darwin" ]; then
    # Do something under Mac OS X platform
    pip install torch==1.5.1 torchvision==0.6.1
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Do something under GNU/Linux platform
    pip install torch==1.5.1+cu92 torchvision==0.6.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html
fi
