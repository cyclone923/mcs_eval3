#!/bin/bash -x

mkdir -p unity_app
cd unity_app

export UNITY_VERSION=0.4.3
# only download if not already installed
if ! ls | grep $UNITY_VERSION; then
    if [ "$(uname)" == "Darwin" ]; then
        # Do something under Mac OS X platform
        wget https://github.com/NextCenturyCorporation/MCS/releases/download/0.4.3/MCS-AI2-THOR-Unity-App-v0.4.3-mac.zip
        unzip -o MCS-AI2-THOR-Unity-App-v0.4.3-mac.zip
        rm MCS-AI2-THOR-Unity-App-v0.4.3-mac.zip
        echo unity_path: \'unity_app/MCSai2thor.app/Contents/MacOS/MCSai2thor\' > ../unity_path.yaml
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
        # Do something under GNU/Linux platform
        wget https://github.com/NextCenturyCorporation/MCS/releases/download/0.4.3/MCS-AI2-THOR-Unity-App-v0.4.3-linux.zip
        unzip -o MCS-AI2-THOR-Unity-App-v0.4.3-linux.zip
        rm MCS-AI2-THOR-Unity-App-v0.4.3-linux.zip
        tar -xzvf MCS-AI2-THOR-Unity-App-v0.4.3_Data.tar.gz
        rm MCS-AI2-THOR-Unity-App-v0.4.3_Data.tar.gz
        chmod a+x MCS-AI2-THOR-Unity-App-v0.4.3.x86_64
        echo unity_path: \'unity_app/MCS-AI2-THOR-Unity-App-v0.4.3.x86_64\' > ../unity_path.yaml
    fi
fi

MCS_CONFIG=../mcs_config.ini
if [ ! -f "$MCS_CONFIG" ]; then
    echo [MCS] > ../mcs_config.ini
    echo debug = false >> ../mcs_config.ini
    echo metadata = oracle >> ../mcs_config.ini
    cd ../
fi

