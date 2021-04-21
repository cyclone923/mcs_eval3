#!/bin/bash -x

mkdir -p unity_app
cd unity_app

if [ "$(uname)" == "Darwin" ]; then
    # Do something under Mac OS X platform
    wget https://github.com/NextCenturyCorporation/MCS/releases/download/0.4.1/MCS-AI2-THOR-Unity-App-v0.4.1-mac.zip
    unzip MCS-AI2-THOR-Unity-App-v0.4.1-mac.zip
    rm MCS-AI2-THOR-Unity-App-v0.4.1-mac.zip
    echo unity_path: \'unity_app/MCSai2thor.app/Contents/MacOS/MCSai2thor\' > ../unity_path.yaml
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Do something under GNU/Linux platform
    wget https://github.com/NextCenturyCorporation/MCS/releases/download/0.4.1/MCS-AI2-THOR-Unity-App-v0.4.1.x86_64
    wget https://github.com/NextCenturyCorporation/MCS/releases/download/0.4.1/MCS-AI2-THOR-Unity-App-v0.4.1_Data.tar.gz
    tar -xzvf MCS-AI2-THOR-Unity-App-v0.4.1_Data.tar.gz
    rm MCS-AI2-THOR-Unity-App-v0.4.1_Data.tar.gz
    chmod a+x MCS-AI2-THOR-Unity-App-v0.4.1.x86_64
    echo unity_path: \'unity_app/MCS-AI2-THOR-Unity-App-v0.4.1.x86_64\' > ../unity_path.yaml

fi

echo [MCS] > ../mcs_config.ini
echo debug = false >> ../mcs_config.ini
echo metadata = oracle >> ../mcs_config.ini
cd ../
