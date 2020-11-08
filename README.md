# MCS Evaluation3: Usage README

## Download

Here are the instructions for downloading and setting up environment for MCS evaluation 3.

### Python Library


1. Install the required third-party Python libraries:

```
pip install -r requirements.txt
```

2. Ensure you've installed `ai2thor` version `2.2.0`:

```
pip show ai2thor
```


### Unity Application

The latest release of the MCS Unity app is `0.3.3`

1. Ensure that both the Unity App and the TAR are in the same directory called "unity_app".

```
mkdir unity_app
cd unity_app
```

2. If you have a Linux system

[Download the Latest MCS Unity App](https://github.com/NextCenturyCorporation/MCS/releases/download/0.3.3/MCS-AI2-THOR-Unity-App-v0.3.3.x86_64)

```
wget https://github.com/NextCenturyCorporation/MCS/releases/download/0.3.3/MCS-AI2-THOR-Unity-App-v0.3.3.x86_64
```

[Download the Latest MCS Unity Data Directory TAR](https://github.com/NextCenturyCorporation/MCS/releases/download/0.3.3/MCS-AI2-THOR-Unity-App-v0.3.3_Data.tar.gz)

```
wget https://github.com/NextCenturyCorporation/MCS/releases/download/0.3.3/MCS-AI2-THOR-Unity-App-v0.3.3_Data.tar.gz
```

Untar the Data Directory:

```
tar -xzvf MCS-AI2-THOR-Unity-App-v0.3.3_Data.tar.gz
```

Mark the Unity App as executable:

```
chmod a+x MCS-AI2-THOR-Unity-App-v0.3.3.x86_64
```

3. If you have a Mac system

[Download the Latest MCS Unity App](https://github.com/NextCenturyCorporation/MCS/releases/download/0.3.3/MCS-AI2-THOR-Unity-App-v0.3.3-mac.zip)

```
wget https://github.com/NextCenturyCorporation/MCS/releases/download/0.3.3/MCS-AI2-THOR-Unity-App-v0.3.3-mac.zip
```
Unzip the unity app:

```
unzip MCS-AI2-THOR-Unity-App-v0.3.3-mac.zip
```

4. Interaction tasks need a Metric-ff planer to run. This repo contains both a Mac and a Linux version. You can refer to the linked website for the source code and compile it for your own system.

(https://fai.cs.uni-saarland.de/hoffmann/ff.html)

5. Go to the project root directory and download the datasets:

```
cd ../
bash get_dataset.sh
```

6. Export the project root directory to the $PYTHONPATH:

```
export PYTHONPATH=$PWD
```
## Run with Agent Task

```
python agent_task.py
```
![alt-text](https://github.com/cyclone923/mcs_eval3/blob/master/demo_0.gif)

## Run with Interaction Task

```
python simple_task.py
```
![alt-text](https://github.com/cyclone923/mcs_eval3/blob/master/demo_1.gif)

## Run with Intuitive Physics Task

```
python int_phy_explain.py
```
![alt-text](https://github.com/cyclone923/mcs_eval3/blob/master/demo_2.gif)

## Run with Human Control Task

| Key | Action | Comment | 
| ------------ | ------------- | ------- |
| w | MoveAhead |  |
| s | MoveBack |  |
| a | MoveLeft |  |
| d | MoveRight |  |
| q | RotateLeft |  |
| e | RotateRight |  |
| r | LookUp |  |
| f | LookDown |  |
| U | PickUpObject | (x,y) coord of pixel of image will be asked to enter |
| N | OpenObject | (x,y) coord of pixel of image will be asked to enter |
```
python human_control.py
```
![alt-text](https://github.com/cyclone923/mcs_eval3/blob/master/demo_3.gif)
