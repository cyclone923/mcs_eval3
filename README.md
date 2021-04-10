# MCS OPICS Evaluation Repository: Usage README

Here are the instructions for downloading and setting up the environment for MCS evaluations.

### Note for Developers:
The master branch contains only the submission-ready code. For the latest in-progress code, check out the develop branch and any component-level branches (e.g. component/physics-voe, component/agency-voe, component/gravity, etc.).

```
git checkout develop (or any component-level branch)
git pull
```

### Mac Users: Install Homebrew

```
# NOTE: Skip this if you already have Homebrew installed on your Mac.

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Python Library

1. Create Python environment.

For Linux systems, create a Python 3.6.8 environment using conda:

```
conda create -n mcs_opics python=3.6.8
conda activate mcs_opics
pip install --upgrade pip setuptools wheel
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install python3-dev
```

For MacOS systems, create a Python 3.7.9 environment using conda:

```
conda create -n mcs_opics python=3.7.9
conda activate mcs_opics
brew install wget
```

2. Install MCS

NOTE: This may be a while. Please be patient.

```
python -m pip install git+https://github.com/NextCenturyCorporation/MCS@master#egg=machine_common_sense
```

3. Install the required third-party Python libraries:

```
pip install -r requirements.txt
```

4. Ensure you've installed `ai2thor` version `2.5.0`:

```
pip show ai2thor
```

5. Set up torch and torchvision:

```
bash setup_torch.sh
```

### Unity Application

Set up the Unity application:

```
bash setup_unity.sh
```

### Vision System

Set up object mask and class predictor:

```
bash setup_vision.sh
```

### To run everything from the project root

Export the project root directory to the $PYTHONPATH:

```
export PYTHONPATH=$PWD
```

## Run dry run submission

```
python eval.py
```

See "MCS Eval 3.5 Oregon State University Submission Helper.txt" for more details on how to run specific scenes

## Run Example Gravity Scenes

```
./setup_unity.sh
python get_gravity_scenes.py
python eval.py --scenes gravity_scenes/[rest of the relative path to the directory with the scenes you want to test]
```

The gravity scenes require Unity v0.3.7 or newer. Running setup_unity.sh should download it and set it to be used in unity_path.yaml.
