# MCS Evaluation3: Usage README

Here are the instructions for downloading and setting up the environment for MCS Evaluation 3.

### Note for Developers:
The master branch contains only the submission-ready code. For the latest in-progress code, check out the develop branch and any component-level branches (e.g. component/physics-voe, component/agency-voe, component/gravity, etc.).

```
git checkout develop (or any component-level branch)
git pull
```

### Python Library

1. For Linux systems, create a Python 3.6.8 environment using conda:

```
conda create -n mcs_eval3 python=3.6.8
conda activate mcs_eval3
pip install --upgrade pip setuptools wheel
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install python3-dev
```

2. For MacOS systems, create a Python 3.7.9 environment using conda:

```
conda create -n mcs_eval3 python=3.7.9
conda activate mcs_eval3
```

3. Install the required third-party Python libraries:

```
pip install -r requirements.txt
```

4. Ensure you've installed `ai2thor` version `2.2.0`:

```
pip show ai2thor
```

5. Set up torch and torchvision:

```
bash setup_torch.sh
```

### Mac Users: Install Homebrew and `wget`

The `wget` command does not come built-in on macOS, and you will need to install `wget` if you do not already have it installed on your machine. The instructions below utilize the Homebrew package manager for installing `wget`.

```
# Skip the first line if you already have Homebrew installed on your Mac.

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install wget
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
python simple_eval3_agent.py
```

The "MCS EVAL 3 Oregon State University Submission Helper.txt" has more details on how to run specific scenes

## Run Gravity Scenes

```
./setup_unity.sh
python get_gravity_scenes.py
python simple_eval3_agent.py --scenes gravity_scenes
```

The gravity scenes require unity at least v3.7. Running setup_unity.sh should download it and set it to be used in unity_path.yaml

## Approach: `oracle-rule_based-_:nice_drop_config`

### Data level
    - Oracle

### Meta used
    - `supporting_object->dimensions`
    - `target_object->dimensions`
    - `pole_object->texture_color_list`

### Summary
* Based on the above data, the target & supporting object's position & dimensions are known throughout the episode. 
* I found out the dimensions of contact surfaces (bottom face of target & top face of support).
* I calculated if the center of gravity of the target's contact surface fell with-in the dimensions of the support's contact surface.

### Assumptions
* Objects are assumed to be rigid with uniform mass density
* Supporting object is assumed to be at rest
* Law of conservation of energy is ignored
* Accn. due to gravity & target object velocity are along the "y" direction
* Target object would fall "nicely" (angular acceleration = 0 & contact surface's normal is along "y" direction)

