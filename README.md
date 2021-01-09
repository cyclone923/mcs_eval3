# MCS Evaluation3: Usage README

Here are the instructions for downloading and setting up environment for MCS evaluation 3.

### Python Library

1. For Linux system, create a python 3.6.8 using conda:

```
conda create -n mcs_eval3 python=3.6.8
conda activate mcs_eval3
pip install --upgrade pip setuptools wheel
sudo apt-get install python3-dev
```

2. For MacOS system, create a python 3.7.9 using conda:

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

### Unity Application

Set up the unity application

```
bash setup_unity.sh
```

### Vision System

Set up object mask and class predictor

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
