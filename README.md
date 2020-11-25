# MCS Evaluation3: Usage README

## Download

Here are the instructions for downloading and setting up environment for MCS evaluation 3.

### Python Library

1. Create a python 3.6.8 using conda:

```
conda create -n mcs_eval3 python=3.6.8
conda activate mcs_eval3
```


2. Install the required third-party Python libraries:

```
pip install -r requirements.txt
```

3. Ensure you've installed `ai2thor` version `2.2.0`:

```
pip show ai2thor
```

4. Set up torch and torchvision:

```
bash setup_torch.sh
```

### Unity Application

Set up the unity application

```
bash setup_unity
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