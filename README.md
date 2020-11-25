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

1. Set up the unity application

```
bash setup_unity
```


### To run everything from the project root

1. Export the project root directory to the $PYTHONPATH:

```
export PYTHONPATH=$PWD
```


## Run dry run submission

```
python simple_eval3_agent.py
```



