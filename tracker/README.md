# Under the path, there are 3 sub-projects:
+ generate training data for instanceSeg and video tracking

+ API used to predict panoptic segmentation from raw data + depth data

+ API used to generate consistent object ID from video sequences.

# Prepare Dataset for training:
    1. follow the README.md from '../' to setup environment and prepare data for generating new data
    2. install extra packages:
        pip install fonts
        pip install font-fredoka-one
        pip install font-amatic-sc
        pip install easy-dict
        
    3. Run the 'simple_task.py' to generate new data.

# API for instance Segmentation
    1. The running is based on python3 + pytorch
