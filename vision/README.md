# Under the path, there are 3 sub-projects:
+ generate training data for instanceSeg and video tracking

+ API used to predict panoptic segmentation from raw data + depth data
    - input in numpy array, raw image with shape [ht, wd, ch] in BGR colorspace + depth image with shape [ht, wd]
    - output is a dictionary contains:
        * mask_prob --  array in shape [fg_stCh+num_objs, ht, wd]
        * fg_stCh -- the start channel of objects in mask_prob, with that,
           + channels before fg_stCh is for semantic segmentation of BG pixels
           + channels afterwards fg_stCh is for object segmentation, with each channel has one object segmented.
        * obj_class_score -- array in shape [num_objs, num_fg_classes+1]
        * net-mask -- array in shape [ht, wd],  the mask prediction the network mask head outputs. class score is not applied on this output.

+ API used to generate consistent object ID from video sequences.
    - to be added:: todo::

# Prepare Dataset for training:
    1. follow the README.md from '../' to setup environment and prepare data for generating new data
    2. install extra packages:
        ```Shell
        pip install fonts
        pip install font-fredoka-one
        pip install font-amatic-sc
        pip install easy-dict
        ```
        
    3. Run the 'simple_task.py' to generate new data.

# API for instance Segmentation
    1. The running is based on python3 + pytorch. Install dependencies:
        ```Shell
        pip install scipy==1.2.0
        ```
    
    2. Compile deformable convolutional layers (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_1.0)). Make sure you have the latest CUDA toolkit installed from [NVidia's Website](https://developer.nvidia.com/cuda-toolkit).
        ```Shell
        cd tracker/instSeg/external/DCNv2
        python setup.py build develop
        cd ../../
        ```
    3. download the weights file and save under 'tracker/instSeg/'.
        ``` Shell
        wget https://oregonstate.box.com/shared/static/0syjouwkkpm0g1zbnt1riheshfvtd2by.pth 
        mv 0syjouwkkpm0g1zbnt1riheshfvtd2by.pth dvis_resnet50_mc.pth 
        ```
        
    4. Run the demo test
        ```Shell
        python inference.py
        cd ../../
        ```

    5. Call the API from path.
        + Add './tracker' to the system path
            - In script: 'import sys  sys.path.append('./tracker')'
            - In command line: export PYTHONPATH=$PYTHONPATH:./tracker
    
        ```Shell
        from instSeg.inference import MaskAndClassPredictor as Predictor
        model = Predictor(dataset='mcsvideo3_inter')
        # prepare bgrI, depthI
        ret = model(bgrI, depthI)
        ```
        
