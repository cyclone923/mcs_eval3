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
        
    3. to generate new data:
        - generate data for interactive scenes 
            ```Shell
            python vision/generateData/simple_task.py
            ```
        - generate data for intphy scenes 
            ```Shell
            python vision/generateData/int_phy_explain.py
            ```
        - statistical on different scenes:
        | scenes                          | object list  |
        |---------------------------------|------------------------------------------------------|
        | interactive scene, transferral  | {'all': 630579, 'trophy': 97931, 'box': 226443, 'changing table': 19098, 'drawer': 38196, 'shelf': 58078, 'blank block cube': 32195, 'plate': 16745, 'duck': 5834, 'sofa chair': 6542, 'bowl': 13004, 'pacifier': 4971, 'crayon': 5255, 'number block cube': 18766, 'sphere': 24839, 'chair': 4753, 'sofa': 10265, 'stool': 4533, 'car': 4957, 'blank block cylinder': 16412, 'cup': 14546, 'apple': 5381, 'table': 691, 'crib': 327, 'potted plant': 817}|
        | intphy scene, object permanance | {'all': 33489, 'trophy': 0, 'box': 0, 'duck': 3487, 'cylinder': 2258, 'turtle': 2823, 'car': 7311, 'sphere': 3870, 'cube': 7525, 'cone': 3438, 'square frustum': 2777} |
        | intphy scene, shape   constancy | {'all': 29134, 'trophy': 0, 'box': 0, 'square frustum': 2502, 'cube': 6085, 'car': 5952, 'cylinder': 2227, 'sphere': 3472, 'turtle': 2653, 'cone': 3360, 'duck': 2883}|
        | intphy scene, spatio temporal continuity| {'all': 31803, 'trophy': 0, 'box': 0, 'cylinder': 6951, 'car': 9900, 'sphere': 5073, 'turtle': 4395, 'duck': 5484}|


# API for instance Segmentation
    1. The running is based on python3 + pytorch. Install dependencies:
        ```Shell
        pip install scipy==1.2.0
        ```
    
    2. Compile deformable convolutional layers (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_1.0)). Make sure you have the latest CUDA toolkit installed from [NVidia's Website](https://developer.nvidia.com/cuda-toolkit).
        ```Shell
        cd vision/instSeg/external/DCNv2
        python setup.py build develop
        cd ../../
        ```
    3. download the weights file and save under 'tracker/instSeg/'.
        + Interact transferal scene
            ``` Shell
            wget https://oregonstate.box.com/shared/static/0syjouwkkpm0g1zbnt1riheshfvtd2by.pth 
            mv 0syjouwkkpm0g1zbnt1riheshfvtd2by.pth dvis_resnet50_mc.pth 
            ```
        + Intphy scenes (VOE scenes)
            ``` Shell
            wget https://oregonstate.box.com/shared/static/zmvcjyumltkziqfqbcqodkh6dgecikci.pth
            mv zmvcjyumltkziqfqbcqodkh6dgecikci.pth dvis_resnet50_mc_voe.pth 
            ```
        + Currently, only 'mcsvideo3_inter' and 'mcsvideo3_voe' are supported. And you could check
            the './vision/instSeg/data/config_xx.py' file to see what are the BG categories and 
            what are the FG categories detected in it.
        
    4. Run the demo test
        ```Shell
        python vision/instSeg/inference.py
        cd ../../
        ```

    5. Call the API from path.
        + Add './vision' to the system path
            - In script: 'import sys  sys.path.append('./vision')'
            - In command line: export PYTHONPATH=$PYTHONPATH:./vision
    
        ```Shell
        from instSeg.inference import MaskAndClassPredictor as Predictor
        model = Predictor(dataset='mcsvideo3_inter', weights='PATH_TO_PTH_FILE')
        # prepare bgrI, depthI
        ret = model(bgrI, depthI)
        ```
        
