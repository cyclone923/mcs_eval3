# Deep Variational Instance Segmetation
```
    ████████║   ██         ██  ██████████    █████████║
    ██      █║   █         █      ║██║       █║
    ██       █║   █       █       ║██║       █████████║
    ██       █║    █     █        ║██║              ██║
    ██      █║      █   █         ║██║              ██║
    ███████║        █████      ██████████    █████████

```

A simple, fully convolutional model for real-time instance segmentation. This is the code for our papers:
 - [Deep Variational Instance Segmentation](https://arxiv.org/abs/2007.11576)

The implementation is based on YOLACT's implementation at [Yolact-github](https://github.com/dbolya/yolact)


# Installation
 - Clone this repository and enter it:
   ```Shell
   git clone  xxx.git
   cd xxx
   ```
 - Set up the environment using one of the following methods:
   - Using [Anaconda](https://www.anaconda.com/distribution/)
     - Run `conda env create -f environment.yml`
   - Manually with pip
     - Set up a Python3 environment (e.g., using virtenv).
     - Install [Pytorch](http://pytorch.org/) and TorchVision.
       ```Shell
        pip install torch==1.5.1+cu92 torchvision==0.6.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html
        pip install torchvision==0.4.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html
       ```
     - Install some other packages:
       ```Shell
       # Cython needs to be installed before pycocotools
       pip install cython
       pip install opencv-python pillow pycocotools matplotlib 
       pip install scikit-image
       pip install scipy==1.2.0
       ```
 - Config the dataset related setting regardding to the 'data/mcsvideo.py' and 'data/config_mcsvideo*.py'.
 - For the 'mcsvideo' dataset, data can be generated from the simulator by running '../vision/generateData/simle_task.py'. Data are arranged as:
    + RGB/Depth/Instance/Semantic images are saved with name: 
        ' original_*.jpg |  'depth_*.png | inst_*.png  | cls_*.png '

    + train.txt | val.txt lists all relative path of the generated scenes to be used. i.e.
    ```Shell
    intphy_scenes/object_permanence/scene_469
    intphy_scenes/object_permanence/scene_470
    intphy_scenes/object_permanence/scene_471
    intphy_scenes/object_permanence/scene_472
    intphy_scenes/shape_constancy/scene_1
    intphy_scenes/shape_constancy/scene_2
    intphy_scenes/shape_constancy/scene_3
    intphy_scenes/shape_constancy/scene_5
    intphy_scenes/shape_constancy/scene_6
    intphy_scenes/shape_constancy/scene_7
    intphy_scenes/shape_constancy/scene_8
    intphy_scenes/shape_constancy/scene_9
    intphy_scenes/shape_constancy/scene_10
    ```
        
 
 - To use DCN in backbone network, compile deformable convolutional layers (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_1.0)).
   Make sure you have the latest CUDA toolkit installed from [NVidia's Website](https://developer.nvidia.com/cuda-toolkit).
   ```Shell
   cd external/DCNv2
   python setup.py build develop
   ```

## Train on your own dataset:
 - You could edit the config_xx.py in data/ to customize the network setting and dataset setting.
 - You could run with specific the arguments on shell command:
   ```Shell
    python train.py --config=plus_resnet50_config_550 --resume=PATH/TO/YOUR/FILE --start_iter=0 --exp_name=dummy     
   ```
 - Or, you could customize the json script in exp_scripts/, and run with:
   ```Shell
    python train.py --scripts=scripts_train/xxx.json
   ```

