Download scenes from https://drive.google.com/drive/u/1/folders/1ez1DsR9bk9Cor2VEipGV3JxkgP_zWn75

This contains scenes for different objects for four types physics-voe scenes. Object Permanence, Spatio temporal & Collisions scenes include around 150 to 200 scenes per each object:

- car_1
- cylinder
- duck_on_wheels
- racecar_red
- sphere
- train_1
- trolley_1
- tube_narrow
- tube_wide
- turtle_on_wheels


Shape Constancy includes the below objects and have 140 scenes per object:
- car_1
- circle_frustum
- cone
- cube
- cylinder
- duck_on_wheels
- pyramid
- racecar_red
- sphere
- square_frustum
- train_1
- trolley_1
- tube_narrow
- tube_wide
- turtle_on_wheels


Make sure the scenes of .json format are present in masks-occluder/scenes/ directory before running the simulator to generate training data


Functions to add in main() function of eval4_data_gen.py to generate relevant data:

1. convert_scenes(env, scenes, train_dataset_path) --> For RGBD cropped object images. This was required for siamese based appearance model in tracker


2. get_data_for_tracking(env, paths, train_dataset_path) --> For RGBD frames and track information of objects as per MOT data format.
Running this function will generate data files for scenes present in masks-occluder/scenes/ directory

Data generated from get_data_for_tracking() function gets stored in tracking_data directory. The directory structure is as follows:

tracking_data
    -Depth
        -Scene0
            -depth_0.png
            -depth_1.png
            ...
        -Scene1
            -depth_0.png
            ...
        ...
    -RGB
        -Scene0
            -rgbImg_0.png
            -rgbImg_1.png
            ...
        -Scene1
            -rgbImg_0.png
            ...
    -Text
        -scene_0.txt
        -scene_1.txt
        ...

The .txt files store information described in the below mentioned format:

https://motchallenge.net/instructions/



To generate data:
1. cd masks-occluder/

2. Run below command

```
python -m masks.data_gen --sim SIM_PATH --scenes SCENES_PATH
```

    ** SIM_PATH is the path of unity_app 
    ** SCENES_PATH is the path of scenes which need to be processed. 


This runs the simulator, translating each scene description `.json` file in `SCENES_PATH` into depth frames, RGB frames and .txt files storing respective object ids x,y, width, height and world coordinate positions. 

3. If .json format data is required of above format, just uncomment the json.dump line at the end of get_data_for_tracking(env, paths, train_dataset_path) function. 