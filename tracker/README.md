1. Install simulator

As described in the parent repository. The THOR executable and data folder should be placed in some folder `SIM_PATH`.

2. Download scene descriptions

Download & unzip `https://evaluation-training-scenes.s3.amazonaws.com/eval3/training-passive-physics.zip`.

3. Generate data

```
python -m masks.data_gen --sim SIM_PATH --scenes SCENES_PATH
```

This runs the simulator, translating each scene description `.json` file in `SCENES_PATH` into a scene history file.

The scenes are processed in a random order, so multiple instances of the above command can be run in parallel to speed up the processing.

4. Use data

```
python track.py --scenes SCENES_PATH
```


`masks.py` shows how the resulting data can be used to isolate objects in each scene. 

It just reads any scene histories present in `SCENES_PATH` and outputs segmented object images for each frame.
Currently it drops to a debug prompt after writing the first set of images. Type `c` to continue to the next batch of output if you want to see more.
