from physicsvoe.data.data_gen import convert_output
from physicsvoe import framewisevoe, occlude
from physicsvoe.timer import Timer
from physicsvoe.data.types import make_camera
import visionmodule.inference as vision
from vision.generateData.frame_collector import Frame_collector
from pathlib import Path
from PIL import Image
import numpy as np
import sys
DEBUG = True
start_scene_number = 0

class VoeAgent:
    def __init__(self, controller, level, out_prefix=None):
        self.controller =  controller
        self.level = level
        if DEBUG:
            self.prefix = out_prefix
        start_scene_number = 0
        self.collector = Frame_collector(scene_dir="./out_1/", start_scene_number=start_scene_number, fg_class_en=False)

    def run_scene(self, config, desc_name):

        self.controller.start_scene(config)
        
        for i, x in enumerate(config['goal']['action_list']):
            step_output = self.controller.step(action=x[0])
            # self.collector.save_frame(step_output)
            self.collector.save_frame(step_output)
            # import pdb
            # pdb.set_trace()
        sys.stdout.flush()
        self.collector.reset()