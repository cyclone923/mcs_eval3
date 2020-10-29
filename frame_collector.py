import os
import numpy as np


class Frame_collector:

    def __init__(self, scene_dir, start_scene_number):
        self.scene_dir = scene_dir
        self.scene_number = start_scene_number
        self.step = 0
        os.makedirs(self.scene_dir, exist_ok=True)

    def save_frame(self, step_output):
        assert len(step_output.image_list) <= 1
        assert len(step_output.object_mask_list) <= 1
        assert len(step_output.depth_mask_list) <= 1

        if len(step_output.image_list) == 1:
            step_output.image_list[0].save(f'{self.scene_dir}/original_{self.scene_number}-{self.step}.jpg')
        if len(step_output.object_mask_list) == 1:
            step_output.object_mask_list[0].save(f'{self.scene_dir}/mask_{self.scene_number}-{self.step}.jpg')
        if len(step_output.depth_mask_list) == 1:
            step_output.depth_mask_list[0].save(f'{self.scene_dir}/depth_{self.scene_number}-{self.step}.jpg')
        self.step += 1

        # for i in step_output.object_list:
        #     if i.shape != "shelf":
        #         print(i.shape, i.color)
        # for i in step_output.structural_object_list:
        #     print(i.uuid, i.color) # uuid need to be finely categorized

    def reset(self):
        self.scene_number += 1
        self.step = 0
        print("Reset, Current Scene: {}".format(self.scene_number))