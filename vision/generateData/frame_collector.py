import os
import numpy as np
from vision.generateData.instSeg_parse_mask import parse_label_info, save_depth_image

class Frame_collector:

    def __init__(self, scene_dir, start_scene_number):
        self.scene_number = start_scene_number
        self.step = 0
        self.scene_dir = scene_dir
        self.result_dir = os.path.join(self.scene_dir, 'scene_'+str(self.scene_number))
        os.makedirs(self.result_dir, exist_ok=True)

        # self.shape_keys  = ['changing table', 'duck', 'drawer', 'box', 'bowl', \
        #                     'sofa chair', 'pacifier', 'number block cube', 'crayon', 'ball', \
        #                     'blank block cube', 'chair', 'plate', 'sofa', 'stool', \
        #                     'racecar', 'blank block cylinder', 'cup', 'apple', 'table']
        self.shape_keys  = ['trophy', 'box']
        self.uuid_keys = ['floor', 'ceiling', 'occluder_pole', 'occluder_wall', 'wall']

        self.shap_new_keys   = []
        self.stru_new_keys   = []

        self.count_hist = {'all': 0}
        for key in self.shape_keys:
            self.count_hist[key] = 0

    def save_frame(self, step_output, saveImage=True):
        # print("Save Image!")
        if saveImage:
            for j in range(len(step_output.image_list)):
                step_output.image_list[j].save(f'{self.result_dir}/original-{self.step}-{j}.jpg')
                maskI = np.asarray(step_output.object_mask_list[j]) # [ht, wd, 3] in RGB
                parse_label_info(maskI,
                                 step_output.structural_object_list,
                                 step_output.object_list,
                                 result_dir=self.result_dir, sname=f'-{self.step}-{j}')

                save_depth_image(np.asarray(step_output.depth_mask_list[j]),
                                            result_dir = self.result_dir, sname=f'-{self.step}-{j}')
            self.step += 1
        else:
            pass
        # for checking the FG objects and BG objects
        for i in step_output.object_list:
            if i.shape not in (self.shape_keys+self.shap_new_keys):
                self.shap_new_keys.append(i.shape)
                self.count_hist[i.shape] = 0
            self.count_hist[i.shape] += 1
            self.count_hist['all'] += 1

        for i in step_output.structural_object_list:
            if not any(s in i.uuid for s in (self.uuid_keys+self.stru_new_keys)):
                self.stru_new_keys.append(i.uuid)
                #print(i.uuid, i.color) # uuid need to be finely categorized

    def reset(self):
        self.scene_number += 1
        self.step = 0
        self.result_dir = os.path.join(self.scene_dir, 'scene_'+str(self.scene_number))
        os.makedirs(self.result_dir, exist_ok=True)
        print("Reset, Current Scene: {}".format(self.scene_number))
        print("un-set uuid key including: ", self.stru_new_keys)
        print("un-set shape key including: ", self.shap_new_keys)
        print("statistical shape key counts: ", self.count_hist)
        print("--------------------------------------------------------")
