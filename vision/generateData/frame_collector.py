import os
import numpy as np
from vision.generateData.instSeg_parse_mask import parse_label_info, save_depth_image
from vision.generateData.instSeg_parse_mask import setup_configuration

class Frame_collector:

    def __init__(self, scene_dir, start_scene_number,
                       scene_type='interact', fg_class_en=False):
        """
        @Param: scene_dir -- directory to saveout data
                start_scene_number -- INT
                task -- 'voe' | 'interact' | 'combine'
                fg_class_en -- if True, specify the shape of FG objects in detail.
        """
        self.scene_number = start_scene_number
        self.step = 0
        self.scene_dir = scene_dir
        self.result_dir = os.path.join(self.scene_dir, 'scene_'+str(self.scene_number))
        os.makedirs(self.result_dir, exist_ok=True)
        self.parse_cfg = setup_configuration(task=scene_type, fg_class_en=fg_class_en)

        self.shape_keys  = self.parse_cfg.object_classes
        self.uuid_keys = self.parse_cfg.bg_classes

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
                parse_label_info(self.parse_cfg,
                                 maskI,
                                 step_output.structural_object_list,
                                 step_output.object_list,
                                 result_dir=self.result_dir, sname=f'-{self.step}-{j}')

                save_depth_image(np.asarray(step_output.depth_map_list[j]),
                                            result_dir = self.result_dir, sname=f'-{self.step}-{j}')
                print(self.result_dir, np.sum(np.asarray(step_output.depth_map_list[j])))
            self.step += 1
        else:
            print("hmm...")
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
        print("AAAAAAAAAAAAAAAAAAHHHHHHHHHHHHHHHHHHHHHH!!!!!!!!!!!!!!! (: ")
        self.scene_number += 1
        self.step = 0
        self.result_dir = os.path.join(self.scene_dir, 'scene_'+str(self.scene_number))
        os.makedirs(self.result_dir, exist_ok=True)
        print("self.result_dir: ", self.result_dir)
        print("Reset, Current Scene: {}".format(self.scene_number))
        #print("un-set uuid key including: ", self.stru_new_keys)
        #print("un-set shape key including: ", self.shap_new_keys)
        print("statistical shape key counts: ", self.count_hist)
        print("--------------------------------------------------------")
