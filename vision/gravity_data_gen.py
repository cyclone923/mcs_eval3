import cv2
import numpy as np
import machine_common_sense as mcs
from dataclasses import dataclass
import json
import os

@dataclass
class ImageDataWriter:

    step_number: int
    step_meta: mcs.StepMetadata
    scene_id: str
    support_id: str
    target_id: str
    pole_id: str
    
    def __post_init__(self):
        self.get_images_from_meta()
        self.step_output = dict(self.step_meta)
        self.write_to_json()

    def get_images_from_meta(self):

        self.rgb_im = self.step_meta.image_list[0]
        self.rgb_im = np.array(self.rgb_im)[:,:,::-1] # BGR -> RGB

        self.obj_mask = self.step_meta.object_mask_list[0]
        self.obj_mask = np.array(self.obj_mask)[:,:,::-1]

        self.depth_map = self.step_meta.depth_map_list[0]
        self.depth_map *= 255. / self.depth_map.max()
        
        if not os.path.exists("gravity/Data/"):
            os.mkdir("gravity/Data/")
        
        if not os.path.exists("gravity/Data/{}/".format(self.scene_id)):
            os.mkdir("gravity/Data/{}/".format(self.scene_id))

        if not os.path.exists("gravity/Data/{}/{}/".format(self.scene_id, self.step_number)):
            os.mkdir("gravity/Data/{}/{}/".format(self.scene_id, self.step_number))

        self.rgb_path = "gravity/Data/{}/{}/rgb.jpg".format(self.scene_id, self.step_number)
        self.mask_path = "gravity/Data/{}/{}/mask.jpg".format(self.scene_id, self.step_number)
        self.depth_path = "gravity/Data/{}/{}/depth.jpg".format(self.scene_id, self.step_number)

        # write images to local dir 
        # cv2.imshow("rgb", self.rgb_im)
        cv2.imwrite(self.rgb_path, self.rgb_im)
        
        # cv2.imshow("mask",self.obj_mask)
        cv2.imwrite(self.mask_path, self.obj_mask)
        
        # cv2.imshow("depth", self.depth_map)
        cv2.imwrite(self.depth_path, self.depth_map)

    def write_to_json(self):
        new_entry = {}

        # image paths
        new_entry["rgb_path"] = self.rgb_path
        new_entry["mask_path"] = self.mask_path
        new_entry["depth_path"] = self.depth_path

        # pole and support kind are hard coded for eval 3.5
        # get pole data
        if self.pole_id in self.step_output["structural_object_list"]:
            new_entry["pole_kind"] = 'cylinder'
            new_entry["pole_bboxes"] = [list(x.values()) for x in self.step_output["structural_object_list"][self.pole_id]['dimensions']]
            new_entry["pole_pos"] = list(self.step_output["structural_object_list"][self.pole_id]['position'].values())

        # get target data
        if self.target_id in self.step_output["object_list"]:
            new_entry["target_kind"] = self.step_output["object_list"][self.target_id]['shape']
            new_entry["target_bboxes"] = [list(x.values()) for x in self.step_output["object_list"][self.target_id]['dimensions']]
            new_entry["target_pos"] = list(self.step_output["object_list"][self.target_id]['position'].values())
        
        # get support data
        if self.support_id in self.step_output["structural_object_list"]:
            new_entry["support_kind"] = 'cube'
            new_entry["support_bboxes"] = [list(x.values()) for x in self.step_output["structural_object_list"][self.support_id]['dimensions']]
            new_entry["support_pos"] = list(self.step_output["structural_object_list"][self.support_id]['position'].values())

        # append the new entry into the json database
        json_data = ''
        if os.path.exists("gravity/Data/scene_data.json"):
            with open("gravity/Data/scene_data.json", "r") as json_file:
                json_data = json.load(json_file)
        else:
            json_data = {
                "entries": []
            }

        json_data["entries"].append(new_entry)

        with open("gravity/Data/scene_data.json", "w") as json_file:
            json.dump(json_data, json_file, indent=4)