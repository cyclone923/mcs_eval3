from argparse import ArgumentParser
import pickle
from pathlib import Path
import os
from .utils import draw_bounding_boxes, draw_appearance_bars, split_obj_masks, get_obj_position, get_mask_box

from .track import track_objects
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torchvision import transforms
import torch.nn as nn
import cv2
import webcolors
import sys

class AppearanceMatchModel():
    def __init__(self):
        ### ATTRIBUTES LEARNED DURING TRAINING ###

        # The learned amount of leeway in feature match distance
        # to allow without raising an appearance mismatch
        self.feature_match_slack = 0
        self.obj_dictionary = dict()    # The dictionary of learned objects and their keypoints and descriptors
        
        ### ATTRIBUTES USED IN TESTING/EVAL ###

        # The object's features from the previous frame,
        # in case no robust feature match can be detected
        # using the learned object dictionary and the robot
        # must fall back to frame-by-frame matching.
        # self.prev_obj_features = dict()
        self.detector = cv2.SIFT()      # The SIFT feature detection object
        self.bf = cv2.BFMatcher()       # Brute-Force matcher w/ default params

    # Search the learned object space to identify what the initial object is
    def identifyInitialObject(self, img_kp, img_des):
        # NOTE: If object descriptors do not sufficiently match up with any other object, return None
        # (system will fall back to frame-by-frame matching)
        return None

    # Match feature descriptors
    def detectFeatureMatch(self, img_kp, img_des, obj):
        if obj['base_image']['shape_id'] is None:
            match = self.frameMatch(img_kp, img_des, obj)  # fall back to frame-by-frame feature matching
        else:
            # feature match
            avg_match_rate = 1     # TODO: Implement good match rate (see OpenCV Python SIFT docs)
            if avg_match_rate >= 1 - self.feature_match_slack:
                match = True
            else:
                match = self.frameMatch(img_kp, img_des, obj)  # fall back to frame-by-frame feature matching
            
        return obj, match

    # Determine feature match with the state of the object in the previous frame
    def frameMatch(self, img_kp, img_des, obj):
        prev_kp = obj['appearance']['keypoint_history'][-1]
        prev_des = obj['appearance']['descriptor_history'][-1]
        matches = self.bf.knnMatch(img_des, prev_des, k=2) # k=2 so we can apply the ratio test next
        good = list()
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        
        return len(good) / len(matches) >= self.feature_match_slack

    # Check for any appearance mismatches in the provided images
    def match(self, image, objects_info, device='cpu', level='level2'):
        for key, obj in objects_info.items():
            if not obj['visible']:
                continue
            top_x, top_y, bottom_x, bottom_y = obj['bounding_box']
            obj_current_image = image.crop((top_y, top_x, bottom_y, bottom_x))

            image_area = np.prod(obj_current_image.size)
            base_image = np.array(image)
            mask_image = np.zeroes(obj['mask'].shape, dtype=base_image.dtype)
            mask_image[obj['mask']] = 255

            # obj_clr_hist_0 = cv2.calcHist([np.array(image)], [0], mask_image, [10], [0, 256])
            # obj_clr_hist_1 = cv2.calcHist([np.array(image)], [1], mask_image, [10], [0, 256])
            # obj_clr_hist_2 = cv2.calcHist([np.array(image)], [2], mask_image, [10], [0, 256])
            # obj_clr_hist = (obj_clr_hist_0 + obj_clr_hist_1 + obj_clr_hist_2) / 3

            # run SIFT on image
            img_kp, img_des = self.detector.detectAndCompute(obj_current_image, None)

            if 'base_image' not in obj.keys() or (len(obj['position_history'] < 5) and obj['base_image']['image_area'] < image_area):
                obj['base_image'] = dict()
                obj['base_image']['shape_id'] = self.identifyInitialObject(img_kp, img_des)
                obj['appearance']['feature_history'] = dict()
                obj['appearance']['feature_history']['keypoints'] = list()
                obj['appearance']['feature_history']['descriptors'] = list()
                # obj['base_image']['histogram'] = obj_clr_hist
                obj['appearance'] = dict()

            # TODO: run detectFeatureMatch
            obj, feature_match = self.detectFeatureMatch(img_kp, img_des, obj)

            # TODO: Tweak occlusion code s.t. occlusion check happens before
            # appearance model and occlusion status lives in object_info
            if not obj['occluded']:
                obj['appearance']['feature_history']['keypoints'].append(img_kp)
                obj['appearance']['feature_history']['descriptors'].append(img_des)
                obj['appearance']['match'] = feature_match

            if 'mismatch_count' not in obj['appearance']:
                obj['appearance']['mismatch_count'] = 0

            if obj['appearance']['match']:
                obj['appearance']['mismatch_count'] = 0
            else:
                obj['appearance']['mismatch_count'] += 1

        return objects_info


    def train(self):
        pass

    def test(self):
        pass

# Convert the image to a Tensor
def obj_image_to_tensor(obj_image, gray=False):
    obj_image = obj_image.resize((50, 50))
    obj_image = np.array(obj_image)
    obj_image = obj_image.reshape((3, 50, 50))
    obj_image = torch.Tensor(obj_image).float()
    if gray:
        obj_image = rgb_to_grayscale(obj_image)
    
    return obj_image

# Convert a color image to grayscale
def rgb_to_grayscale(img, num_output_channels: int = 1):
    if num_output_channels not in (1, 3):
        raise ValueError('num_output_channels should be either 1 or 3')

    r, g, b = img.unbind(dim=-3)
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-3)

    if num_output_channels == 3:
        return l_img.expand(img.shape)

    return l_img

def make_parser():
    parser = ArgumentParser()
    return parser
