import pickle
import gzip
from pathlib import Path
from argparse import ArgumentParser
import os
from .utils import draw_bounding_boxes, draw_appearance_bars, split_obj_masks, get_obj_position, get_mask_box

from .track import track_objects
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
import torch.nn as nn
import cv2
import webcolors


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def getNearestWebSafeColor(x):
    r, g, b = x
    r = int(round((r / 255.0) * 5) * 51)
    g = int(round((g / 255.0) * 5) * 51)
    b = int(round((b / 255.0) * 5) * 51)
    return (r, g, b)


def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


def obj_image_to_tensor(obj_image):
    obj_image = obj_image.resize((50, 50))
    obj_image = np.array(obj_image)
    obj_image = obj_image.reshape((3, 50, 50))
    obj_image = torch.Tensor(obj_image).float()
    return obj_image


class AppearanceMatchModel(nn.Module):
    def __init__(self):
        super(AppearanceMatchModel, self).__init__()

        self._shape_labels = ['car', 'cone', 'cube', 'cylinder', 'duck', 'sphere', 'square frustum', 'turtle']
        self._color_labels = ['black', 'blue', 'brown', 'green', 'grey', 'red', 'yellow']

        self.feature = nn.Sequential(nn.Conv2d(3, 6, 2),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(6, 6, 3),
                                     nn.LeakyReLU())

        self.color_feature = nn.Sequential(nn.Conv2d(3, 6, 2),
                                           nn.LeakyReLU(),
                                           nn.Conv2d(6, 6, 3),
                                           nn.LeakyReLU())

        self.shape_classifier = nn.Sequential(nn.Linear(13254, len(self._shape_labels)))
        self.color_classifier = nn.Sequential(nn.Linear(13254, len(self._color_labels)))

    def shape_label(self, id):
        return self._shape_labels[id]

    def shape_labels(self):
        return self._shape_labels

    def color_label(self, id):
        return self._color_labels[id]

    def color_labels(self):
        return self._color_labels

    def forward(self, x):
        feature = self.feature(x)
        color_feature = self.color_feature(x)
        feature = feature.flatten(1)
        color_feature = color_feature.flatten(1)
        return feature, self.shape_classifier(feature), self.color_classifier(color_feature)


def object_appearance_match(model, image, objects_info, device='cpu'):
    for obj_key in objects_info:

        top_x, top_y, bottom_x, bottom_y = objects_info[obj_key]['bounding_box']
        obj_current_image = image.crop((top_y, top_x, bottom_y, bottom_x))

        with torch.no_grad():
            obj_current_image_tensor = obj_image_to_tensor(obj_current_image).to(device)
            obj_current_image_tensor = obj_current_image_tensor.unsqueeze(0)

            # extract appearance ( shape) info
            _, object_shape_logit, object_color_logit = model(obj_current_image_tensor)
            object_shape_logit = object_shape_logit.squeeze(0)
            object_shape_prob = torch.softmax(object_shape_logit, dim=0)

            object_color_logit = object_color_logit.squeeze(0)
            object_color_prob = torch.softmax(object_color_logit, dim=0)

        current_object_shape_id = torch.argmax(object_shape_prob).item()
        current_object_color_id = torch.argmax(object_color_prob).item()

        base_image = np.array(image)
        mask_image = np.zeros(objects_info[obj_key]['mask'].shape, dtype=base_image.dtype)
        mask_image[objects_info[obj_key]['mask']] = 255

        obj_clr_hist_0 = cv2.calcHist([np.array(image)], [0], mask_image, [10], [0, 256])
        obj_clr_hist_1 = cv2.calcHist([np.array(image)], [1], mask_image, [10], [0, 256])
        obj_clr_hist_2 = cv2.calcHist([np.array(image)], [2], mask_image, [10], [0, 256])
        obj_clr_hist = (obj_clr_hist_0 + obj_clr_hist_1 + obj_clr_hist_2) / 3

        if 'base_image' not in objects_info[obj_key] or len(objects_info[obj_key]['position_history']) < 5:
            objects_info[obj_key]['base_image'] = {}
            objects_info[obj_key]['appearance'] = {}
            objects_info[obj_key]['base_image']['shape_id'] = current_object_shape_id
            objects_info[obj_key]['base_image']['shape'] = model.shape_label(current_object_shape_id)
            objects_info[obj_key]['base_image']['color_id'] = current_object_color_id
            objects_info[obj_key]['base_image']['color'] = model.color_label(current_object_color_id)

            objects_info[obj_key]['base_image']['histogram'] = obj_clr_hist
            base_shape_id = current_object_shape_id
            base_color_id = current_object_color_id
        else:
            base_shape_id = objects_info[obj_key]['base_image']['shape_id']
            base_color_id = objects_info[obj_key]['base_image']['color_id']

        # shape match
        objects_info[obj_key]['appearance']['shape_match_quotient'] = object_shape_prob[base_shape_id].item()
        objects_info[obj_key]['appearance']['shape_prob'] = object_shape_prob.cpu().numpy()
        objects_info[obj_key]['appearance']['shape_prob_labels'] = model.shape_labels()

        # color match
        objects_info[obj_key]['appearance']['color_hist'] = obj_clr_hist
        objects_info[obj_key]['appearance']['color_match_quotient'] = object_color_prob[base_color_id].item()
        objects_info[obj_key]['appearance']['color_prob'] = object_color_prob.cpu().numpy()
        objects_info[obj_key]['appearance']['color_prob_labels'] = model.color_labels()

        # objects_info[obj_key]['appearance']['color'] = current_object_color_id
        # objects_info[obj_key]['appearance']['color_id'] = current_object_color_id
        # objects_info[obj_key]['appearance']['dominant_color_name'] = dominant_color_name
        # objects_info[obj_key]['appearance']['dominant_color_rgb'] = dominant_color_rgb

        objects_info[obj_key]['appearance']['color_hist_quotient'] = cv2.compareHist(obj_clr_hist,
                                                                                     objects_info[obj_key][
                                                                                         'base_image']['histogram'],
                                                                                     cv2.HISTCMP_CORREL)

        # Todo: size match?

        # match
        shape_matches = current_object_shape_id == base_shape_id
        color_matches = current_object_color_id == base_color_id
        objects_info[obj_key]['appearance']['match'] = shape_matches and color_matches

    return objects_info
