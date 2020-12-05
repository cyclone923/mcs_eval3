import gzip; import pickle

import pickle
import gzip
import numpy as np
from PIL import Image
import os
import cv2
import sys
train_scenes_path = '../../chengxi_scenes/tmp/'
train_scenes = os.listdir(train_scenes_path)


def get_mask_box(obj_mask):
    height, width = obj_mask.shape
    rows, cols = np.where(obj_mask == True)
    box_top_x, box_top_y = max(0, rows.min() - 1), max(0, cols.min() - 1)
    box_bottom_x, box_bottom_y = min(rows.max() + 1, height - 1), min(cols.max() + 1, width - 1)
    return (box_top_x, box_top_y), (box_bottom_x, box_bottom_y)

def draw_bounding_boxes(base_image, frame_objects_info):
    box_img = np.array(base_image)
    (box_top_x, box_top_y), (box_bottom_x, box_bottom_y) = frame_objects_info
    box_img = cv2.rectangle(box_img, (box_top_y, box_top_x), (box_bottom_y, box_bottom_x), (255, 255, 0), 2)
    box_img = cv2.putText(box_img, str(1), (box_top_y, box_top_x), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    return Image.fromarray(box_img)

def filter_objects(scene_frame, depth_frame, masks):
    results = {'objects':[], 'occluders':[]}
    for idx, mask in enumerate(masks):
        (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) = get_mask_box(mask)
        obj_image = scene_frame.crop((top_left_y, top_left_x, bottom_right_y, bottom_right_x))
        depth_crop = depth_frame[ top_left_x+1: bottom_right_x, top_left_y+1:bottom_right_y]
        name = size_filter(obj_image, depth_crop)
        if name is 'object':
            results['objects'].append(mask)
        else:
            results['occluders'].append(mask)
    return results


def generate_mask_data(scenes_files):
    data = []
    label = []
    counter = 0
    actual_counter = 0
    for scene_file in sorted(scenes_files):
        print (train_scenes_path + scene_file)
        with gzip.open(train_scenes_path + scene_file, 'rb') as fd:
            scene_data = pickle.load(fd)
        for frame_num, frame in enumerate(scene_data):
            img = frame.image
            objs = frame.obj_data
            structs = frame.struct_obj_data
            depth = frame.depth_mask
            obj_masks = split_obj_masks(frame.obj_mask, len(objs))
            struct_masks = split_obj_masks(frame.struct_mask, len(structs))
            for obj_i, (obj, obj_mask) in enumerate(zip(objs, obj_masks)):
                if True not in obj_mask:
                    objs.remove(obj)
                    # del objs[obj_i]
                else:
                    actual_counter +=1
                    (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) = get_mask_box(obj_mask)
                    obj_image = frame.image.crop((top_left_y, top_left_x, bottom_right_y, bottom_right_x))
                    depth_crop = depth[ top_left_x+1: bottom_right_x, top_left_y+1:bottom_right_y]
                    name = size_filter(obj_image, depth_crop)
                    info =  (top_left_x, top_left_y), (bottom_right_x, bottom_right_y)
                    bounded_img = draw_bounding_boxes(frame.image, info )
                    if name is 'object':
                        counter +=1
                    obj_image = obj_image.resize((50,   50))
                    data.append(np.array(obj_image))
                    label.append(1)
            for struct_i, (struct, struct_mask) in enumerate(zip(structs, struct_masks)):
                if True not in struct_mask:
                    structs.remove(struct)
                    # del structs[struct_i]
                else:
                    struct_img = mask_img(struct_mask, img)

                    (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) = get_mask_box(struct_mask)
                    struct_image = struct_img.crop((top_left_y, top_left_x, bottom_right_y, bottom_right_x))
                    depth_crop = depth[ top_left_x+1: bottom_right_x, top_left_y+1:bottom_right_y]

                    name = size_filter(struct_image, depth_crop)
                    actual_counter  +=1
                    counter +=1
                    if name=='object':
                        counter -=1
                        struct_image.save(f'{frame_num:02d}_{struct_i}_{counter}_{name}.png')
                    struct_image = struct_image.resize((50, 50))


        print (counter/actual_counter,  counter, actual_counter)


def size_filter(img, depth_crop):
    size = img.size
    w,h  = size[0], size[1]
    if w/h>=3.0 and w<400: # and np.std(depth_crop)<0.03:
        name = 'pole'
    elif h/w>2.0 and h<100:
        name = 'pole'
    elif w>400:
        name = 'wall_floor'
    elif h/w>1.4 and h>120:
        name = 'occluder'
    elif h>100 or w>100:
        name = 'occluder'
    else:
        name = 'object'
    return name


# def size_filter(img, depth_crop):
#     size = img.size
#     w,h  = size[0], size[1]
#     if w/h>=3 and w<400: # and np.std(depth_crop)<0.03:
#         name = 'pole'
#     elif h/w>2.0 and h<100:
#         name = 'pole'
#     elif w>400:
#         name = 'wall_floor'
#     elif h/w>1.4 and h>120:
#         name = 'occluder'
#     elif h>100 or w>100:
#         name = 'occluder'
#     else:
#         name = 'object'
#     return name


# def demo_voe_segmentation():
#     import glob
#     import cv2
#     import scipy.misc as smisc  # scipy in version <= 1.2.0

#     model = MaskAndClassPredictor(dataset='mcsvideo3_voe',
#                                   config='plus_resnet50_config_depth_MC',
#                                   weights='./vision/instSeg/dvis_resnet50_mc_voe.pth')

#     img_list = glob.glob('./vision/instSeg/demo/voe/*.jpg')
#     for rgb_file in img_list:

#         bgrI   = cv2.imread(rgb_file)
#         depthI = smisc.imread(depth_file, mode='P')
#         ret    = model.step(bgrI, depthI)



def split_obj_masks(mask, num_objs):
    obj_masks = []
    for obj_idx in range(num_objs):
        obj_mask = (mask == obj_idx)
        obj_masks.append(obj_mask)
    return obj_masks


def mask_img(mask, img):
    img_arr = np.asarray(img)
    masked_arr = img_arr * mask[:, :, np.newaxis]
    return Image.fromarray(masked_arr)


# generate_mask_data(train_scenes)
