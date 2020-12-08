import gzip; import pickle

import pickle
import gzip
import numpy as np
from PIL import Image
import os
import cv2
import sys

import glob
from skimage import measure as smeasure


train_scenes_path = '/home/jay/chengxi_scenes/tmp/'
train_scenes = os.listdir(train_scenes_path)

from vision.instSeg.inference import MaskAndClassPredictor


model = MaskAndClassPredictor(dataset='mcsvideo3_voe',
                              config='plus_resnet50_config_depth_MC',
                              weights='/home/jay/mcs_eval3/vision/instSeg/dvis_resnet50_mc_voe.pth')



def get_mask_box(obj_mask):
    height, width = obj_mask.shape
    rows, cols = np.where(obj_mask == True)
    box_top_x, box_top_y = max(0, rows.min() - 1), max(0, cols.min() - 1)
    box_bottom_x, box_bottom_y = min(rows.max() + 1, height - 1), min(cols.max() + 1, width - 1)
    return (box_top_x, box_top_y), (box_bottom_x, box_bottom_y)
    
def draw_bounding_boxes(base_image, frame_objects_info):
    box_img = np.array(base_image)
    (box_top_x, box_top_y), (box_bottom_x, box_bottom_y) = frame_objects_info
    box_img = cv2.rectangle(box_img, (box_top_x, box_top_y), (box_bottom_x, box_bottom_y), (255, 255, 0), 2)
    # box_img = cv2.rectangle(box_img, (box_top_y, box_top_x), (box_bottom_y, box_bottom_x), (255, 255, 0), 2)
    # box_img = cv2.putText(box_img, str(1), (box_top_y, box_top_x), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
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

def filter_objects_model(scene_frame, depth_frame, masks=None):
    ret = model.step(np.array(img), depth)

    results = {'objects':[], 'occluders':[]}
    # if not isinstance(scene_frame, np.ndarray):
    #     scene_frame = np.array(scene_frame)
    if not isinstance(depth_frame, np.ndarray):
        depth_frame = np.array(depth_frame)

    ret = model.step(scene_frame, depth_frame)
    labelI = ret['mask_prob'].argmax(axis=0)
    for i in range(1, labelI.max() +1):
        conn_labelI = smeasure.label(labelI==i)
        props = smeasure.regionprops(conn_labelI)
        for idx, prop in enumerate(props):
            y0,x0,y1,x1 = prop.bbox
            obj_image = scene_frame.crop(( x0, y0,  x1,  y1))
            mask = cv2.rectangle(np.array(img), (x0,y0), (x1,y1), (0, 0, 0), -1)
            mask = mask[:,:,0]==0
            name = size_filter(obj_image, None)
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
    print (w,h)
    if w*h<80: # and np.std(depth_crop)<0.03:
        name = 'ignore' 
        print (name, w,h)
    elif w/h>=3.0 and w<400: # and np.std(depth_crop)<0.03:
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


def model_test(scenes_files):
    import glob
    import cv2
    import scipy.misc as smisc  # scipy in version <= 1.2.0

    model = MaskAndClassPredictor(dataset='mcsvideo3_voe',
                                  config='plus_resnet50_config_depth_MC',
                                  weights='/home/jay/mcs_eval3/vision/instSeg/dvis_resnet50_mc_voe.pth')

    import random
    random.shuffle(scenes_files)
    # for scene_file in sorted(scenes_files):
    for scene_file in scenes_files:

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


            ret = model.step(np.array(img), depth)
            masks = ret['mask_prob'][1:] > 0.5
            results = {'objects':[], 'occluders':[]}

            # # ret = model.step(scene_frame, depth_frame)
            # labelI = ret['mask_prob'][1:] > 0.5
            
            labelI = ret['mask_prob'].argmax(axis=0)

            from matplotlib import pyplot as plt
            # img.save( f'scene.png')
            # plt.imshow(depth, cmap='gray')
            # plt.savefig('depth.png')

            # cv2.imwrite('scene.jpg', np.array(img)[:,:,::-1])
            # cv2.imwrite('depth.jpg', depth)

            # plt.imshow(labelI)
            # plt.savefig('output.png')

            # Image.fromarray(np.uint8(labelI)).save( f'op_ascene.png')  
            for i in range(1, labelI.max() +1):
                conn_labelI = smeasure.label(labelI==i)
                props = smeasure.regionprops(conn_labelI)
                # y0,x0,y1,x1 = prop.bbox
                # print (i, len(props))
                for idx, prop in enumerate(props):
                    y0,x0,y1,x1 = prop.bbox
                    info = (x0,y0), (x1,y1)
                    # print (x0, y0, x1, y1)

                    obj_image = img.crop(( x0, y0,  x1,  y1))
                    mask = cv2.rectangle(np.array(img), (x0,y0), (x1,y1), (0, 0, 0), -1)
                    mask = mask[:,:,0]==0
                    # cv2.imwrite(f'{i:02d}_{idx:02d}_masked_output.png', np.array(img) * mask[:, :, np.newaxis]) 
                    # Image.fromarray(mask).save( f'{i:02d}_{idx:02d}_masks.png')

                    bounded_img = draw_bounding_boxes(img,info)

                    # print (obj_image.size)
                    # import pdb; pdb.set_trace()
                    # bounded_img.save( f'{i:02d}_{idx:02d}_bounded_scene.png')
                    name = size_filter(obj_image, None)
                    if name is 'object':
                        results['objects'].append(mask)
                        # obj_image.save( f'run2/{i:02d}_{idx:02d}_cropped_scene.png')
                        # cv2.imwrite(f'run2/{frame_num:02d}.png', np.array(img) * mask[:, :, np.newaxis]) 

                        obj_image.save( f'{i:02d}_{idx:02d}_cropped_scene.png')
                        cv2.imwrite(f'{frame_num:02d}.png', np.array(img) * mask[:, :, np.newaxis])                         
                    else:
                        results['occluders'].append(mask)

        #         # depth_crop = depth[ top_left_x+1: bottom_right_x, top_left_y+1:bottom_right_y]
        #             name = size_filter(obj_image, None)
        #             print (name)
        #             if name is 'object':
        #                 results['objects'].append(mask)
        #                 cv2.imwrite(f'{frame_num:02d}.png', np.array(img) * mask[:, :, np.newaxis]) 
        #             else:
        #                 results['occluders'].append(mask)

        #     for idx, mask in enumerate(masks):
        #         (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) = get_mask_box(mask)
        #         obj_image = img.crop((top_left_y, top_left_x, bottom_right_y, bottom_right_x))
        #         depth_crop = depth[ top_left_x+1: bottom_right_x, top_left_y+1:bottom_right_y]
        #         name = size_filter(obj_image, depth_crop)
        #         if name is 'object':
        #             results['objects'].append(mask)
        #             cv2.imwrite(f'{frame_num:02d}.png', np.array(img) * mask[:, :, np.newaxis]) 
        #         else:
        #             results['occluders'].append(mask)
        #     break
        # break

# def model_test(scenes_files):
#     import glob
#     import cv2
#     import scipy.misc as smisc  # scipy in version <= 1.2.0

#     model = MaskAndClassPredictor(dataset='mcsvideo3_voe',
#                                   config='plus_resnet50_config_depth_MC',
#                                   weights='/home/jay/mcs_eval3/vision/instSeg/dvis_resnet50_mc_voe.pth')


#     for scene_file in sorted(scenes_files):
#         print (train_scenes_path + scene_file)

#         with gzip.open(train_scenes_path + scene_file, 'rb') as fd:
#             scene_data = pickle.load(fd)
#         for frame_num, frame in enumerate(scene_data):
#             img = frame.image
#             objs = frame.obj_data
#             structs = frame.struct_obj_data
#             depth = frame.depth_mask
#             obj_masks = split_obj_masks(frame.obj_mask, len(objs))
#             struct_masks = split_obj_masks(frame.struct_mask, len(structs))


#             ret = model.step(np.array(img), depth)
#             masks = ret['mask_prob'][1:] > 0.5
#             results = {'objects':[], 'occluders':[]}

#             # # ret = model.step(scene_frame, depth_frame)
#             # labelI = ret['mask_prob'][1:] > 0.5
            
#             labelI = ret['mask_prob'].argmax(axis=0)
#             print (labelI.shape)

#             from matplotlib import pyplot as plt
#             # img.save( f'scene.png')
#             # plt.imshow(depth, cmap='gray')
#             # plt.savefig('depth.png')

#             # cv2.imwrite('scene.jpg', np.array(img)[:,:,::-1])
#             # cv2.imwrite('depth.jpg', depth)

#             plt.imshow(labelI)
#             plt.savefig('output.png')

#             # Image.fromarray(np.uint8(labelI)).save( f'op_ascene.png')  
#             for i in range(1, labelI.max() +1):
#                 conn_labelI = smeasure.label(labelI==i)
#                 props = smeasure.regionprops(conn_labelI)
#                 # y0,x0,y1,x1 = prop.bbox
#                 print (i, len(props))
#                 for idx, prop in enumerate(props):
#                     y0,x0,y1,x1 = prop.bbox
#                     info = (x0,y0), (x1,y1)
#                     print (x0, y0, x1, y1)

#                     # (box_top_x, box_top_y), (box_bottom_x, box_bottom_y) = frame_objects_info
#                     # box_img = cv2.rectangle(box_img, (box_top_x, box_top_y), (box_bottom_x, box_bottom_y), (255, 255, 0), 2)
#                     # box_img = cv2.rectangle(box_img, (box_top_y, box_top_x), (box_bottom_y, box_bottom_x), (255, 255, 0), 2)

#                     obj_image = img.crop(( x0, y0,  x1,  y1))
                    
#                     mask_img = cv2.rectangle(np.array(img), (x0,y0), (x1,y1), (0, 0, 0), -1)
#                     mask_img = mask_img[:,:,0]==0
#                     # print (mask_img)
#                     # cv2.imwrite(f'{i:02d}_{idx:02d}_masked_output.png', np.array(img) * mask_img[:, :, np.newaxis]) 

#                     Image.fromarray(mask_img).save( f'{i:02d}_{idx:02d}_masks.png')
#                     obj_image.save( f'{i:02d}_{idx:02d}_crops.png')

#                     bounded_img = draw_bounding_boxes(img,info)

#                     # print (obj_image.size)
#                     # import pdb; pdb.set_trace()
#                     bounded_img.save( f'{i:02d}_{idx:02d}_scene.png')
#                     obj_image.save( f'{i:02d}_{idx:02d}_crops.png')

#                 img.save( f'scene.png')

#                 # # depth_crop = depth[ top_left_x+1: bottom_right_x, top_left_y+1:bottom_right_y]
#                 #     name = size_filter(obj_image, None)
#                 #     print (name)
#                 #     if name is 'object':
#                 #         results['objects'].append(mask)
#                 #         cv2.imwrite(f'{frame_num:02d}.png', np.array(img) * mask[:, :, np.newaxis]) 
#                 #     else:
#                 #         results['occluders'].append(mask)

#             # for idx, mask in enumerate(masks):
#             #     (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) = get_mask_box(mask)
#             #     obj_image = img.crop((top_left_y, top_left_x, bottom_right_y, bottom_right_x))
#             #     depth_crop = depth[ top_left_x+1: bottom_right_x, top_left_y+1:bottom_right_y]
#             #     name = size_filter(obj_image, depth_crop)
#             #     if name is 'object':
#             #         results['objects'].append(mask)
#             #         cv2.imwrite(f'{frame_num:02d}.png', np.array(img) * mask[:, :, np.newaxis]) 
#             #     else:
#             #         results['occluders'].append(mask)
#             # break
#         break
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

model_test(train_scenes)
# generate_mask_data(train_scenes)
