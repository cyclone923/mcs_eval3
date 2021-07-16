from .mcs_env import McsEnv
from .types import ThorFrame, CameraInfo
import os
import json
import cv2
import pickle
import random
import numpy as np
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser
#from vision.generateData.instSeg_parse_mask import parse_label_info, save_depth_image
from .utils import draw_bounding_boxes, draw_appearance_bars, split_obj_masks, get_obj_position, get_mask_box
from .save_tool import SaveTool
from numpyencoder import NumpyEncoder
from . import depthutils as du
import torch
from .get_camera_info import make_camera





from collections import namedtuple


#Eval 4 training data for siamese based appearance only-> 
# Step 1: Process json scenes into mask-object RGBD scenes (done by eval4_data_gen.py)
# Step 2: Fetch the cropped objects and annotate the images as per classes of objects (siameseAppearence.py)


from matplotlib import pyplot as plt


def calc_world_pos(depth, mask, camera):
    mask = torch.tensor(mask)
    depth = torch.tensor(depth)
    obj_masks, all_ids = du.separate_obj_masks(mask)
    obj_pos, obj_present = du.project_points_frame(depth, obj_masks, camera)
    return obj_pos, obj_present



def get_data_for_tracking(env, paths, train_dataset_path):
    idx = 0
    for scene_path in paths:
        print(f"---------Getting depth, rgb and tracks from {scene_path} and storing in scene {idx}------------")
        depthDir = os.path.join(train_dataset_path, 'Depth/', f'Scene{idx}/')
        rgbDir = os.path.join(train_dataset_path, 'RGB/', f'Scene{idx}/')
        txtDir = os.path.join(train_dataset_path, 'Text/')
        if depthDir not in os.listdir(train_dataset_path):
            Path(depthDir).mkdir(exist_ok=True, parents=True)

        if rgbDir not in os.listdir(train_dataset_path):          
            Path(rgbDir).mkdir(exist_ok=True, parents=True)

        if txtDir not in os.listdir(train_dataset_path):
            Path(txtDir).mkdir(exist_ok=True, parents=True)
        jsonData = {}
        if idx not in jsonData:
            jsonData[idx] = {}
        # print(depthDir, type(rgbDir), jsonDir)
         
        scene_output = [convert_frame(o, i) for i, o in enumerate(env.run_scene(scene_path))]
       	 
        for framenum, frame in enumerate(scene_output): 
            camera_info = frame.camera_info  
            img = frame.image
            objs = frame.obj_data
            # structs = frame.struct_obj_data
            depth = frame.depth_mask
            obj_masks = split_obj_masks(frame.obj_mask, len(objs))
            #struct_masks = split_obj_masks(frame.struct_mask, len(structs))
            obj_pos, obj_present = calc_world_pos(depth, frame.obj_mask, camera_info)
             
            cv2.imwrite(f'{rgbDir}'+'rgbImg_'+f'{framenum}'+'.png', np.float32(img))

            plt.imshow(depth)
            plt.savefig(f'{depthDir}'+'depth_'+f'{framenum}'+'.png')
            
            if framenum not in jsonData[idx]:
                jsonData[idx][framenum] = {}

            for i in range(len(obj_pos)):
                if i not in jsonData[idx][framenum]:
                    jsonData[idx][framenum][i] = {}
                    jsonData[idx][framenum][i]['world coordinates'] = obj_pos[i].numpy()
                else:
                    jsonData[idx][framenum][i]['world coordinates'] = obj_pos[i].numpy()

            for obj_i, (obj, obj_mask) in enumerate(zip(objs, obj_masks)):
                if True not in obj_mask:
                        # Remove any object which doesn't have a valid mask.
                    print('Empty Mask found. It will be ignored for scene processing')
                    objs.remove(obj)
                    del obj_masks[obj_i]
                else:
                    (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) = get_mask_box(obj_mask)
                    pos_x, pos_y = (top_left_x+bottom_right_x)/2, (top_left_y+bottom_right_y)/2
                    
                    wx, wy, wz = jsonData[idx][framenum][obj_i]['world coordinates']
                    obj_image = frame.image.crop((top_left_y, top_left_x, bottom_right_y, bottom_right_x))
                    jsonData[idx][framenum][obj_i]['area'] = np.prod(obj_image.size).astype(float)
                    w,h = obj_image.size
                    conf = 0
                    print("\nFrame num: Height of object image:\t", framenum, w)
                    print("\nWidth of object image:\t", h)
                    jsonData[idx][framenum][obj_i]['bounding box'] = [pos_x, pos_y]
                    jsonData[idx][framenum][obj_i]['height'] = h
                    jsonData[idx][framenum][obj_i]['width'] = w
                    file1 = open(f'{txtDir}'+ 'scene_' + f'{idx}'+'.txt', 'a')
                    file1.write(f'{framenum}'+" "+ f'{obj_i}'+" "+ f'{pos_x}'+" "+f'{pos_y}'+" "+ f'{w}'+" "+f'{h}'+" "+f'{conf}'+" "+f'{wx}'+" "+f'{wy}'+" "+f'{wz}'+"\n")

            #to add into json
            # with open(f'{txtDir}'+ 'scene_' + f'{idx}'+'.json', 'w') as outfile:
            #     json.dump(jsonData[idx], outfile, cls=NumpyEncoder)
        
        
        idx+=1


def process_scene(scene_data):
    data = {'images': [], 'shapes': [], 'materials': [], 'textures': []}
    for frame_num, frame in enumerate(scene_data):
        img = frame.image
        objs = frame.obj_data
        structs = frame.struct_obj_data
        depth = frame.depth_mask
        obj_masks = split_obj_masks(frame.obj_mask, len(objs))
        struct_masks = split_obj_masks(frame.struct_mask, len(structs))
        
        # Display objects
        for obj_i, (obj, obj_mask) in enumerate(zip(objs, obj_masks)):
            if True not in obj_mask:
                    # Remove any object which doesn't have a valid mask.
                print('Empty Mask found. It will be ignored for scene processing')
                objs.remove(obj)
                del obj_masks[obj_i]
            else:
                (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) = get_mask_box(obj_mask)
                obj_image = frame.image.crop((top_left_y, top_left_x, bottom_right_y, bottom_right_x))
                depth_image = Image.fromarray(depth).crop((top_left_y, top_left_x, bottom_right_y, bottom_right_x))
                depth_image = depth_image.resize((50, 50))
                depth_image = np.array(depth_image).reshape(50, 50)
                    # Todo: Think again about this re-sizing
                    # this is to ensure all images have same size.
                obj_image = obj_image.resize((50, 50))
                obj_image = np.array(obj_image).reshape(3, 50, 50)  # Because channels comes first for Conv2d
                data['images'].append(np.array(obj_image))
                data['shapes'].append(obj.shape)
                data['materials'].append(obj.material_list)
                data['textures'].append(obj.texture_color_list)
                
                if 'depth' not in data:
                    data['depth'] = [np.array(depth_image)]
                else:
                    data['depth'].append(np.array(depth_image))

    print('Len of Dataset:', len(data['images']))
    for x in data:
        data[x] = np.array(data[x])
    return data
    


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

def convert_scenes(env, paths, train_dataset_path):
    for scene_path in paths:
        check_out_path = scene_path.with_suffix('.p')
        if check_out_path.exists():
            print(f'{out_path} exists, skipping')
            continue
        print(f'Converting {scene_path}')
        out_path = os.path.basename(scene_path)
        out_path = out_path.split('.json')[0]+'.p'
        scene_output = [convert_frame(o, i) for i, o in enumerate(env.run_scene(scene_path))]
        # with gzip.open(out_path, 'wb') as fd:
        #     pickle.dump(scene_output, fd)
        data = process_scene(scene_output)
       
        pickle.dump(data, open(os.path.join(train_dataset_path, out_path), 'wb'))



# --------uncomment only if you already have rgb and not depth cropped images. This function adds depth in the rgb data--------

# def get_depth_from_scenes(env, paths, train_dataset_path):
#     i=0
#     for scene_path in paths:
#         check_out_path = scene_path.with_suffix('.p')
#         print(check_out_path)
#         if check_out_path.exists():
#             print(f'{check_out_path} exists..')
#             print(f'Adding depth to {scene_path}')
#             scene = pickle.load(open(check_out_path, 'rb'))
#             print(scene.keys(), len(scene['images']))

#             out_path = os.path.basename(scene_path)
#             out_path = out_path.split('.json')[0]+'.p'
#             print(out_path,train_dataset_path)
#             scene_output = [getDepth(o, i) for i, o in enumerate(env.run_scene(scene_path))]
            
#             for framenum, frame in enumerate(scene_output):
#                 img = frame.image
#                 objs = frame.obj_data
#                 depth = frame.depth_mask
#                 obj_masks = split_obj_masks(frame.obj_mask, len(objs))
                
#                 for obj_i, (obj, obj_mask) in enumerate(zip(objs, obj_masks)):
#                     if True not in obj_mask:
#                             # Remove any object which doesn't have a valid mask.
#                         print('Empty Mask found. It will be ignored for scene processing')
#                         objs.remove(obj)
#                         del obj_masks[obj_i]
#                     else:
#                         (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) = get_mask_box(obj_mask)
#                         depth_image = Image.fromarray(depth).crop((top_left_y, top_left_x, bottom_right_y, bottom_right_x))
#                         depth_image = depth_image.resize((50, 50))
#                         depth_image = np.array(depth_image).reshape(50, 50)
#                         if 'depth' not in scene:
#                             scene['depth'] = [np.array(depth_image)]
#                         else:
#                             scene['depth'].append(np.array(depth_image))
                            
#             if len(scene['images'])!=len(scene['depth']):
#                 i+=1
#             print("mismatch depth and rgb",i)
#             pickle.dump(scene, open(os.path.join(train_dataset_path, out_path), 'wb'))
#             print("---done---")

#         else:
#             print("No Json file found so ignoring for now...")
#                 # save_depth_image(frame, result_dir = '/home/gulsh/mcs_opics/masks-occluder/evalTest/', framenum = framenum)

        


      
def save_depth_image(rgb,depth_img, result_dir, framenum):
    # print(depth_img.shape)
    rgb_img = Image.fromarray(rgb, 'RGB')
    rgb_img.save(f'{result_dir}'+'rgb_'+{framenum}+'.png')
    plt.imshow(depth_img)
    plt.savefig(f'{result_dir}'+'depth_'+{framenum}+'.png')
    
    # imgSaver.save_single_pilImage_gray(depth_img, 'range',
    #          save_path=os.path.join(result_dir, 'depth_'+f'{framenum}'+'.png'))

def getDepth(o,i):
    objs = o.object_list
    structs = o.structural_object_list
    img = o.image_list[-1]
    obj_mask = convert_obj_mask(o.object_mask_list[-1], objs)
    depth_mask = o.depth_map_list[-1]
    return ThorFrame(objs, structs, img, depth_mask, obj_mask)


def convert_frame(o, i):
    objs = o.object_list
    structs = o.structural_object_list
    img = o.image_list[-1]
    obj_mask = convert_obj_mask(o.object_mask_list[-1], objs)
    struct_mask = convert_obj_mask(o.object_mask_list[-1], structs)
    depth_mask = np.array(o.depth_map_list[-1])
    camera_desc = make_camera(o)
        # Project depth map to a 3D point cloud - removed for performance
        # depth_pts = depth_to_points(depth_mask, *camera_desc)
    return ThorFrame(objs, structs, img, depth_mask, obj_mask, struct_mask, camera_desc)
    


def convert_obj_mask(mask, objs):
    convert_color = lambda col: (col['r'], col['g'], col['b'])
    color_map = {convert_color(o.color):i for i, o in enumerate(objs)}
    arr_mask = np.array(mask)
    out_mask = -np.ones(arr_mask.shape[0:2], dtype=np.int8)
    for x in range(arr_mask.shape[0]):
        for y in range(arr_mask.shape[1]):
            idx = color_map.get(tuple(arr_mask[x, y]), -1)
            out_mask[x, y] = idx
    return out_mask

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--sim', type=Path, default=Path('data/thor'))
    parser.add_argument('--scenes', type=Path, default=Path('/home/gulsh/mcs_opics/masks-occluder/scenes/'))
    parser.add_argument('--train_dataset_path', type=Path,default=Path('/home/gulsh/mcs_opics/masks-occluder/tracking_data/'))
    parser.add_argument('--config', type=Path, default=Path('/Users/jaypatravali/Documents/mcs_opics/mcs_config.ini'))
    parser.add_argument('--filter', type=str, default=None)
    return parser


def main(sim_path, data_path, config_path, train_dataset_path,filter):
    env = McsEnv(sim_path, data_path, config_path, filter)

    scenes = list(env.all_scenes)
    print(f'Found {len(scenes)} scenes')
    random.shuffle(scenes)
    # Work around stupid sim bug
    Path('SCENE_HISTORY/evaluation3Training').mkdir(exist_ok=True, parents=True)
    # convert_scenes(env, scenes, train_dataset_path)
    get_data_for_tracking(env, scenes, train_dataset_path)
    # get_depth_from_scenes(env, scenes, train_dataset_path)

if __name__ == '__main__':
    args = make_parser().parse_args()
    training_data_Dir = args.train_dataset_path
    Path(training_data_Dir).mkdir(exist_ok=True, parents=True)
    main(args.sim, args.scenes, args.config, args.train_dataset_path, args.filter)
