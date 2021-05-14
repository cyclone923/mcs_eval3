from .mcs_env import McsEnv
from .types import ThorFrame, CameraInfo
import os
import pickle
import random
import itertools
import gzip
import numpy as np
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser
#from vision.generateData.instSeg_parse_mask import parse_label_info, save_depth_image
from .utils import draw_bounding_boxes, draw_appearance_bars, split_obj_masks, get_obj_position, get_mask_box



#Eval 4 training data-> 
# Step 1: Process json scenes into mask-object RGBD scenes (done by eval4_data_gen.py)
# Step 2: Fetch the cropped objects and annotate the images as per classes of objects (siameseAppearence.py)




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

                    # Todo: Think again about this re-sizing
                    # this is to ensure all images have same size.
                obj_image = obj_image.resize((50, 50))
                obj_image = np.array(obj_image).reshape(3, 50, 50)  # Because channels comes first for Conv2d
                data['images'].append(np.array(obj_image))
                data['shapes'].append(obj.shape)
                data['materials'].append(obj.material_list)
                data['textures'].append(obj.texture_color_list)

    print('Len of Dataset:', len(data['images']))
    for x in data:
        data[x] = np.array(data[x])
    return data
    # pickle.dump(data, open(args.scenes, 'wb'))
        
        # Display structures (occluders, walls)
        #Not needed for training data

        # for struct_num, (struct, mask) in enumerate(zip(structs, struct_masks)):
        #     name = struct.uuid #Look at this to determine what kind of occluder it is
        #     if mask.sum() == 0: continue
        #     struct_img = mask_img(mask, img)
        #     struct_img.save(f'{frame_num:02d}_STRUCT_{name}.png')
        # import pdb ; pdb.set_trace()


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


def convert_frame(o, i):
    objs = o.object_list
    structs = o.structural_object_list
    img = o.image_list[-1]
    obj_mask = convert_obj_mask(o.object_mask_list[-1], objs)
    struct_mask = convert_obj_mask(o.object_mask_list[-1], structs)
    depth_mask = np.array(o.depth_map_list[-1])
    # print(len(o.object_mask_list), len(o.depth_map_list), len(o.image_list))
    camera_desc = CameraInfo(o.camera_field_of_view, o.position, o.rotation, o.head_tilt)
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
    parser.add_argument('--scenes', type=Path, default=Path('data/thor/scenes'))
    parser.add_argument('--train_dataset_path', type=Path,default=Path('/home/gulsh/mcs_opics/masks-occluder/eval4dataset-1/'))
    parser.add_argument('--config', type=Path, default=Path('/home/gulsh/mcs_opics/mcs_config.ini'))
    parser.add_argument('--filter', type=str, default=None)
    return parser


def main(sim_path, data_path, config_path, train_dataset_path,filter):
    env = McsEnv(sim_path, data_path, config_path, filter)

    scenes = list(env.all_scenes)
    print(f'Found {len(scenes)} scenes')
    random.shuffle(scenes)
    # Work around stupid sim bug
    Path('SCENE_HISTORY/evaluation3Training').mkdir(exist_ok=True, parents=True)
    convert_scenes(env, scenes, train_dataset_path)

if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args.sim, args.scenes, args.config, args.train_dataset_path, args.filter)