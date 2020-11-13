from .types import ThorFrame, CameraInfo

import pickle
import gzip
import numpy as np
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser


def process_scene(data):
    for frame_num, frame in enumerate(data):
        img = frame.image
        objs = frame.obj_data
        depth = frame.depth_mask
        obj_masks = split_obj_masks(frame.obj_mask, len(objs))
        if len(objs) < 1:
            continue # Wait for an interesting frame
        img.save(f'{frame_num:02d}_img.png')
        for obj_num, (obj, mask) in enumerate(zip(objs, obj_masks)):
            shape = obj.shape
            obj_img = mask_img(mask, img)
            obj_img.save(f'{frame_num:02d}_{shape}{obj_num}.png')
        import pdb ; pdb.set_trace()


def split_obj_masks(mask, num_objs):
    obj_masks = []
    for obj_idx in range(num_objs):
        obj_masks.append(mask == obj_idx)
    return obj_masks


def mask_img(mask, img):
    img_arr = np.asarray(img)
    masked_arr = img_arr * mask[:, :, np.newaxis]
    return Image.fromarray(masked_arr)


def main(scenes_path):
    all_scenes = list(scenes_path.glob('*.pkl.gz'))
    print(f'Found {len(all_scenes)} scenes')
    for scene_file in all_scenes:
        with gzip.open(scene_file, 'rb') as fd:
            scene_data = pickle.load(fd)
        print(f'{scene_file.name}')
        process_scene(scene_data)


def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--scenes', type=Path)
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args.scenes)
