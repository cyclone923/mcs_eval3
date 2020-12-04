import gzip
import math
import os
import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from .utils import draw_bounding_boxes, split_obj_masks, get_obj_position, get_mask_box


def l2_distance(src_pos, dest_pos):
    return np.sqrt(sum((src_pos[axis] - dest_pos[axis]) ** 2 for axis in ['x', 'y']))


def track_objects(frame_mask, track_info={}):
    if 'object_index' not in track_info:
        track_info['object_index'] = 0
    if 'objects' not in track_info:
        track_info['objects'] = {}

    # Note : Assumption of sequential order
    objs = frame_mask.max() + 1
    obj_masks = split_obj_masks(frame_mask, objs)

    # Remove any object which doesn't have a valid mask.
    for frame_obj_mask in obj_masks:
        if True not in frame_obj_mask:
            objs -= 1
            obj_masks.remove(frame_obj_mask)

    # process objects
    resolved_objs = []
    for frame_obj_mask in obj_masks:
        frame_obj_position = get_obj_position(frame_obj_mask)
        track_to_exist_obj = []
        for exist_obj_key, exist_obj in track_info['objects'].items():
            if exist_obj_key not in resolved_objs:
                theta, flag = obj_matches_track(exist_obj['position_history'], frame_obj_position)
                if flag:
                    dis = l2_distance(exist_obj['position_history'][-1], frame_obj_position)
                    track_to_exist_obj.append((theta, dis, exist_obj_key))

        (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) = get_mask_box(frame_obj_mask)
        position = {'x': (top_left_x + bottom_right_x) / 2, 'y': (top_left_y + bottom_right_y) / 2}

        if len(track_to_exist_obj) == 0:
            # add as a new object
            _key = track_info['object_index']
            track_info['object_index'] += 1
            if 'objects' not in track_info:
                track_info['objects'] = {}
            track_info['objects'][_key] = {'position_history': [], 'area_history': {}, 'hidden_for': 0}

        else:
            _, _, _key = min(track_to_exist_obj)

        resolved_objs.append(_key)
        track_info['objects'][_key]['bounding_box'] = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        track_info['objects'][_key]['position_history'].append(position)
        track_info['objects'][_key]['mask'] = frame_obj_mask
        track_info['objects'][_key]['area_history'] = frame_obj_mask.sum()
        track_info['objects'][_key]['visible'] = True
        track_info['objects'][_key]['hidden_for'] = 0

    for obj_key, obj in track_info['objects'].items():
        if obj_key not in resolved_objs:
            obj['visible'] = False
            prev_mask = track_info['objects'][obj_key]['mask']
            track_info['objects'][obj_key]['mask'] = np.zeros_like(prev_mask)
            track_info['objects'][obj_key]['hidden_for'] += 1

    visible_obj_tracked = len([o for o in track_info['objects'].values() if o['visible']])
    assert objs == visible_obj_tracked, \
        ' no. of objects are not matching, {} {}'.format(objs, visible_obj_tracked)

    return track_info


def process_video(video_data, save_path=None, save_mp4=False):
    track_info = {}
    processed_frames = []
    for frame_num, frame in enumerate(video_data):
        track_info = track_objects(frame.obj_mask, track_info)
        processed_frames.append(draw_bounding_boxes(frame.image, track_info['objects']))

    # save gif
    processed_frames[0].save(save_path + '.gif', save_all=True,
                             append_images=processed_frames[1:], optimize=False, loop=1)

    # save video
    if save_mp4:
        import moviepy.editor as mp
        clip = mp.VideoFileClip(save_path + '.gif')
        clip.write_videofile(save_path + '.mp4')
        os.remove(save_path + '.gif')


def obj_matches_track(position_history, new_position):
    # Note: If we have only 1 point, then we cannot estimate direction of motion.
    # There by we just rely on distance metric with new position.
    if len(position_history) == 1:
        return 0, l2_distance(position_history[-1], new_position) < 50

    pt1 = np.mean([[p['x'], p['y']] for p in position_history[:-1]], axis=0)
    pt1 = {'x': pt1[0], 'y': pt1[1]}
    pt2 = position_history[-1]

    # formulate 2 vectors
    # 1st vector : Expected origin  and last point in history
    # 2nd vector: Expected origin and new position
    a = np.array([pt2['x'] - pt1['x'], pt2['y'] - pt1['y']])
    b = np.array([new_position['x'] - pt1['x'], new_position['y'] - pt1['y']])

    # if these vectors are same, simply return True
    if all(a == b):
        return 0, True

    # Otherwise estimate angle between them
    theta = np.arccos(sum(a * b) / (math.sqrt(sum(a ** 2)) * math.sqrt(sum(b ** 2))))
    theta = math.degrees(theta)

    if theta < 45:
        return theta, True
    else:
        return theta, False


def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--scenes-path', required=True, type=Path)
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()

    all_scenes = list(args.scenes_path.glob('*.pkl.gz'))
    print(f'Found {len(all_scenes)} scenes')

    for scene_file in all_scenes:
        with gzip.open(scene_file, 'rb') as fd:
            scene_data = pickle.load(fd)
        print(f'{scene_file.name}')
        process_video(scene_data, os.path.join(os.getcwd(), scene_file.name), save_mp4=True)
