from masks import ThorFrame, CameraInfo

import pickle
import gzip
import numpy as np
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser
import cv2
import os
import math

show_images = False
import matplotlib.pyplot as plt


def plot_line(positions):
    x, y = [], []
    for p in positions:
        x.append(p['x'])
        y.append(p['y'])

    plt.plot(x, y)
    plt.show()


def l2_distance(src_pos, dest_pos):
    return np.sqrt(sum((src_pos[axis] - dest_pos[axis]) ** 2 for axis in ['x', 'y']))


def obj_matches_track(position_history, new_position):
    if len(position_history) == 1:
        # Note: If we have only 1 point, then we cannot estimate direction of motion.
        # Therebym we just rely on distance metric with new position.
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


def get_mask_box(obj_mask):
    height, width = obj_mask.shape
    rows, cols = np.where(obj_mask == True)
    box_top_x, box_top_y = max(0, rows.min() - 1), max(0, cols.min() - 1)
    box_bottom_x, box_bottom_y = min(rows.max() + 1, height - 1), min(cols.max() + 1, width - 1)

    return (box_top_x, box_top_y), (box_bottom_x, box_bottom_y)


def get_obj_position(obj_mask):
    (box_top_x, box_top_y), (box_bottom_x, box_bottom_y) = get_mask_box(obj_mask)
    position = {'x': (box_top_x + box_bottom_x) / 2, 'y': (box_top_y + box_bottom_y) / 2}
    return position


def process_scene(data, scene_path, save_mp4=True):
    obj_key_idx = -1
    obj_trackers = {}  # id:
    obj_history = []
    processed_scenes = []
    for frame_num, frame in enumerate(data):
        img = frame.image
        objs = frame.obj_data
        obj_history.append(len(objs))
        depth = frame.depth_mask
        obj_masks = split_obj_masks(frame.obj_mask, len(objs))

        # Remove any object which doesn't have a valid mask.
        for frame_i, (frame_obj, frame_obj_mask) in enumerate(zip(objs, obj_masks)):
            if True not in frame_obj_mask:
                print('error')
                objs.remove(frame_obj)
                obj_masks.remove(frame_obj_mask)

        if len(objs) == 0:
            for obj_key in obj_trackers:
                obj_trackers[obj_key]['visible'] = False
            processed_scenes.append(img)
            continue  # Wait for an interesting frame

        # #############################
        # Object Tracking
        # #############################

        resolved_objs = []
        for frame_obj, frame_obj_mask in zip(objs, obj_masks):
            frame_obj_position = get_obj_position(frame_obj_mask)
            track_to_exist_obj = []
            for exist_obj_key, exist_obj in obj_trackers.items():
                print(exist_obj_key)
                if exist_obj_key not in resolved_objs:
                    theta, flag = obj_matches_track(exist_obj['position_history'], frame_obj_position)
                    if flag:
                        dis = l2_distance(exist_obj['position_history'][-1], frame_obj_position)
                        track_to_exist_obj.append((theta, dis, exist_obj_key))

            (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) = get_mask_box(frame_obj_mask)
            position = {'x': (top_left_x + bottom_right_x) / 2, 'y': (top_left_y + bottom_right_y) / 2}
            if len(track_to_exist_obj) == 0:
                # add as a new object
                obj_key_idx += 1
                obj_trackers[obj_key_idx] = {'position_history': []}
                _key = obj_key_idx
            else:
                _, _, _key = min(track_to_exist_obj)

            resolved_objs.append(_key)
            obj_trackers[_key]['bounding_box'] = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
            obj_trackers[_key]['position_history'].append(position)
            obj_trackers[_key]['mask'] = frame_obj_mask
            obj_trackers[_key]['visible'] = True

        for obj_key, obj in obj_trackers.items():
            if obj_key not in resolved_objs:
                obj['visible'] = False

        visible_obj_tracked = len([o for o in obj_trackers.values() if o['visible']])
        try:
            assert len(objs) == visible_obj_tracked, \
                ' no. of objects are not matching, {} {}'.format(len(objs), visible_obj_tracked)
        except:
            pass

        # #######################################################
        # Draw Bounding box around objects and also show their id
        # #######################################################

        box_img = np.array(img)
        for obj_key, obj_info in obj_trackers.items():
            if obj_info['visible']:
                (box_top_x, box_top_y), (box_bottom_x, box_bottom_y) = get_mask_box(obj_info['mask'])
                box_img = cv2.rectangle(box_img, (box_top_y, box_top_x), (box_bottom_y, box_bottom_x), (255, 255, 0), 2)
                box_img = cv2.putText(box_img, str(obj_key), (box_top_y, box_top_x), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            pass

        processed_scenes.append(Image.fromarray(box_img))

    processed_scenes[0].save(scene_path + '.gif', save_all=True,
                             append_images=processed_scenes[1:], optimize=False, loop=1)
    # save video
    if save_mp4:
        import moviepy.editor as mp
        clip = mp.VideoFileClip(scene_path + '.gif')
        clip.write_videofile(scene_path + '.mp4')
        os.remove(scene_path + '.gif')


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
        process_scene(scene_data, os.path.join(os.getcwd(), scene_file.name))


def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--scenes', type=Path)
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args.scenes)
