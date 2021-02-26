import pickle
import gzip
from pathlib import Path
from argparse import ArgumentParser
import os
from .utils import draw_bounding_boxes, draw_appearance_bars, split_obj_masks, get_obj_position, get_mask_box, rgb_to_grayscale, closest_colour, getNearestWebSafeColor, get_colour_name, obj_image_to_tensor

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
        self.OPENCV_OBJECT_TRACKERS = {
            'csrt': cv2.TrackerCSRT_create,
            'kcf': cv2.TrackerKCF_create,
            'mil': cv2.TrackerMIL_create,
            'goturn': cv2.TrackerGOTURN_create
        }
    
    def initNewTracker(self, frame, bbox, type='kcf'):
        return self.OPENCV_OBJECT_TRACKERS[type]().init(frame, bbox)

    def object_appearance_match(self, frame, objects_info, device='cpu', level='level2'):
        for key, obj in objects_info.items():
            if not obj['visible']:
                continue
            
            initBB = obj['bounding_box']
            obj_current_image = frame.crop(initBB)

            image_area = np.prod(obj_current_image.size)
            base_image = np.array(frame)
            mask_image = np.zeroes(obj['mask'].shape, dtype=base_image.dtype)
            mask_image[obj['mask']] = 255

            if 'base_image' not in obj.keys() or (len(obj['position_history'] < 5) and obj['base_image']['image_area'] < image_area):
                obj['base_image'] = dict()
                obj['tracker'] = self.initNewTracker(frame, initBB)
                obj['appearance'] = dict()
            
            # timer = cv2.getTickCount()

            # Update tracker
            ok, currBB = obj['tracker'].update(frame)
            # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # if ok:
            #     p1 = (int(currBB[0]), int(currBB[1]))
            #     p2 = (int(currBB[0] + currBB[2]), int(currBB[1] + currBB[3]))
            #     cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            # else:
            #     cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

            # cv2.putText(frame, "Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
            # cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
            # cv2.imshow("Tracking", frame)

            if not obj['occluded']:
                obj['appearance']['match'] = ok

                if 'mismatch_count' not in obj['appearance']:
                    obj['appearance']['mismatch_count'] = 0

                if obj['appearance']['match']:
                    obj['appearance']['mismatch_count'] = 0
                else:
                    obj['appearance']['mismatch_count'] += 1

        return objects_info            


def generate_data(scenes_files):
    data = {'images': [], 'shapes': [], 'materials': [], 'textures': []}
    for scene_file in tqdm(sorted(scenes_files)):
        with gzip.open(scene_file, 'rb') as fd:
            scene_data = pickle.load(fd)

        for frame_num, frame in enumerate(scene_data):

            objs = frame.obj_data
            obj_masks = split_obj_masks(frame.obj_mask, len(objs))

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

    for x in data:
        data[x] = np.array(data[x])

    print('Len of Dataset:', len(data['images']))
    return data

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--test-scenes-path', required=True, type=Path)
    parser.add_argument('--train-scenes-path', required=True, type=Path)
    parser.add_argument('--train-dataset-path', required=False, type=Path,
                        default=os.path.join(os.getcwd(), 'train_object_dataset.p'))
    parser.add_argument('--test-dataset-path', required=False, type=Path,
                        default=os.path.join(os.getcwd(), 'test_object_dataset.p'))
    parser.add_argument('--batch-size', required=False, type=int, default=32)
    parser.add_argument('--results-dir', required=False, type=Path, default=os.path.join(os.getcwd(), 'results'))
    parser.add_argument('--run', required=False, type=int, default=1)
    parser.add_argument('--lr', required=False, type=float, default=0.001)
    parser.add_argument('--epochs', required=False, type=int, default=50)
    parser.add_argument('--checkpoint-interval', required=False, type=int, default=1)
    parser.add_argument('--log-interval', required=False, type=int, default=1)
    parser.add_argument('--opr', choices=['generate_dataset', 'run', 'demo'], default='demo',
                        help='operation (opr) to be performed')

if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    args = make_parser().parse_args()
    args.device = 'cuda' if torch.cuda_is_available() else 'cpu'

    # paths
    experiment_path = os.path.join(args.results_dir, 'run_{}'.format(args.run), )
    os.makedirs(experiment_path, exist_ok=True)
    log_path = os.path.join(experiment_path, 'logs')
    # model_path = os.path.join(experiment_path, 'model.p')
    # checkpoint_path = os.path.join(experiment_path, 'checkpoint.p')

    # Determine the operation to be performed
    if args.opr == 'generate_dataset':
        # train_scenes = list(args.train_scenes_path.glob('*.pkl.gz'))
        # data = generate_data(train_scenes)
        # pickle.dump(data, open(args.train_dataset_path, 'wb'))

        # test_scenes = list(args.test_scenes_path.glob('*.pkl.gz'))
        # data = generate_data(test_scenes)
        # pickle.dump(data, open(args.test_dataset_path, 'wb'))
        pass
    
    elif args.opr == 'run':
        pass

    elif args.opr == 'demo':
        all_scenes = list(args.train_scenes_path.glob('*.pkl.gz'))
        print(f'Found {len(all_scenes)} scenes')

        # model = pickle.load(model_path)

        violations = list()
        np.random.seed(0)
        np.random.shuffle(all_scenes)
        mismatch_cases = list()

        for idx, scene_file in enumerate(all_scenes):
            with gzip.open(scene_file, 'rb') as fd:
                scene_data = pickle.load(fd)
            
            print(f'{idx:} {scene_file.name}')
            # v = model.process_video(scene_data, os.path.join(os.getcwd(), scene_file.name), save_mp4=True, device=args.device, save_gif=True)
            v = None

            if v:
                mismatch_cases.append(scene_file)
            violations.append(v)
        print(mismatch_cases)
        print((len(violations) - sum(violations)) / len(violations))