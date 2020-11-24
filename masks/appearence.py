import pickle
import gzip
from pathlib import Path
from argparse import ArgumentParser
import os
from utils import draw_bounding_boxes, draw_appearance_bars, split_obj_masks, get_obj_position, get_mask_box

from track import track_objects
import torch
import numpy as np


def object_appearance_match(appearance_model, objects_info):
    for obj_key in objects_info:
        base_object_image, current_object_image = None, None
        prob = appearance_model(base_object_image, current_object_image)
        objects_info[obj_key]['appearance'] = prob.item()

    return objects_info


def process_video(video_data, appearance_model, save_path=None, save_mp4=False):
    track_info = {}
    processed_frames = []
    for frame_num, frame in enumerate(video_data):
        track_info = track_objects(frame, track_info)
        track_info['objects'] = object_appearance_match(appearance_model, track_info['objects'])

        img = draw_bounding_boxes(frame.image, track_info['objects'])
        img = draw_appearance_bars(img, track_info['objects'])
        processed_frames.append(img)

    # save gif
    processed_frames[0].save(save_path + '.gif', save_all=True,
                             append_images=processed_frames[1:], optimize=False, loop=1)

    # save video
    if save_mp4:
        import moviepy.editor as mp
        clip = mp.VideoFileClip(save_path + '.gif')
        clip.write_videofile(save_path + '.mp4')
        os.remove(save_path + '.gif')


def generate_dataset(scenes_files, save_path):
    dataset = []
    for scene_file in sorted(scenes_files):
        with gzip.open(scene_file, 'rb') as fd:
            scene_data = pickle.load(fd)

        for frame_num, frame in enumerate(scene_data):

            objs = frame.obj_data
            obj_masks = split_obj_masks(frame.obj_mask, len(objs))

            for obj, obj_mask in zip(objs, obj_masks):
                if True not in obj_mask:
                    # Remove any object which doesn't have a valid mask.
                    print('Empty Mask found. It will be ignored for scene processing')
                    objs.remove(obj)
                    obj_masks.remove(obj_mask)
                else:
                    (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) = get_mask_box(obj_mask)
                    obj_image = frame.image.crop((top_left_y, top_left_x, bottom_right_y, bottom_right_x))
                    dataset.append([np.array(obj_image), obj.shape, obj.material_list, obj.texture_color_list])

    pickle.dumps(dataset, save_path)


def train_appearance(dataset, epochs):
    for epoch in epochs:
        for batch in batches:
            batch = dataset.sample()
            pass

        # Todo: Log training loss.
        # Todo: Enable saving of model at regular interval
        # Todo: Checkpoint

    # Todo: return model


def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--scenes-path', required=True, type=Path)
    parser.add_argument('--model-path', required=False, type=Path)
    parser.add_argument('--appearance-dataset-path', required=False, type=Path)
    parser.add_argument('--opr', choices=['generate_dataset', 'train', 'test'], default='test',
                        help='operation (opr) to be performed')
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.opr == 'generate_dataset':
        all_scenes = list(args.scenes_path.glob('*.pkl.gz'))
        generate_dataset(all_scenes, save_path=os.path.join(os.getcwd(), 'dataset.p'))

    elif args.opr == 'train':
        dataset = pickle.load(args.appearance_dataset_path)

        # Todo: Ensure dataset is compatible with pytorch dataloader
        # Todo: load from checkpoint
        # Todo: Create Model
        # Todo: Create Optimizer

        train_appearance(dataset, epochs=100)

    elif args.opr == 'test':
        all_scenes = list(args.scenes_path.glob('*.pkl.gz'))
        print(f'Found {len(all_scenes)} scenes')

        for scene_file in all_scenes:
            appearance_model = lambda x, y: torch.Tensor([1.0])
            with gzip.open(scene_file, 'rb') as fd:
                scene_data = pickle.load(fd)

            print(f'{scene_file.name}')
            process_video(scene_data, appearance_model, os.path.join(os.getcwd(), scene_file.name), save_mp4=True)
