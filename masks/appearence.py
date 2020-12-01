import pickle
import gzip
from pathlib import Path
from argparse import ArgumentParser
import os
from utils import draw_bounding_boxes, draw_appearance_bars, split_obj_masks, get_obj_position, get_mask_box

from track import track_objects
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from tqdm import tqdm


def obj_image_to_tensor(obj_image):
    obj_image = obj_image.resize((50, 50))
    obj_image = np.array(obj_image)
    obj_image = obj_image.reshape((3, 50, 50))
    obj_image = torch.Tensor(obj_image).float()
    return obj_image


class AppearanceMatchModel(nn.Module):
    def __init__(self, labels):
        super(AppearanceMatchModel, self).__init__()

        self._labels = labels
        self.feature = nn.Sequential(nn.Conv2d(3, 6, 2),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(6, 6, 3),
                                     nn.LeakyReLU())
        self.shape_classifier = nn.Sequential(nn.Linear(13254, len(self._labels)))

    def label(self, id):
        return self._labels[id]

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.flatten(1)
        return feature, self.shape_classifier(feature)


def object_appearance_match(appearance_model, frame, objects_info, device='cpu'):
    for obj_key in objects_info:

        top_x, top_y, bottom_x, bottom_y = objects_info[obj_key]['bounding_box']
        obj_current_image = frame.image.crop((top_y, top_x, bottom_y, bottom_x))
        obj_current_image = obj_image_to_tensor(obj_current_image).to(device)
        obj_current_image = obj_current_image.unsqueeze(0)

        _, object_shape_logit = appearance_model(obj_current_image)
        object_shape_logit = object_shape_logit.squeeze(0)
        object_shape_prob = torch.softmax(object_shape_logit, dim=0)
        current_object_shape_id = torch.argmax(object_shape_prob).item()

        if 'base_image' not in objects_info[obj_key]:
            objects_info[obj_key]['base_image'] = {}
            objects_info[obj_key]['appearance'] = {}

            objects_info[obj_key]['base_image']['shape_id'] = current_object_shape_id
            objects_info[obj_key]['base_image']['shape'] = model.label(current_object_shape_id)
            objects_info[obj_key]['appearance']['match'] = True
            objects_info[obj_key]['appearance']['prob'] = object_shape_prob[current_object_shape_id].item()
        else:
            base_shape_id = objects_info[obj_key]['base_image']['shape_id']
            objects_info[obj_key]['appearance']['prob'] = object_shape_prob[base_shape_id].item()
            objects_info[obj_key]['appearance']['match'] = current_object_shape_id == base_shape_id

    return objects_info


def process_video(video_data, appearance_model, save_path=None, save_mp4=False, device='cpu'):
    track_info = {}
    processed_frames = []
    for frame_num, frame in enumerate(video_data):
        track_info = track_objects(frame.obj_mask, track_info)
        track_info['objects'] = object_appearance_match(appearance_model, frame,
                                                        track_info['objects'], device)

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


class ObjectDataset(Dataset):
    """ Dataset of objects with labels indictating their shape"""

    def __init__(self, data):
        self.data = data
        self._labels = {name: id for id, name in enumerate(sorted(set(data['shapes'])))}
        self.data['shapes'] = np.array(self.data['shapes'])

        for shape, shape_id in self._labels.items():
            self.data['shapes'][self.data['shapes'] == shape] = shape_id

    def get_label(self, id):
        return self._labels[id]

    @property
    def labels(self):
        return sorted([k for k in self._labels])

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return {'images': self.data['images'][idx],
                'shapes': int(self.data['shapes'][idx])}


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


def train_appearance_matching(dataloader, model, optimizer, epochs: int, writer, checkpoint_path,
                              checkpoint_interval: int = 1, log_interval: int = 1,
                              restore_checkpoint=False):
    if restore_checkpoint:
        checkpoint = torch.load(restore_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch']
    else:
        init_epoch = 0

    model.train()
    for epoch in range(init_epoch, epochs):
        batch_losses = {'shape': []}
        for i_batch, batch in enumerate(dataloader):
            object_image = batch['images']
            object_shape = batch['shapes']

            object_image = object_image.float()
            feature, shape_logits = model(object_image)

            shape_loss = nn.CrossEntropyLoss()(shape_logits, object_shape)
            batch_losses['shape'].append(shape_loss.item())
            optimizer.zero_grad()
            shape_loss.backward()
            optimizer.step()

        writer.add_scalar('losses/shape', np.mean(batch_losses['shape']), epoch)

        if (epoch % checkpoint_interval) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

            # save model separately as well
            torch.save(model.state_dict(), model_path)

        if epoch % log_interval == 0:
            print('Epoch: {} Shape Loss: {}'.format(epoch, np.mean(batch_losses['shape'])))


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
    parser.add_argument('--opr', choices=['generate_dataset', 'train', 'demo'], default='demo',
                        help='operation (opr) to be performed')

    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # paths
    experiment_path = os.path.join(args.results_dir, 'run_{}'.format(args.run), )
    os.makedirs(experiment_path, exist_ok=True)
    log_path = os.path.join(experiment_path, 'logs')
    model_path = os.path.join(experiment_path, 'model.p')
    checkpoint_path = os.path.join(experiment_path, 'checkpoint.p')

    # Determine the operation to be performed
    if args.opr == 'generate_dataset':
        train_scenes = list(args.train_scenes_path.glob('*.pkl.gz'))
        data = generate_data(train_scenes)
        pickle.dump(data, open(args.train_dataset_path, 'wb'))

        test_scenes = list(args.test_scenes_path.glob('*.pkl.gz'))
        data = generate_data(test_scenes)
        pickle.dump(data, open(args.test_dataset_path, 'wb'))

    elif args.opr == 'train':
        # flush every 1 minutes
        summary_writer = SummaryWriter(log_path, flush_secs=60 * 1)

        object_dataset = ObjectDataset(pickle.load(open(args.train_dataset_path, 'rb')))
        dataloader = DataLoader(object_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
        model = AppearanceMatchModel(object_dataset.labels)
        optimizer = Adam(model.parameters(), lr=args.lr)

        # train
        train_appearance_matching(dataloader, model, optimizer, args.epochs, summary_writer,
                                  checkpoint_path, args.checkpoint_interval, args.log_interval)

    elif args.opr == 'demo':
        all_scenes = list(args.test_scenes_path.glob('*.pkl.gz'))
        print(f'Found {len(all_scenes)} scenes')

        # Todo: Don't load dataset over here. It's only required for label count
        train_object_dataset = ObjectDataset(pickle.load(open(args.train_dataset_path, 'rb')))
        model = AppearanceMatchModel(train_object_dataset.labels)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model = model.to(args.device)

        for scene_file in all_scenes:
            with gzip.open(scene_file, 'rb') as fd:
                scene_data = pickle.load(fd)

            print(f'{scene_file.name}')
            process_video(scene_data, model, os.path.join(os.getcwd(), scene_file.name), save_mp4=True,
                          device=args.device)
