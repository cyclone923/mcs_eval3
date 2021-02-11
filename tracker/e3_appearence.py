import pickle
import gzip
from pathlib import Path
from argparse import ArgumentParser
import os
from .utils import draw_bounding_boxes, draw_appearance_bars, split_obj_masks, get_obj_position, get_mask_box

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


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def getNearestWebSafeColor(x):
    r, g, b = x
    r = int(round((r / 255.0) * 5) * 51)
    g = int(round((g / 255.0) * 5) * 51)
    b = int(round((b / 255.0) * 5) * 51)
    return (r, g, b)


def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


def obj_image_to_tensor(obj_image, gray=False):
    obj_image = obj_image.resize((50, 50))
    if gray:
        obj_image = np.array(obj_image)
        obj_image = obj_image.reshape((3, 50, 50))
        obj_image = torch.Tensor(obj_image).float()
        obj_image = rgb_to_grayscale(obj_image)
    else:
        obj_image = np.array(obj_image)
        obj_image = obj_image.reshape((3, 50, 50))
        obj_image = torch.Tensor(obj_image).float()

    return obj_image


class AppearanceMatchModel(nn.Module):
    def __init__(self):
        super(AppearanceMatchModel, self).__init__()

        self._shape_labels = ['car', 'cone', 'cube', 'cylinder', 'duck', 'sphere', 'square frustum', 'turtle']
        self._color_labels = ['black', 'blue', 'brown', 'green', 'grey', 'red', 'yellow']

        self.feature = nn.Sequential(nn.Conv2d(1, 6, 2),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(6, 6, 3),
                                     nn.LeakyReLU())

        self.color_feature = nn.Sequential(nn.Conv2d(3, 6, 2),
                                           nn.LeakyReLU(),
                                           nn.Conv2d(6, 6, 3),
                                           nn.LeakyReLU())

        self.shape_classifier = nn.Sequential(nn.Linear(13254, len(self._shape_labels)))
        self.color_classifier = nn.Sequential(nn.Linear(13254, len(self._color_labels)))

    def shape_label(self, id):
        return self._shape_labels[id]

    def shape_labels(self):
        return self._shape_labels

    def color_label(self, id):
        return self._color_labels[id]

    def color_labels(self):
        return self._color_labels

    def forward(self, image, gray_image):
        feature = self.feature(gray_image)
        color_feature = self.color_feature(image)
        feature = feature.flatten(1)
        color_feature = color_feature.flatten(1)
        return feature, self.shape_classifier(feature), self.color_classifier(color_feature)


def object_appearance_match(appearance_model, image, objects_info, device='cpu', level='level2'):
    for obj_key in [k for k in objects_info.keys() if objects_info[k]['visible']]:
        top_x, top_y, bottom_x, bottom_y = objects_info[obj_key]['bounding_box']
        obj_current_image = image.crop((top_y, top_x, bottom_y, bottom_x))

        with torch.no_grad():
            image_area = np.prod(obj_current_image.size)
            obj_current_image_tensor = obj_image_to_tensor(obj_current_image).to(device)
            obj_current_image_tensor = obj_current_image_tensor.unsqueeze(0)

            obj_current_gray_image_tensor = obj_image_to_tensor(obj_current_image, gray=True).to(device)
            obj_current_gray_image_tensor = obj_current_gray_image_tensor.unsqueeze(0)

            # extract appearance ( shape) info
            _, object_shape_logit, object_color_logit = appearance_model(obj_current_image_tensor,
                                                                         obj_current_gray_image_tensor)
            object_shape_logit = object_shape_logit.squeeze(0)
            object_shape_prob = torch.softmax(object_shape_logit, dim=0)

            object_color_logit = object_color_logit.squeeze(0)
            object_color_prob = torch.softmax(object_color_logit, dim=0)

        current_object_shape_id = torch.argmax(object_shape_prob).item()
        current_object_color_id = torch.argmax(object_color_prob).item()

        base_image = np.array(image)
        mask_image = np.zeros(objects_info[obj_key]['mask'].shape, dtype=base_image.dtype)
        mask_image[objects_info[obj_key]['mask']] = 255

        obj_clr_hist_0 = cv2.calcHist([np.array(image)], [0], mask_image, [10], [0, 256])
        obj_clr_hist_1 = cv2.calcHist([np.array(image)], [1], mask_image, [10], [0, 256])
        obj_clr_hist_2 = cv2.calcHist([np.array(image)], [2], mask_image, [10], [0, 256])
        obj_clr_hist = (obj_clr_hist_0 + obj_clr_hist_1 + obj_clr_hist_2) / 3

        if ('base_image' not in objects_info[obj_key].keys()) or \
                (len(objects_info[obj_key]['position_history']) < 5 and
                 (image_area > objects_info[obj_key]['base_image']['image_area'])):
            objects_info[obj_key]['base_image'] = {}
            objects_info[obj_key]['appearance'] = {}
            objects_info[obj_key]['base_image']['image_area'] = image_area
            objects_info[obj_key]['base_image']['shape_id'] = current_object_shape_id
            objects_info[obj_key]['base_image']['shape'] = appearance_model.shape_label(current_object_shape_id)
            objects_info[obj_key]['base_image']['color_id'] = current_object_color_id
            objects_info[obj_key]['base_image']['color'] = appearance_model.color_label(current_object_color_id)
            objects_info[obj_key]['base_image']['histogram'] = obj_clr_hist
            base_shape_id = current_object_shape_id
            base_color_id = current_object_color_id
        else:
            base_shape_id = objects_info[obj_key]['base_image']['shape_id']
            base_color_id = objects_info[obj_key]['base_image']['color_id']

        # shape match
        objects_info[obj_key]['appearance']['shape_match_quotient'] = object_shape_prob[base_shape_id].item()
        objects_info[obj_key]['appearance']['shape_prob'] = object_shape_prob.cpu().numpy()
        objects_info[obj_key]['appearance']['shape_prob_labels'] = appearance_model.shape_labels()

        # color match
        objects_info[obj_key]['appearance']['color_hist'] = obj_clr_hist
        objects_info[obj_key]['appearance']['color_match_quotient'] = object_color_prob[base_color_id].item()
        objects_info[obj_key]['appearance']['color_prob'] = object_color_prob.cpu().numpy()
        objects_info[obj_key]['appearance']['color_prob_labels'] = appearance_model.color_labels()

        objects_info[obj_key]['appearance']['color_hist_quotient'] = cv2.compareHist(obj_clr_hist,
                                                                                     objects_info[obj_key][
                                                                                         'base_image']['histogram'],
                                                                                     cv2.HISTCMP_CORREL)

        # Todo: size match?

        # match

        if level == 'level1':
            shape_matches = current_object_shape_id in sorted(
                [(x, i) for i, x in enumerate(objects_info[obj_key]['appearance']['shape_prob'])], reverse=True)[:3]
            color_matches = current_object_color_id in sorted(
                [(x, i) for i, x in enumerate(objects_info[obj_key]['appearance']['color_prob'])], reverse=True)[:3]
        else:
            shape_matches = current_object_shape_id == base_shape_id
            color_matches = current_object_color_id == base_color_id

        objects_info[obj_key]['appearance']['match'] = shape_matches and color_matches

        if 'mismatch_count' not in objects_info[obj_key]['appearance']:
            objects_info[obj_key]['appearance']['mismatch_count'] = 0

        if objects_info[obj_key]['appearance']['match']:
            objects_info[obj_key]['appearance']['mismatch_count'] = 0
        else:
            objects_info[obj_key]['appearance']['mismatch_count'] += 1

    return objects_info


def process_video(video_data, appearance_model, save_path=None, save_mp4=False, save_gif=False, device='cpu'):
    appearance_model.eval()

    track_info = {}
    processed_frames = []
    violation = False
    for frame_num, frame in enumerate(video_data):
        track_info = track_objects(frame.obj_mask, track_info)
        track_info['objects'] = object_appearance_match(appearance_model, frame.image,
                                                        track_info['objects'], device)
        if save_gif:
            img = draw_bounding_boxes(frame.image, track_info['objects'])
            img = draw_appearance_bars(img, track_info['objects'])
            processed_frames.append(img)

        if 'objects' in track_info and len(track_info['objects'].keys()) > 0:
            for o in track_info['objects'].values():
                if not o['appearance']['match']:
                    # print('Appearance Mis-Match')
                    violation = True

    # save gif
    if save_gif:
        processed_frames[0].save(save_path + '.gif', save_all=True,
                                 append_images=processed_frames[1:], optimize=False, loop=1)

    # save video
    if save_gif and save_mp4:
        import moviepy.editor as mp
        clip = mp.VideoFileClip(save_path + '.gif')
        clip.write_videofile(save_path + '.mp4')
        os.remove(save_path + '.gif')

    return violation


def rgb_to_grayscale(img, num_output_channels: int = 1):
    if num_output_channels not in (1, 3):
        raise ValueError('num_output_channels should be either 1 or 3')

    r, g, b = img.unbind(dim=-3)
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-3)

    if num_output_channels == 3:
        return l_img.expand(img.shape)

    return l_img


# def rgb_to_grayscale(img, num_output_channels: int = 1):
#     if num_output_channels not in (1, 3):
#         raise ValueError('num_output_channels should be either 1 or 3')

#     r, g, b = img.unbind(dim=-3)
#     l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
#     l_img = l_img.unsqueeze(dim=-3)

#     if num_output_channels == 3:
#         return l_img.expand(img.shape)

#     return l_img


class ObjectDataset(Dataset):
    """ Dataset of objects with labels indictating their shape"""

    def __init__(self, data, transform=None):
        self.data = data
        self.shape_labels = sorted(set(data['shapes']))
        self.color_labels = sorted(set(np.array(data['textures']).squeeze(1)))

        self.data['shapes'] = np.array(self.data['shapes'])
        self.data['color'] = np.array(data['textures']).squeeze(1)

        for shape_id, shape in enumerate(self.shape_labels):
            self.data['shapes'][self.data['shapes'] == shape] = shape_id

        for color_id, color in enumerate(self.color_labels):
            self.data['color'][self.data['color'] == color] = color_id

        self.data['shapes'] = self.data['shapes'].astype(np.int)
        self.data['color'] = self.data['color'].astype(np.int)
        self.transform = transform

    def shape_label_name(self, label_id):
        return self.shape_labels[label_id]

    def shape_labels_count(self):
        _labels, counts = np.unique(self.data['shapes'], return_counts=True)
        return [counts[_labels == label_i][0] for label_i, _ in enumerate(self.shape_labels)]

    def color_label_name(self, label_id):
        return self.color_labels[label_id]

    def color_labels_count(self):
        _labels, counts = np.unique(self.data['color'], return_counts=True)
        return [counts[_labels == label_i][0] for label_i, _ in enumerate(self.shape_labels)]

    def get_dataset_weights(self, label_probs):
        weights = np.random.randn(self.__len__())
        for label_i, prob in enumerate(label_probs):
            weights[self.data['shapes'] == label_i] = prob
        return weights

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'images': self.data['images'][idx],
                  'shapes': self.data['shapes'][idx],
                  'color': self.data['color'][idx]}

        if self.transform:
            sample['gray_images'] = rgb_to_grayscale(torch.FloatTensor(sample['images']))

        return sample


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


def test_appearance_matching(dataloader, model):
    model.eval()
    batch_acc = {'shape': [], 'color': []}
    for i_batch, batch in enumerate(dataloader):
        object_image = batch['images']
        object_gray_image = batch['gray_images']
        object_shape = batch['shapes']
        object_color = batch['color']

        object_image = object_image.float()
        feature, shape_logits, color_logits = model(object_image, object_gray_image)

        shape_acc = sum(torch.argmax(shape_logits, dim=1) == object_shape).item() / len(object_shape)
        color_acc = sum(torch.argmax(color_logits, dim=1) == object_color).item() / len(object_color)

        batch_acc['shape'].append(shape_acc)
        batch_acc['color'].append(color_acc)

    print('Shape Accuracy:{} Color Acc: {}'.format(np.mean(batch_acc['shape']),
                                                   np.mean(batch_acc['color'])))


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
        batch_losses = {'shape': [], 'color': []}
        batch_acc = {'shape': [], 'color': []}
        for i_batch, batch in enumerate(dataloader):
            object_image = batch['images']
            object_gray_image = batch['gray_images']
            object_shape = batch['shapes']
            object_color = batch['color']

            object_image = object_image.float()
            feature, shape_logits, color_logits = model(object_image, object_gray_image)

            shape_loss = nn.CrossEntropyLoss()(shape_logits, object_shape)
            color_loss = nn.CrossEntropyLoss()(color_logits, object_color)
            shape_acc = sum(torch.argmax(shape_logits, dim=1) == object_shape).item() / len(object_shape)
            color_acc = sum(torch.argmax(color_logits, dim=1) == object_color).item() / len(object_color)
            batch_losses['shape'].append(shape_loss.item())
            batch_acc['shape'].append(shape_acc)
            batch_losses['color'].append(color_loss.item())
            batch_acc['color'].append(color_acc)

            # optimize
            optimizer.zero_grad()
            (shape_loss + color_loss).backward()
            optimizer.step()

        writer.add_scalar('train/shape_loss', np.mean(batch_losses['shape']), epoch)
        writer.add_scalar('train/shape_accuracy', np.mean(batch_acc['shape']), epoch)
        writer.add_scalar('train/color_loss', np.mean(batch_losses['color']), epoch)
        writer.add_scalar('train/color_accuracy', np.mean(batch_acc['color']), epoch)

        if (epoch % checkpoint_interval) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

            # save model separately as well
            torch.save(model.state_dict(), model_path)

        if epoch % log_interval == 0:
            print('Epoch: {} Shape Loss: {} Shape Accuracy:{} Color Acc: {}'.format(
                epoch, np.mean(batch_losses['shape']), np.mean(batch_acc['shape']),
                np.mean(batch_acc['color'])))


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
    parser.add_argument('--opr', choices=['generate_dataset', 'train', 'test', 'demo'], default='demo',
                        help='operation (opr) to be performed')

    return parser


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

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

        # create balanced distribution
        train_object_dataset = ObjectDataset(pickle.load(open(args.train_dataset_path, 'rb')),
                                             transform=transforms.Compose([transforms.Grayscale()]))
        probs = 1 / torch.Tensor(train_object_dataset.shape_labels_count())
        weights = train_object_dataset.get_dataset_weights(probs)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=True)
        dataloader = DataLoader(train_object_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1,
                                sampler=sampler)

        model = AppearanceMatchModel()
        optimizer = Adam(model.parameters(), lr=args.lr)

        # train
        train_appearance_matching(dataloader, model, optimizer, args.epochs, summary_writer,
                                  checkpoint_path, args.checkpoint_interval, args.log_interval)

    elif args.opr == 'test':
        train_object_dataset = ObjectDataset(pickle.load(open(args.train_dataset_path, 'rb')),
                                             transform=transforms.Compose([transforms.Grayscale()]))
        dataloader = DataLoader(train_object_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
        model = AppearanceMatchModel()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model = model.to(args.device)

        # train
        test_appearance_matching(dataloader, model)

    elif args.opr == 'demo':
        all_scenes = list(args.train_scenes_path.glob('*.pkl.gz'))
        print(f'Found {len(all_scenes)} scenes')

        model = AppearanceMatchModel()

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model = model.to(args.device)

        i = 0

        violations = []
        np.random.seed(0)
        np.random.shuffle(all_scenes)
        mis_match_cases = []

        for scene_file in all_scenes:
            with gzip.open(scene_file, 'rb') as fd:
                scene_data = pickle.load(fd)

            print(f'{i:} {scene_file.name}')
            v = process_video(scene_data, model, os.path.join(os.getcwd(), scene_file.name), save_mp4=True,
                              device=args.device, save_gif=True)
            if v:
                mis_match_cases.append(scene_file)
            violations.append(v)
        print(mis_match_cases)
        print((len(violations) - sum(violations)) / len(violations))
