from argparse import ArgumentParser
import pickle
import gzip
from pathlib import Path
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
import matplotlib.pyplot as plt
from rich.console import Console

console = Console()

class ObjectFeatures():
    def __init__(self, kp, des):
        self.keypoints = kp
        self.descriptors = des

class ObjectDataset():
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

class OpenCVModel():
    def __init__(self):
        self.OPENCV_OBJECT_TRACKERS = {
            'csrt': cv2.TrackerCSRT_create(),
            'kcf': cv2.TrackerKCF_create(),
            'mil': cv2.TrackerMIL_create(),
            'tld': cv2.TrackerTLD_create(),
            'goturn': cv2.TrackerGOTURN_create()
        }

    def initNewTracker(self, frame, bbox, type='tld'):
        tracker = self.OPENCV_OBJECT_TRACKERS[type]
        tracker.init(frame, bbox)
        return tracker

    def match(self, frame, objects_info, device='cpu', level='level2'):
        base_image = np.array(frame)
        (H, W) = base_image.shape[:2]
        for key, obj in objects_info.items():

            if 'appearance' not in obj.keys():
                obj['appearance'] = dict()
                obj['appearance']['match'] = True

            # if not obj['visible']:
            #     continue

            if obj['occluded']:
                continue

            top_x, top_y, bottom_x, bottom_y = obj['bounding_box']
            obj_current_image = frame.crop((top_x, top_y, bottom_x, bottom_y))
            init_bb = (top_y, top_x, bottom_y - top_y, bottom_x - top_x)
            console.log(init_bb)

            image_area = np.prod(obj_current_image.size)
            mask_image = np.zeros(obj['mask'].shape, dtype=base_image.dtype)
            mask_image[obj['mask']] = 255

            # opencv_image = cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR)

            if 'base_image' not in obj.keys():
                obj['base_image'] = dict()
                obj['tracker'] = self.initNewTracker(base_image, init_bb, type='tld')
                console.log(obj['tracker'])

            ok, curr_bb = obj['tracker'].update(base_image)

            if ok:
                (x, y, w, h) = [int(v) for v in curr_bb]
                console.log((x, y, w, h))
                cv2.rectangle(base_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                info = [
                    ("Object " + str(key) + " Success", "Yes" if ok else "No")
                ]
                for (k, v) in info:
                    text = "{}: {}".format(k, v)
                    cv2.putText(base_image, text, (10, H - ((key * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if not obj['occluded']:
                obj['appearance']['match'] = ok

                if 'mismatch_count' not in obj['appearance'].keys():
                    obj['appearance']['mismatch_count'] = 0

                if obj['appearance']['match']:
                    obj['appearance']['mismatch_count'] = 0
                else:
                    obj['appearance']['mismatch_count'] += 1
        
        i = len(objects_info.keys())
        text="Tracker: TLD"
        cv2.putText(base_image, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Frame", base_image)
        key = cv2.waitKey(1) & 0xFF


        return objects_info

class SIFTModel():
    def __init__(self):
        self.feature_match_slack = 0.5          # The amount of match rate to allow without declaring a mismatch 
        self.obj_dictionary = dict()            # The learned dictionary of objects and their descriptors
        self.detector = cv2.SIFT_create()       # The SIFT feature detection object
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params).knnMatch       # FLANN based matcher

    def eval(self):
        obj_dictionary_pth = 'tracker/siftModel_t.p'
        self.obj_dictionary = pickle.load( open(obj_dictionary_pth, 'rb') )

    # Search the learned object space to identify what the initial object is
    # def identifyInitialObject(self, img_kp, img_des):
    #     # NOTE: If object descriptors do not sufficiently match up with any other object, return None
    #     # (system will fall back to frame-by-frame matching)
    #     match_avgs = dict()
    #     for obj_id, obj in self.obj_dictionary.items():
    #         o_match_rates = list()
    #         for o_img in range(0, len(obj['descriptors'])):
    #             o_kp = obj['keypoints'][o_img]
    #             o_des = obj['descriptors'][o_img]
    #             o_matches = self.matcher(img_des, o_des, k=2)
                
    #             o_good = list()
    #             for m, n in o_matches:
    #                 if m.distance < 0.7 * n.distance:
    #                     o_good.append([m])
    #             o_match_rates.append(len(o_good) / len(o_matches))

    #         o_match_rates = np.array(o_match_rates)
    #         o_match_avg = np.mean(o_match_rates)
    #         match_avgs[obj_id] = o_match_avg
    #     max_o = max(match_avgs, key=lambda o: match_avgs[o])
    #     console.log('best matching object:', max_o)
    #     console.log('match rate:', match_avgs[max_o])
    #     return max_o if match_avgs[max_o] >= 1 - self.feature_match_slack else None

    # Match feature descriptors
    def detectFeatureMatch(self, obj_des, img_des):
        matches = self.matcher(obj_des, img_des, k=2)
        good = list()
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append([m])
        return len(good) >= 1 - self.feature_match_slack
        # if obj['base_image']['shape_id'] is None:   # modeler was unable to match seen object with any learned object
        #     match = self.frameMatch(img_kp, img_des, obj)  # fall back to frame-by-frame feature matching
        # else:
        #     shape_id = obj['base_image']['shape_id']
        #     # feature match
        #     l_match_rates = list()
        #     for l_obj_img_des in self.obj_dictionary[shape_id]['descriptors']:
        #         l_matches = self.matcher(img_des, l_obj_img_des, k=2)

        #         l_good = list()
        #         for m, n in l_matches:
        #             if m.distance < 0.7 * n.distance:
        #                 l_good.append([m])
        #         l_match_rates.append(len(l_good) / len(l_matches))
            
        #     l_match_rates = np.array(l_match_rates)
        #     avg_match_rate = np.mean(l_match_rates)
        #     console.log('average match rate:', avg_match_rate)
        #     if avg_match_rate >= 1 - self.feature_match_slack:
        #         match = True
        #     else:
        #         match = self.frameMatch(img_kp, img_des, obj)  # fall back to frame-by-frame feature matching
            
        # return obj, match

    # Determine feature match with the state of the object in the previous frame
    # def frameMatch(self, img_kp, img_des, obj):
    #     try:
    #         prev_kp = obj['appearance']['feature_history']['keypoints'][-1]
    #         prev_des = obj['appearance']['feature_history']['descriptors'][-1]
    #     except KeyError:
    #         return True
    #     except IndexError:
    #         return True
    #     f_matches = self.matcher(img_des, prev_des, k=2) # k=2 so we can apply the ratio test next
    #     f_good = list()
    #     for m, n in f_matches:
    #         if m.distance < 0.7 * n.distance:
    #             f_good.append([m])

    #     match_rate = len(f_good) / len(f_matches)
    #     console.log('frame-by-frame match rate:', match_rate)
        
    #     return match_rate >= 1 - self.feature_match_slack
    
    def match(self, image, objects_info, device='cpu', level='level2'):
        for key, obj in objects_info.items():
            if not obj['visible']:  # if the object is not visible, don't check for an appearance match; just go with the last match decision
                continue
            if 'appearance' not in obj.keys():
                obj['appearance'] = dict()
                # obj['appearance']['feature_history'] = dict()
                # obj['appearance']['feature_history']['keypoints'] = list()
                # obj['appearance']['feature_history']['descriptors'] = list()
                obj['appearance']['match'] = True
                obj['appearance']['mismatch_count'] = 0

                top_x, top_y, bottom_x, bottom_y = obj['bounding_box']
                obj_current_image = image.crop((top_y, top_x, bottom_y, bottom_x))
                base_obj_image = np.array(obj_current_image)

                obj_kp, obj_des = self.detector.detectAndCompute(base_obj_image, None)

                obj['appearance']['base_image'] = dict()
                obj['appearance']['base_image']['keypoints'] = obj_kp
                obj['appearance']['base_image']['descriptors'] = obj_des
                
                continue

            if obj['occluded']:
                continue
            
            # top_x, top_y, bottom_x, bottom_y = obj['bounding_box']
            # obj_current_image = image.crop((top_y, top_x, bottom_y, bottom_x))

            # image_area = np.prod(obj_current_image.size)
            base_image = np.array(image)
            # mask_image = np.zeros(obj['mask'].shape, dtype=base_image.dtype)
            # mask_image[obj['mask']] = 255

            console.log('creating base image for object with ID', key, '...')

            # obj_clr_hist_0 = cv2.calcHist([np.array(image)], [0], mask_image, [10], [0, 256])
            # obj_clr_hist_1 = cv2.calcHist([np.array(image)], [1], mask_image, [10], [0, 256])
            # obj_clr_hist_2 = cv2.calcHist([np.array(image)], [2], mask_image, [10], [0, 256])
            # obj_clr_hist = (obj_clr_hist_0 + obj_clr_hist_1 + obj_clr_hist_2) / 3

            # run SIFT on image
            img_kp, img_des = self.detector.detectAndCompute(base_image, None)

            # if 'base_image' not in obj.keys():
            #     obj['base_image'] = dict()
            #     obj['base_image']['shape_id'] = self.identifyInitialObject(img_kp, img_des)
                # obj['base_image']['histogram'] = obj_clr_hist

            # Run detectFeatureMatch
            feature_match = self.detectFeatureMatch(obj['appearance']['base_image']['descriptors'], img_des)

            # Update feature match indicator if the object is not occluded
            # if not obj['occluded']:
                # obj['appearance']['feature_history']['keypoints'].append(img_kp)
                # obj['appearance']['feature_history']['descriptors'].append(img_des)
            obj['appearance']['match'] = feature_match

            if obj['appearance']['match']:
                obj['appearance']['mismatch_count'] = 0
            else:
                obj['appearance']['mismatch_count'] += 1

        return objects_info

class AppearanceMatchModel():
    def __init__(self, modeler):
        if modeler == 'sift':
            self.modeler = SIFTModel()
        else:
            self.modeler = OpenCVModel()  # TEMP: Replace with KCF/CSRT here when implemented

    # Check for any appearance mismatches in the provided images
    def appearanceMatch(self, image, objects_info, device='cpu', level='level2'):
        return self.modeler.match(image, objects_info, device, level)


    def to_png(self, images):
        print("generating images")
        for i in tqdm(range(len(images))):
            img = images[i].reshape(50,50,3)
            plt.imshow(img)
            plt.savefig('allTrainImages/trImg_'+str(i))

        return os.listdir('allTrainImages/')

    def train(self, scenes, model_path, checkpoint_path, restore_checkpoint):
        print('entered training mode')

        if restore_checkpoint:
            checkpt = pickle.load(open(checkpoint_path, 'rb'))
            image_idx = checkpt['image idx']
            self.obj_dictionary = pickle.load(open(model_path, 'rb'))
            
        else:
            image_idx = 0
        scene_data = scenes.data   
        train_imgs = []
        shapes = scenes.shape_labels
        colors = scenes.color_labels
        self.obj_dictionary = pickle.load(open(model_path, 'rb'))
        
        # uncomment the below code to generate .png files since cv2.SIFT works only on jpg/png formatted files
        #follow below code only if the dataset is not extremely large. If testing on eentire dataset, consider using the code below for-loop

        # allTrainImages = self.to_png(scene_data['images'])
        # allTrainImages = os.listdir('trainTrial/')
        # # print(len(allTrainImages))
        # allTrainImages = sorted(allTrainImages, key=lambda x: int(x.partition('_')[2].partition('.')[0]))
        
        
        # for k in self.obj_dictionary.keys():
        #     print(k, len(self.obj_dictionary[k]['descriptors']), len(self.obj_dictionary[k]['keypoints']))
        
        # print(sum([len(self.obj_dictionary[k]['descriptors']) for k in self.obj_dictionary.keys()]))
        
        # for i in range(len(allTrainImages)):
        #     train_imgs.append((allTrainImages[i], shapes[scene_data['shapes'][i]], colors[scene_data['color'][i]]))
        
        for i in range(len(scene_data['images'])):
            train_imgs.append((scene_data['images'][i], shapes[scene_data['shapes'][i]], colors[scene_data['color'][i]]))
        # print(train_imgs[3960])
        for i,(img, shape, color) in tqdm(enumerate(train_imgs[3959:])):  

            img = img.reshape(50,50,3)
            plt.imshow(img)
            plt.savefig('trainTrial/trImg_'+str(i+3959))   
            if shape not in self.obj_dictionary:
                self.obj_dictionary[shape] = dict()
                self.obj_dictionary[shape]['color'] = [color]
                self.obj_dictionary[shape]['keypoints'] = list()
                self.obj_dictionary[shape]['descriptors'] = list()
                
                # path of the image directory + image number whose features to be detected
                trimg = cv2.imread('trainTrial/trImg_'+str(i)+'.png')
                # trimg = cv2.imread('trainTrial/'+str(img))
                k,d = self.modeler.detector.detectAndCompute(trimg, None)
                kpts = [p.pt for p in k]
                self.obj_dictionary[shape]['keypoints'].append(kpts)
                self.obj_dictionary[shape]['descriptors'].append(d)
                
            else:
                self.obj_dictionary[shape]['color'].append(color)
                trimg = cv2.imread('trainTrial/trImg_'+str(i+3959)+'.png')
                # trimg = cv2.imread('trainTrial/'+str(img))
                k,d = self.modeler.detector.detectAndCompute(trimg, None)
                kpts = [p.pt for p in k]
                self.obj_dictionary[shape]['keypoints'].append(kpts)
                self.obj_dictionary[shape]['descriptors'].append(d)
                # print(k.pt)
            
            #for some reason checkpt isn't storing the indices so manually added here:/
            if (i+3959)%1000==0:
                modelDict = open(model_path, 'wb')
                pickle.dump(self.obj_dictionary, modelDict)

                checkpoint = open(checkpoint_path,'wb')
                pickle.dump({'image idx': i+3959, 'model': self.obj_dictionary}, checkpoint)
                print("saved checkpoint and model at image {}".format(i+3959))

        print(self.obj_dictionary.keys())
        modelDict = open(model_path, 'wb')
        pickle.dump(self.obj_dictionary, modelDict)


    def test(self, dataloader):     # Tests how good the robot was at building the object dictionary
        # batch_acc = { 'shape': list(), 'color': list() }
        batch_acc = list()
        for i, batch in enumerate(dataloader):  # for each object
            object_image = batch['images']
            object_gray_image = batch['gray_images']
            object_shape = batch['shapes']
            object_color = batch['color']
            
            object_image_kp, object_image_des = self.detector.detectAndCompute(object_image, None)
            
            l_match_rates = list()
            for l_obj_img in self.obj_dictionary[object_shape]:
                l_matches = self.modeler.matcher(object_image_des, l_obj_img.descriptors, k=2)

                l_good = list()
                for m, n in l_matches:
                    if m.distance < 0.7 * n.distance:
                        l_good.append([m])
                l_match_rates.append(len(l_good) / len(l_matches))
            
            l_match_rates = np.array(l_match_rates)
            avg_match_rate = np.mean(l_match_rates)
            batch_acc.append(avg_match_rate)
        
        print('Accuracy:', np.mean(batch_acc))
            
        pass

    def process_video(self, video_data, save_path=None, save_mp4=False, save_gif=False, device='cpu'):
        track_info = dict()
        processed_frames = list()
        violation = False
        for frame_num, frame in enumerate(video_data):
            track_info = track_objects(frame.obj_mask, track_info)
            track_info['objects'] = self.appearanceMatch(frame.image, track_info['objects'], device)

            if save_gif:
                img = draw_bounding_boxes(frame.image, track_info['objects'])
                img = draw_appearance_bars(img, track_info['objects'])
                processed_frames.append(img)
            
            if 'objects' in track_info and len(track_info['objects'].keys()) > 0:
                for o in track_info['objects'].values():
                    if not o['appearance']['match']:
                        # print('Appearance Mismatch')
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

# Convert the image to a Tensor
def obj_image_to_tensor(obj_image, gray=False):
    obj_image = obj_image.resize((50, 50))
    obj_image = np.array(obj_image)
    obj_image = obj_image.reshape((3, 50, 50))
    obj_image = torch.Tensor(obj_image).float()
    if gray:
        obj_image = rgb_to_grayscale(obj_image)
    
    return obj_image

# Convert a color image to grayscale
def rgb_to_grayscale(img, num_output_channels: int = 1):
    if num_output_channels not in (1, 3):
        raise ValueError('num_output_channels should be either 1 or 3')

    r, g, b = img.unbind(dim=-3)
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-3)

    if num_output_channels == 3:
        return l_img.expand(img.shape)

    return l_img

def closest_colour(requested_colour):
    min_colours = dict()
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
    parser.add_argument('--opr', choices=['generate_dataset', 'train', 'test', 'demo'], default='demo',
                        help='operation (opr) to be performed')

    return parser

if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    args = make_parser().parse_args()
    args.device = 'cuda' if torch.cuda_is_available() else 'cpu'

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
        train_object_dataset = ObjectDataset(pickle.load(open(args.train_dataset_path, 'rb')))
        # dataloader = DataLoader(train_object_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
        model = AppearanceMatchModel('sift')

        model.train(train_object_dataset, 'siftModel_t.p','checkpoint_pth.p',False)

    elif args.opr == 'test':
        train_object_dataset = ObjectDataset(pickle.load(open(args.train_dataset_path, 'rb')),
                                             transform=transforms.Compose([transforms.Grayscale()]))
        dataloader = DataLoader(train_object_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
        model = pickle.load(model_path)

        model.test(dataloader)

    elif args.opr == 'demo':
        all_scenes = list(args.train_scenes_path.glob('*.pkl.gz'))
        print(f'Found {len(all_scenes)} scenes')

        model = pickle.load(model_path)

        violations = list()
        np.random.seed(0)
        np.random.shuffle(all_scenes)
        mismatch_cases = list()

        for idx, scene_file in enumerate(all_scenes):
            with gzip.open(scene_file, 'rb') as fd:
                scene_data = pickle.load(fd)
            
            print(f'{idx:} {scene_file.name}')
            v = model.process_video(scene_data, os.path.join(os.getcwd(), scene_file.name), save_mp4=True, device=args.device, save_gif=True)

            if v:
                mismatch_cases.append(scene_file)
            violations.append(v)
        print(mismatch_cases)
        print((len(violations) - sum(violations)) / len(violations))





