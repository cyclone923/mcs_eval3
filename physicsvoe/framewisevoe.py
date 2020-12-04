from . import depthutils as du
from . import occlude
from .data.dataset import ThorDataset, collate
from .nets import ThorNLLS

from collections import defaultdict
from pathlib import Path
from argparse import ArgumentParser

from matplotlib import pyplot as plt
import torch
import numpy as np


class FramewiseVOE:
    def __init__(self, min_hist_count, max_hist_count, dist_thresh):
        self.frame_history = {}
        self.all_ids = set()
        self.dist_thresh = dist_thresh
        self.min_hist_count = min_hist_count
        self.max_hist_count = max_hist_count
        self.net = ThorNLLS(oracle=True)

    def record_obs(self, time, ids, pos, present, occluded):
        assert time not in self.frame_history
        self.frame_history[time] = (ids, pos, present, occluded)
        self.all_ids = self.all_ids.union(ids)

    def predict(self, time):
        in_ = self._get_inputs()
        if in_ is None:
            return None
        tgt = self._get_targets(time)
        pred = self.net(*in_, *tgt)
        pred_l = pred.squeeze(0)
        ids_l = tgt[1].squeeze(0).tolist()
        obj_ids = in_[2]
        mask_l = [i in obj_ids for i in ids_l]
        return ids_l, pred_l, mask_l

    def detect(self, time, actual_poss, occluded, actual_ids, depth, camera):
        violations = []
        pred_info = self.predict(time)
        if pred_info is None:
            return None
        pred_ids, pred_poss, pred_masks = pred_info
        for pred_pos, pred_id, pred_mask in zip(pred_poss, pred_ids, pred_masks):
            if not pred_mask:
                continue
            if pred_id in actual_ids:
                _idx = actual_ids.index(pred_id)
                if occluded[_idx]:
                    continue
                actual_pos = actual_poss[_idx]
                err = torch.dist(actual_pos, pred_pos)
                if err > self.dist_thresh:
                    v = PositionViolation(pred_id, pred_pos, actual_pos)
                    violations.append(v)
            else:
                # TODO: Check for occlusion
                v = PresenceViolation(pred_id, pred_pos, camera)
                violations.append(v)
        valid_violations = [v for v in violations if not v.ignore(depth, camera)]
        return valid_violations

    def _get_inputs(self):
        time_l = []
        id_l = []
        pos_l = []
        for time, frame_info in self.frame_history.items():
            for id_, pos, present, occluded in zip(*frame_info):
                if occluded or not present:
                    continue
                time_l.append(time)
                id_l.append(id_)
                pos_l.append(pos)
        time_l, id_l, pos_l = self._filter_inputs(time_l, id_l, pos_l)
        if len(time_l) == 0:
            return None
        obj_ts = torch.tensor(time_l).unsqueeze(0)
        obj_ids = torch.tensor(id_l).unsqueeze(0)
        obj_pos = torch.stack(pos_l).unsqueeze(0)
        obj_mask = torch.ones_like(obj_ts, dtype=torch.bool).unsqueeze(0)
        return obj_pos, obj_ts, obj_ids, obj_mask

    def _filter_inputs(self, time_l, id_l, pos_l):
        if not time_l:
            return time_l, id_l, pos_l
        obj_count = defaultdict(int)
        obj_valid = {i:(id_l.count(i) >= self.min_hist_count) for i in set(id_l)}
        comb = list(zip(time_l, id_l, pos_l))
        comb = sorted(comb, key=lambda x: x[0], reverse=True) #Latest timesteps first
        new_time_l = []
        new_id_l = []
        new_pos_l = []
        for t, i, p in comb:
            if obj_count[i] < self.max_hist_count and obj_valid[i]:
                new_time_l.append(t)
                new_id_l.append(i)
                new_pos_l.append(p)
                obj_count[i] += 1
        return new_time_l, new_id_l, new_pos_l

    def _get_targets(self, time):
        tgt_ids = torch.tensor(list(self.all_ids)).unsqueeze(0)
        tgt_ts = torch.tensor([time]*len(self.all_ids)).unsqueeze(0)
        return tgt_ts, tgt_ids

class NoViolation:
    pass

class PositionViolation:
    def __init__(self, object_id, pred_pos, actual_pos):
        self.object_id = object_id
        self.pred_pos = pred_pos
        self.actual_pos = actual_pos

    def fill_heatmap(self, hmap, obj_mask):
        return hmap + (obj_mask == self.object_id)

    def ignore(self, *_):
        return False

    def describe(self):
        return f'Object {self.object_id} is at {self.actual_pos}, but should be at {self.pred_pos}'

class PresenceViolation:
    def __init__(self, object_id, pred_pos, camera):
        self.object_id = object_id
        self.pred_pos = pred_pos
        self.radius = 5
        self._calc_mask(camera)

    def _calc_mask(self, camera):
        fpos = du.reverse_project(self.pred_pos, camera)
        rev_ratio = list(reversed(camera.aspect_ratio))
        pos = np.array(rev_ratio) * (0.5+fpos.numpy()/2)
        pxs = np.stack(np.meshgrid(*[np.arange(x) for x in rev_ratio], indexing='ij'),
                       axis=-1)
        dist = (pxs-pos)**2
        self.mask = dist.sum(-1) < self.radius**2

    def fill_heatmap(self, hmap, obj_mask):
        return hmap + self.mask

    def ignore(self, depth, camera):
        scene_depth = du.query_depth(depth, self.mask)
        pred_vec = self.pred_pos - torch.tensor(camera.position)
        pred_depth = pred_vec[2]
        is_occluded = scene_depth < pred_depth
        return is_occluded

    def describe(self):
        return f'Object {self.object_id} is not visible, but should be at {self.pred_pos}'

def make_voe_heatmap(viols, obj_mask):
    hmap = np.zeros_like(obj_mask, dtype=bool)
    viols = viols or []
    for v in viols:
        hmap = v.fill_heatmap(hmap, obj_mask)
    return hmap

def make_occ_heatmap(obj_occluded, obj_ids, obj_mask):
    hmap = np.zeros_like(obj_mask, dtype=bool)
    for occluded, idx in zip(obj_occluded, obj_ids):
        if not occluded:
            continue
        mask = obj_mask == idx
        hmap[mask] = True
    return hmap

def output_voe(viols):
    viols = viols or []
    for v in viols:
        print(v.describe())

def show_scene(scene_name, frame, depth, masks, hmap, omap=None):
    trip = np.repeat(depth[:, :, np.newaxis], axis=2, repeats=3)
    trip /= depth.max()
    trip[masks!=-1] = [0, 1, 0]
    if omap is not None:
        oidxs = np.nonzero(omap)
        trip[oidxs] = [1, 1, 0]
    hidxs = np.nonzero(hmap)
    trip[hidxs] = [1, 0, 0]
    plt.imshow(trip)
    plt.savefig(f'{scene_name}/{frame:02d}.png')
    plt.close('all')

###

DEFAULT_CAMERA = {'vfov': 42.5, 'pos': [0, 1.5, -4.5]}

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--path', type=Path, default=Path('./data/thor/scenes'))
    parser.add_argument('--filter', type=str, default=None)
    parser.add_argument('--exclude', type=str, default=None)
    return parser

def main(path, filter, exclude):
    # Load files
    scene_paths = find_scenes(path, filter, exclude)
    for path in scene_paths:
        data = ThorDataset._load_file_raw(path)
        full_voe(data)

def full_voe(data):
    voe = FramewiseVOE(min_hist_count=3, max_hist_count=8, dist_thresh=0.8)
    num_frames = len(data.objs)
    for frame_num in range(num_frames):
        print(f'Frame {frame_num}')
        # Get acutal obj positions
        depth = data.scene_depth[frame_num]
        masks = data.scene_idxs[frame_num]
        obj_occluded = occlude.detect_occlusions(depth, masks)
        obj_ids, obj_pos, obj_present = calc_world_pos(depth, masks, DEFAULT_CAMERA)
        # Infer positions from history
        viols = voe.detect(frame_num, obj_pos, obj_ids)
        voe_hmap = make_voe_heatmap(viols, masks)
        occ_hmap = make_occ_heatmap(obj_occluded, obj_ids, masks)
        output_voe(viols)
        show_scene(frame_num, depth, voe_hmap, occ_hmap)
        # Update tracker
        voe.record_obs(frame_num, obj_ids, obj_pos, obj_present)

def calc_world_pos(depth, mask, camera):
    mask = torch.tensor(mask)
    depth = torch.tensor(depth)
    obj_masks, all_ids = du.separate_obj_masks(mask)
    obj_pos, obj_present = du.project_points_frame(depth, obj_masks, camera)
    return all_ids, obj_pos, obj_present

def find_scenes(path, filter, exclude):
    if path.is_dir():
        apply_filters = lambda n: (not filter or filter in n) and (not exclude or exclude not in n)
        return [p for p in path.glob('*.pkl.gz') if apply_filters(p.name)]
    else:
        return [path]

if __name__=='__main__':
    args = make_parser().parse_args()
    main(args.path, args.filter, args.exclude)
