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
        # Maximum world-space position prediction error to tolerate before
        # raising a position VOE
        self.dist_thresh = dist_thresh
        # Minimum number of frames an object must be visible for before
        # it is elegible to raise violations
        self.min_hist_count = min_hist_count
        # Maximum number of recent history frames to store for use as input for
        # motion prediction model
        self.max_hist_count = max_hist_count
        # Motion prediction model -- 'oracle' here means it takes position
        # data as input rather than raw depth+object masks
        self.net = ThorNLLS(oracle=True)

    def record_obs(self, time, ids, pos, present, occluded, vis_count, pos_hists, camera_info):
        # Extract useful information from the model + simulator outputs,
        # store what we need to do the dynamics predictions.
        assert time not in self.frame_history
        # Just store the entire history in a dictionary, why not
        self.frame_history[time] = (ids, pos, present, occluded)
        # Determine which object IDs from the tracking model might actually
        # be real objects.
        # Objects are probably real if we've seen them for three prior frames
        # and they're currently visible without being occluded
        valid_ids = []
        for _id, _present, _occluded in zip(ids, present, occluded):
            probably_real = vis_count[_id] > 3
            if _present and not _occluded and probably_real:
                valid_ids.append(_id)
        # Which objects are newly considered 'real objects' this frame?
        # They need to be checked for entrance violations.
        new_ids = [i for i in valid_ids if i not in self.all_ids]
        viols = []
        for _id in new_ids:
            THRESH = 20
            # We don't care about where the object is now, we care about
            # the FIRST PLACE we saw it
            first_pos = pos_hists[_id][0]
            x, y = first_pos['x'], first_pos['y']
            ar = camera_info.aspect_ratio #`aspect ratio` actually gets the screen size in pixels
            # New objects should appear at the edge of the screen. If this
            # object was first seen too far from the edge, raise an entrance
            # violation.
            at_edge = (x < THRESH or x > ar[1]-THRESH) or (y < THRESH or y > ar[0]-THRESH)
            if not at_edge:
                viols.append(EntranceViolation(_id, first_pos))
        # Update the set of all object IDs with the set of valid object IDs
        # from this frame.
        self.all_ids = self.all_ids.union(valid_ids)
        return viols

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
        # Detect VOEs from prepared object position info + perception data
        # from models + simulator
        violations = []
        all_errs = []
        # Use the stored object position history to predict where each object
        # should be at this frame `time`
        pred_info = self.predict(time)
        if pred_info is None:
            # If we don't have enough data to make any reasonable predictions,
            # predict will return None.
            # In this case we have no clue where any object should be, so
            # we can't hope to detect any violations. Just give up early and
            # return.
            return None
        pred_ids, pred_poss, pred_masks = pred_info # Unpack function result
        for pred_pos, pred_id, pred_mask in zip(pred_poss, pred_ids, pred_masks):
            if not pred_mask:
                # We can't see the object -- `pred_mask` will be False for each
                # pixel and therefore evaluates to False.
                continue
            if pred_id in actual_ids:   # Object is a previously identified object
                _idx = actual_ids.index(pred_id)
                actual_pos = actual_poss[_idx] # Get actual world position for the object, determined from observed masks
                err = torch.dist(actual_pos, pred_pos)  # How far the object is from our predicted position
                all_errs.append(err)
                thresh = self.dist_thresh
                if occluded[_idx]: # If the object is occluded, then our position estimation is probably wrong
                    # Give way more wiggle room for object position violations
                    thresh *= 3
                if err > thresh:    # If the object is sufficiently far from our predicted position
                    v = PositionViolation(pred_id, pred_pos, actual_pos)
                    violations.append(v)
            else:
                # Our object prediction model thinks this object should exist,
                # but we don't 'see' any observed mask for it
                # Add an appearance violation at the object's predicted
                # location. We will ignore it later if the depth map suggests
                # the object is just occluded.
                v = PresenceViolation(pred_id, pred_pos, camera)
                violations.append(v)
        # Check if we should ignore any of the violations, and filter those out
        # (This is just for testing the presence violations)
        valid_violations = [v for v in violations if not v.ignore(depth, camera)]
        return valid_violations, all_errs

    def _get_inputs(self):
        # Translate the data in the object history dicts into tensor
        # representations for the dynamics model to consume
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

"""
Violation class:
    fill_heatmap
        Updates the provided screen-space violation heatmap with this
        violation's pixel-level spatial representation
    xy_pos
        Calculate the single screen-space point associated with this violation
    ignore
        Decide whether or not this violation is invalid and should be ignored.
    describe
        Return a string describing the violation, for debug output
"""

class NoViolation:
    pass

class PositionViolation:
    def __init__(self, object_id, pred_pos, actual_pos):
        self.object_id = object_id
        self.pred_pos = pred_pos
        self.actual_pos = actual_pos

    def fill_heatmap(self, hmap, obj_mask):
        # If we raised a position violation, we must be able to 'see' the
        # object in the object mask (otherwise we wouldn't have a position
        # to test our prediction against)
        return hmap + (obj_mask == self.object_id)

    def ignore(self, *_):
        # Position violations never need to be ignored.
        return False

    def xy_pos(self, camera):
        # Project the object's 3D position back to screen space,
        # return that point
        spos = du.reverse_project(self.actual_pos, camera).tolist()
        return {'x': spos[0], 'y': spos[1]}

    def describe(self):
        return f'Object {self.object_id} is at {self.actual_pos}, but should be at {self.pred_pos}'

class PresenceViolation:
    def __init__(self, object_id, pred_pos, camera):
        self.object_id = object_id
        self.pred_pos = pred_pos
        self.radius = 10
        self._calc_mask(camera)

    def _calc_mask(self, camera):
        # Create a screen-space 'mask' for the violation by filling in all the
        # pixels close enough to where we think the object SHOULD be
        spos = du.reverse_project(self.pred_pos, camera)
        rev_ratio = list(reversed(camera.aspect_ratio))
        pos = np.array(rev_ratio) * (0.5+spos.numpy()/2)
        pxs = np.stack(np.meshgrid(*[np.arange(x) for x in rev_ratio], indexing='ij'),
                       axis=-1)
        dist = (pxs-pos)**2
        self.mask = dist.sum(-1) < self.radius**2
        self.spos = spos.tolist()

    def fill_heatmap(self, hmap, obj_mask):
        return hmap + self.mask

    def ignore(self, depth, camera):
        # The presence violation should be ignored if our predicted object
        # location is mostly or completely off the screen.
        # We can test this by looking at how may screen pixels are lit up on
        # the violation's screen-space mask (from _calc_mask)
        # If not many pixels are present, it's at least partially ofscreen
        # and therefore we fully expect not to detect the object in the
        # simulator's output.
        if self.mask.sum() < 50: #Off screen!
            return True
        # ...it should also be ignored if, looking at the scene, we notice that
        # there is something potentially blocking our view of the object.
        # We can check for this by testing the depth map values of the
        # predicted violation screen pixels, and comparing their representative
        # depth value to the object's predicted depth
        # If we think the object is being hidden from view by scene geometry,
        # ignore the presence violation.
        scene_depth = du.query_depth(depth, self.mask)
        pred_vec = self.pred_pos - torch.tensor(camera.position)
        pred_depth = pred_vec[2]
        is_occluded = scene_depth < pred_depth
        return is_occluded

    def xy_pos(self, camera):
        return {'x': self.spos[0], 'y': self.spos[1]}

    def describe(self):
        return f'Object {self.object_id} is not visible, but should be at {self.pred_pos}'

class AppearanceViolation:
    def __init__(self, object_id, pos):
        self.object_id = object_id
        self.pos = pos

    def fill_heatmap(self, hmap, obj_mask):
        # Just highlight the object that looks weird.
        return hmap + (obj_mask == self.object_id)

    def xy_pos(self, camera):
        # Project the object's world location into screen space
        spos = du.reverse_project(self.pos, camera).tolist()
        return {'x': spos[0], 'y': spos[1]}

    def describe(self):
        return f'Object {self.object_id}\'s appearance changed'

class EntranceViolation:
    def __init__(self, object_id, entrance_pos):
        self.object_id = object_id
        self.pos = (entrance_pos['y'], entrance_pos['x'])

    def fill_heatmap(self, hmap, obj_mask):
        # Just highlight the object that we now realize entered weird.
        # TODO: Might want to update this to use self.pos?
        return hmap + (obj_mask == self.object_id)

    def ignore(self, *_):
        return False

    def xy_pos(self, camera):
        # XY position of the violation is just the place it 'entered' at.
        return {'x': self.pos[0], 'y': self.pos[1]}

    def describe(self):
        return f'Object {self.object_id} entered the scene in an unlikely location, {self.pos}'

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
