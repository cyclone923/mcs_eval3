from .common import pad_tensors
from .types import ThorFrame

import pickle
import itertools
import random
import gzip
import numpy as np

from argparse import ArgumentParser
from pathlib import Path
from collections import namedtuple

import torch
from torch.utils.data import DataLoader, IterableDataset

DatasetEntry = namedtuple('DatasetEntry', ('file', 'input_idxs', 'target_idxs'))
class ThorDataset(IterableDataset):
    def __init__(self,
                 base,
                 max_input_frames=5,
                 target_dist='all',
                 target_type='flat',
                 shuffle=True,
                 random_origin=False,
                 filter=None):
        super().__init__()
        packs = Path(base).glob('*.pkl.gz')
        self.random_origin = random_origin
        assert target_dist in ('all', 'auto', 'all-drop', 'next')
        self.target_dist = target_dist
        assert target_type in ('masked', 'flat')
        self.target_type = target_type
        self.shuffle = shuffle
        self.max_input_frames = max_input_frames
        self.files = sorted(list(packs))
        if filter:
            self.files = [f for f in self.files if filter in f.name]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_files = list(self.files)
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            worker_files = self.files[worker_id::num_workers]
        return self._iter_from_files(worker_files)

    def _iter_from_files(self, files):
        init_load = 15
        files_iter = iter(files)
        entries = []
        has_files = True
        for _ in range(init_load):
            try:
              entries += self._load_file(next(files_iter))
            except StopIteration:
              has_files = False
              break
        if self.shuffle: random.shuffle(entries)
        while len(entries) > 0:
            yield entries.pop(0)
            if has_files and len(entries) == 0:
                for _ in range(init_load):
                    try:
                        entries += self._load_file(next(files_iter))
                    except StopIteration:
                        has_files = False
                        break
                if self.shuffle: random.shuffle(entries)

    def _load_file(self, path):
        raw_data = self._load_file_raw(path)
        if raw_data is None:
            return []
        scene_len = len(raw_data.objs)
        all_entries = []
        for ref_idx in range(scene_len):
            in_bounds = lambda x: 0 <= x < scene_len
            candidate_idxs = list(range(ref_idx-2*self.max_input_frames+1, ref_idx+1))
            input_idxs = random.sample(candidate_idxs, self.max_input_frames)
            target_idxs = self._get_target_idxs(ref_idx, scene_len, input_idxs)
            input_idxs = [x for x in sorted(input_idxs) if in_bounds(x)]
            target_idxs = [x for x in sorted(target_idxs) if in_bounds(x)]
            entry = self._make_entry(raw_data, ref_idx, input_idxs, target_idxs, self.target_type)
            if entry:
                if self.random_origin:
                    for _ in range(self.random_origin):
                        mod_entry = self._move_origin(entry)
                        all_entries.append(mod_entry)
                else:
                    all_entries.append(entry)
        return all_entries

    def _move_origin(self, entry):
        origin = (torch.rand((1, 3))-0.5)*20
        origin[0, 1] = 0
        new_entry = entry.copy()
        new_entry['obj_data'] = entry['obj_data']+origin
        new_entry['tgt_data'] = entry['tgt_data']+origin
        return new_entry

    def _get_target_idxs(self, ref_idx, scene_len, input_idxs):
        all_idxs = [x for x in range(scene_len)]
        if self.target_dist == 'all':
            target_idxs = all_idxs
        elif self.target_dist == 'all-drop':
            target_idxs = random.sample(all_idxs, len(all_idxs)//3)
            target_idxs = list(set(target_idxs+input_idxs))
        elif self.target_dist == 'auto':
            target_idxs = list(input_idxs)
        elif self.target_dist == 'next':
            target_count = self.max_input_frames
            target_idxs = list(range(ref_idx+1, ref_idx+1+target_count))
        return target_idxs

    @classmethod
    def _load_file_raw(cls, path):
        FrameData = namedtuple('FrameData', ('objs', 'scene_depth', 'scene_idxs', 'num_objs'))
        frame_list = cls._load_raw(path)
        if not cls._has_objects(frame_list):
            return None
        frame_list = cls._trim_ends(frame_list)
        id_to_idx = cls._calc_uuid_map(frame_list)
        obj_dicts = cls._calc_obj_dicts(frame_list, id_to_idx)
        #scene_pts = [frame.depth_pts for frame in frame_list]
        scene_depth = [frame.depth_mask for frame in frame_list]
        scene_idxs = [cls._translate_idx_mask(frame, id_to_idx) for frame in frame_list]
        return FrameData(obj_dicts, scene_depth, scene_idxs, len(id_to_idx))

    @staticmethod
    def _translate_idx_mask(frame, id_to_idx):
        mask = frame.obj_mask
        orig_mask = mask.copy()
        objs = frame.obj_data
        for mask_idx, obj in enumerate(objs):
            obj_id = id_to_idx[obj.uuid]
            if mask_idx != obj_id:
                mask[orig_mask==mask_idx] = obj_id
        return mask

    @staticmethod
    def _make_entry_objs(raw_data, ref_idx, input_idxs):
        in_objs = []
        in_scenes = []
        for in_idx in input_idxs:
            dt = in_idx - ref_idx
            # Obj data
            for obj_id, obj_data in raw_data.objs[in_idx].items():
                obj_pos = obj_data['pos']
                in_objs.append((dt, obj_id, obj_pos))
            # Scene points
            # pts = raw_data.scene_pts[in_idx]
            depth = raw_data.scene_depth[in_idx]
            px_ids = raw_data.scene_idxs[in_idx]
            in_scenes.append((dt, depth, px_ids))
        return in_objs, in_scenes

    @staticmethod
    def _make_entry_tgt(raw_data, ref_idx, target_idxs):
        tgt_objs = []
        for tgt_idx in target_idxs:
            dt = tgt_idx - ref_idx
            # Obj data
            for obj_id, obj_data in raw_data.objs[tgt_idx].items():
                obj_pos = obj_data['pos']
                tgt_objs.append((dt, obj_id, obj_pos, True))
        return tgt_objs

    @classmethod
    def _make_entry_masked_tgt(self, raw_data, ref_idx, target_idxs):
        tgt_ts = torch.tensor(target_idxs) - ref_idx
        tgt_ids = torch.arange(raw_data.num_objs)
        tgt_poss = torch.zeros((raw_data.num_objs, len(tgt_ts), 3), dtype=torch.float)
        tgt_mask = torch.zeros(tgt_poss.shape[:2], dtype=torch.bool)
        for t_idx, target_idx in enumerate(target_idxs):
            obj_data = raw_data.objs[target_idx]
            for obj_idx in range(raw_data.num_objs):
                if obj_idx in obj_data:
                    p = torch.tensor(obj_data[obj_idx]['pos'], dtype=torch.float)
                    tgt_poss[obj_idx, t_idx] = p
                    tgt_mask[obj_idx, t_idx] = True
        return tgt_ts, tgt_ids, tgt_poss, tgt_mask

    @classmethod
    def _make_entry(cls, raw_data, ref_idx, input_idxs, target_idxs, target_type='flat'):
        if len(input_idxs) == 0 or len(target_idxs) == 0:
            return None
        # Get input data for timesteps
        in_objs, in_scenes = cls._make_entry_objs(raw_data, ref_idx, input_idxs)
        if len(in_objs) == 0:
            return None
        obj_ts, obj_ids, obj_poss = [torch.tensor(x) for x in zip(*in_objs)]
        depth_ts_l, depths_l, depth_ids_l = zip(*in_scenes)
        depth_ts = torch.tensor(depth_ts_l)
        depths = torch.stack([torch.tensor(x) for x in depths_l])
        depth_ids = torch.stack([torch.tensor(x) for x in depth_ids_l])
        # Get target data for timesteps
        if target_type == 'flat':
            tgt_objs = cls._make_entry_tgt(raw_data, ref_idx, target_idxs)
            if len(in_objs) == 0:
                return None
            tgt_ts, tgt_ids, tgt_poss, tgt_mask = [torch.tensor(x) for x in zip(*tgt_objs)]
        elif target_type == 'masked':
            tgt_ts, tgt_ids, tgt_poss, tgt_mask = cls._make_entry_masked_tgt(raw_data, ref_idx, target_idxs)
            if tgt_mask.sum() == 0:
                return None
        # Sanity check
        # Collect into tensors
        result = {'obj_ts': obj_ts,
                  'obj_ids': obj_ids,
                  'obj_data': obj_poss,
                  'depths': depths,
                  'depth_ts': depth_ts,
                  'depth_ids': depth_ids,
                  'tgt_ts': tgt_ts,
                  'tgt_ids': tgt_ids,
                  'tgt_data': tgt_poss,
                  'tgt_mask': tgt_mask}
        return result

    @staticmethod
    def _load_raw(file):
        with gzip.open(file, 'rb') as fd:
            import sys
            print([x for x in sys.modules if 'data' in x])
            sys.modules['tpcthor'] = sys.modules['physicsvoe'] # HACK
            sys.modules['tpcthor.data.thor'] = sys.modules['physicsvoe.data'] # HACK
            frame_list = pickle.load(fd)
        return frame_list

    @staticmethod
    def _calc_obj_dicts(frames, id_dict):
        def calc_obj_dict(frame, id_dict):
            frame_dict = {}
            for o in frame.obj_data:
                idx = id_dict[o.uuid]
                pos = (o.position['x'], o.position['y'], o.position['z'])
                obj_dict = {'pos': pos}
                frame_dict[idx] = obj_dict
            return frame_dict
        dict_list = []
        for f in frames:
            frame_dict = calc_obj_dict(f, id_dict)
            dict_list.append(frame_dict)
        return dict_list

    @staticmethod
    def _calc_uuid_map(frames):
        id_to_idx = {}
        idx = 0
        for f in frames:
            for obj in f.obj_data:
                uuid = obj.uuid
                if uuid not in id_to_idx:
                    id_to_idx[uuid] = idx
                    idx += 1
        return id_to_idx

    @staticmethod
    def _has_objects(frames):
        for f in frames:
            if len(f.obj_data) != 0:
                return True
        return False

    @staticmethod
    def _trim_ends(frames):
        def _first_valid(fs):
            for i, f in enumerate(fs):
                if len(f.obj_data) != 0: return i
        first_valid = _first_valid(frames)
        last_valid = _first_valid(reversed(frames))
        _dl = frames[first_valid:]
        if last_valid != 0: _dl = _dl[:-last_valid]
        return _dl

def collate(scenes):
    keys = ('obj_ts', 'obj_ids', 'obj_data', 'depths', 'depth_ts', 'depth_ids', 'tgt_ts', 'tgt_ids', 'tgt_mask', 'tgt_data')
    obj_ts, obj_ids, obj_data, depths, depth_ts, depth_ids, tgt_ts, tgt_ids, tgt_mask, tgt_data = \
        [[s[key] for s in scenes] for key in keys]
    obj_masks = [torch.ones_like(i, dtype=torch.bool) for i in obj_ids]
    pad_obj_ts = pad_tensors(obj_ts, dims=[0])
    pad_obj_ids = pad_tensors(obj_ids, dims=[0], value=-1)
    pad_obj_data = pad_tensors(obj_data, dims=[0])
    pad_obj_mask = pad_tensors(obj_masks, dims=[0])
    pad_tgt_ts = pad_tensors(tgt_ts, dims=[0])
    pad_tgt_ids = pad_tensors(tgt_ids, dims=[0], value=-1)
    pad_dims = [0,1] if tgt_data[0].dim() == 3 else [0]
    pad_tgt_data = pad_tensors(tgt_data, dims=pad_dims)
    pad_tgt_mask = pad_tensors(tgt_mask, dims=pad_dims)
    pad_depths = pad_tensors(depths, dims=[0])
    pad_depth_ts = pad_tensors(depth_ts, dims=[0])
    pad_depth_ids = pad_tensors(depth_ids, dims=[0], value=-1)
    depth_mask = [torch.ones_like(i, dtype=torch.bool) for i in depth_ts]
    pad_depth_mask = pad_tensors(depth_mask, dims=[0])
    return {'obj_ts': pad_obj_ts.int(),
            'obj_ids': pad_obj_ids.int(),
            'obj_data': pad_obj_data,
            'obj_mask': pad_obj_mask,
            'tgt_ts': pad_tgt_ts.int(),
            'tgt_ids': pad_tgt_ids.int(),
            'tgt_data': pad_tgt_data,
            'tgt_mask': pad_tgt_mask,
            'depths': pad_depths.float(),
            'depth_ts': pad_depth_ts.int(),
            'depth_ids': pad_depth_ids,
            'depth_mask': pad_depth_mask
            }

def main(data):
    from torch.utils.data import DataLoader
    ds = ThorDataset(data)
    loader = DataLoader(ds, num_workers=8, batch_size=16, collate_fn=collate)
    for idx, item in enumerate(loader):
        print(idx)
        if idx > 12:
            break

def delete_bad(data):
    ds = ThorDataset(data)
    bad_files = []
    for path in ds.files:
        with gzip.open(path, 'rb') as fd:
            # Stupid hack
            import __main__
            setattr(__main__, ThorFrame.__name__, ThorFrame)
            frame_list = pickle.load(fd)
        try:
            frame_list = ThorDataset._trim_ends(frame_list)
        except:
            bad_files.append(path)
    print(f'Deleting {len(bad_files)} scenes...')
    for f in bad_files:
        f.unlink()

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('./data/thor/scenes'))
    return parser

if __name__=='__main__':
    args = make_parser().parse_args()
    main(args.data)
