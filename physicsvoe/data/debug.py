from pathlib import Path
from .dataset import ThorDataset, collate
import numpy as np
import pickle
import itertools
import gzip

from torch.utils.data import DataLoader

def main():
    path = Path('data/thor/scenes')
    debug_ds(path)


def debug_ds(path):
    ds = ThorDataset(path, target_type='masked', target_dist='auto')
    loader = DataLoader(ds, batch_size=4, collate_fn=collate)
    for item in loader:
        debug_item(item)


def debug_item(i):
    depth_ts = i['depth_ts']
    tgt_data = i['tgt_data']
    tgt_mask = i['tgt_mask']
    bs = depth_ts.size(0)
    timesteps = depth_ts.size(1)
    objs = tgt_data.size(1)
    for bid in range(bs):
        for oid in range(objs):
            time_pos = []
            for tid in range(timesteps):
                time = depth_ts[bid, tid]
                obj_pos = tgt_data[bid, oid, tid]
                mask = tgt_mask[bid, oid, tid]
                if mask:
                    time_pos.append((time.item(), obj_pos))
            vels = calc_vels(time_pos)
            if vels and max(vels) > 0.7:
                print(time_pos)
                import pdb; pdb.set_trace()  # XXX BREAKPOINT


def debug_item_flat(i):
    tgt_data = i['tgt_data']
    tgt_mask = i['tgt_mask']
    tgt_ids = i['tgt_ids']
    tgt_ts = i['tgt_ts']
    bs = tgt_data.size(0)
    for bid in range(bs):
        for obj in tgt_ids.unique():
            time_pos = []
            sel = (tgt_ids[bid] == obj)*tgt_mask[bid]
            ts = tgt_ts[bid][sel]
            poss = tgt_data[bid][sel]
            time_pos = []
            for t, pos in zip(ts, poss):
                time_pos.append((t.item(), pos))
            vels = calc_vels(time_pos)
            if vels and max(vels) > 0.7:
                print(time_pos)

def calc_vels(time_pos):
    vels = []
    prev = None
    time_pos = sorted(time_pos, key=lambda x: x[0])
    for tp in time_pos:
        if prev is not None:
            dt = tp[0] - prev[0]
            dp = tp[1] - prev[1]
            vel = np.linalg.norm(dp)/dt
            vels.append(vel)
        prev = tp
    return vels

def debug_file(path):
    with gzip.open(path, 'rb') as fd:
        data = pickle.load(fd)
    all_objs = calc_all_objs(data)
    max_delta = []
    for obj_id in all_objs:
        deltas = debug_obj(obj_id, data)
        max_delta.append(max(deltas))
    if max_delta:
        return max(max_delta)
    else:
        return None


def debug_obj(uuid, data):
    prev = None
    prev_t = 0
    deltas = []
    for fid, frame in enumerate(data):
        for obj in frame.obj_data:
            if obj.uuid != uuid:
                continue
            _p = obj.position
            pos = (_p['x'], _p['y'], _p['z'])
            pos = np.array(pos)
            if prev is not None:
                dp = np.linalg.norm(pos-prev)
                dt = fid-prev_t
                deltas.append(dp/dt)
            prev = pos
            prev_t = fid
    return deltas

def calc_all_objs(scene):
    _l = [[o.uuid for o in frame.obj_data] for frame in scene]
    return set(itertools.chain(*_l))

if __name__ == '__main__':
    main()
