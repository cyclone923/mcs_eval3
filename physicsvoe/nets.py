from . import depthutils

import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial, lru_cache

def next_dim(kernel, stride, in_):
    dilation = 1
    padding = 0
    return math.floor(1 + (in_ + 2*padding - dilation*(kernel-1) - 1)/stride)

class ThorNLLS:
    def __init__(self, oracle, model_cache=True):
        self.camera_info = {'vfov': 42.5,
                            'pos': [0, 1.5, -4.5]}
        self.oracle = oracle
        self.model_cache = dict() if model_cache else None
        self.forward = self.forward_oracle if oracle else self.forward_depth

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward_oracle(self, obj_pos, obj_ts, obj_ids, obj_mask, tgt_ts, tgt_ids):
        bs = obj_pos.size(0)
        pred = torch.zeros((bs, tgt_ids.size(1), 3), dtype=torch.float, device=tgt_ids.device)
        for b in range(bs):
            all_ids = [x for x in obj_ids[b].unique().tolist() if x != -1]
            for obj_id in all_ids:
                mask = (obj_ids[b] == obj_id)
                _p = obj_pos[b][mask]
                ts = obj_ts[b][mask]
                pred_mask = (tgt_ids[b] == obj_id)
                pred_ts = tgt_ts[b][pred_mask]
                pred_pos = self._pred(_p, ts, pred_ts)
                pred[b][pred_mask] = pred_pos
        return pred.detach()

    def _pred(self, pos, ts, pred_ts):
        ref_time = ts.min()
        floor = self.est_floor(pos)
        with torch.enable_grad():
            params = self.solve_params(pos, ts, floor, ref_time, 1e-5)
        pred_pos = self.model(*params, floor, ref_time, pred_ts)
        return pred_pos

    def forward_depth(self, depths, depth_ts, depth_ids, obj_mask, tgt_ts, tgt_ids):
        max_objs = tgt_ids.max()+1 #TODO: This is a hack
        all_ids = list(range(max_objs))
        obj_masks, obj_idx_ids = depthutils.separate_obj_masks(depth_ids, all_ids)
        pts_list = depthutils.project_points(depths, obj_masks, self.camera_info)
        bs = depths.size(0)
        pred = torch.zeros((bs, tgt_ids.size(1), 3), dtype=torch.float, device=tgt_ids.device)
        for obj_idx, (est_pts, est_mask) in enumerate(pts_list):
            for b in range(bs):
                obj_id = obj_idx_ids[obj_idx]
                _m = est_mask[b]
                if _m.sum() == 0:
                    continue
                _p = est_pts[b][_m]
                ts = depth_ts[b][_m]
                pred_mask = (tgt_ids[b] == obj_id)
                pred_ts = tgt_ts[b][pred_mask]
                pred_pos = self._pred(_p, ts, pred_ts)
                pred[b][pred_mask] = pred_pos
        return pred.detach()

    def solve_params(self, ps, ts, floor, ref_time, tol):
        if self.model_cache is not None:
            _h1 = hash(tuple(tuple(p) for p in ps.tolist()))
            _h2 = hash(tuple(ts.tolist()))
            _h = hash((_h1, _h2, floor, ref_time.item()))
            if _h in self.model_cache:
                return self.model_cache[_h]
        d = ps.device
        _, closest = torch.min((ts-ref_time).abs(), 0)
        p0 = ps[closest].clone().detach().requires_grad_()
        v0 = torch.zeros(3, requires_grad=True, device=d)
        a = torch.zeros(3, requires_grad=True, device=d)
        params = [p0, v0]
        opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)
        for _ in range(5000):
            errs = []
            preds = self.model(p0, v0, a, floor, ref_time, ts)
            opt.zero_grad()
            err = F.mse_loss(preds, ps)
            if err < tol:
                break
            err.backward()
            nn.utils.clip_grad_value_(params, 0.1)
            opt.step()
        if self.model_cache is not None:
            self.model_cache[_h] = (p0, v0, a)
        return p0, v0, a

    @staticmethod
    def model(p0, v0, a, floor, ref, t, leak=True):
        _t = (t-ref).unsqueeze(1)
        est = p0.unsqueeze(0) + _t*v0.unsqueeze(0) + (_t**2)*a.unsqueeze(0)
        for e in est:
            if e[1] < floor: e[1] = floor
        return est

    @staticmethod
    def est_floor(ps):
        BLIND_FLOOR = 0.25
        min_y = ps[:, 1].min()
        if min_y > 0.5:
            return BLIND_FLOOR
        diff = ps[:, 1]-min_y
        close = diff < 0.1
        return ps[:, 1][close].mean()

    def get_args(self, i):
        if self.oracle:
            return (i['obj_data'],
                    i['obj_ts'],
                    i['obj_ids'],
                    i['obj_mask'],
                    i['tgt_ts'],
                    i['tgt_ids']
                    )
        else:
            return (i['depths'],
                    i['depth_ts'],
                    i['depth_ids'],
                    i['depth_mask'],
                    i['tgt_ts'],
                    i['tgt_ids']
                    )
