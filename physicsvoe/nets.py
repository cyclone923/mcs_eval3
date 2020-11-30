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

class _ThorBase:
    def make_conv(self, in_size, conv_info, out_size, img_size):
        self.depth_convs = nn.ModuleList()
        dims = list(img_size)
        first = True
        for kernel, next_size, stride in conv_info:
            layers = []
            if not first:
                layers.append(nn.BatchNorm2d(in_size))
            first = False
            layers.append(nn.Conv2d(in_size, next_size, kernel, stride))
            layers.append(nn.ReLU())
            block = nn.Sequential(*layers)
            self.depth_convs.append(block)
            dims = [next_dim(kernel, stride, x) for x in dims]
            in_size = next_size
        flat_size = dims[0]*dims[1]*in_size
        comp_layers = []
        comp_layers.append(nn.BatchNorm1d(flat_size))
        comp_layers.append(nn.Linear(flat_size, out_size))
        comp_layers.append(nn.ReLU())
        comp_layers.append(nn.BatchNorm1d(out_size))
        comp_layers.append(nn.Linear(out_size, out_size))
        self.depth_compress = nn.Sequential(*comp_layers)

    def make_obj_encoder(self, latent_sizes,
                         weight_hidden, c_mid, final_hidden,
                         combine_hidden, in_size, space_dim):
        self.time_convs = nn.ModuleList()
        self.combine_mlps = nn.ModuleList()
        default_args = {'weight_hidden': weight_hidden, 'c_mid': c_mid,
                        'final_hidden': final_hidden}
        for latent_sz in latent_sizes:
            args = dict(default_args)
            args.update({'neighbors': -1, 'c_in': in_size, 'c_out': latent_sz,
                         'dim': self.time_encode.out_dim+space_dim})
            pc = pointconv.PointConv(**args)
            self.time_convs.append(pc)
            mlp_args = {'in_size': in_size+latent_sz, 'out_size': latent_sz,
                        'hidden_sizes': combine_hidden, 'reduction': 'none',
                        'deepsets': False}
            pn = pointnet.SetTransform(**mlp_args)
            self.combine_mlps.append(pn)
            in_size = latent_sz

    def make_target_encoder(self, latent_sz, pred_latent):
        weight_hidden = [2**6]*4
        c_mid = 2**7
        final_hidden = [2**7]*4
        args = {'weight_hidden': weight_hidden, 'c_mid': c_mid,
                'final_hidden': final_hidden, 'neighbors': -1,
                'c_in': latent_sz, 'c_out': pred_latent,
                'dim': self.time_encode.out_dim}
        self.pred_pc = pointconv.PointConv(**args)

    def make_decoder(self, pred_latent, pred_hidden, out_size):
        mlp_args = {'in_size': pred_latent, 'out_size': out_size,
                    'hidden_sizes': pred_hidden, 'reduction': 'none',
                    'deepsets': False}
        self.predict_mlp = pointnet.SetTransform(**mlp_args)

    def depth_encode(self, depths, obj_masks_l):
        # Depths:
        #  Batch x Timestep x Height x Width
        # combined:
        #  Batch x Object x Timestep x Channels x Height x Width
        obj_masks = torch.stack(obj_masks_l, dim=1).float()
        exp_depth = depths.unsqueeze(1).unsqueeze(3).expand(-1, obj_masks.size(1), -1, -1, -1, -1)
        combined = torch.cat((exp_depth, obj_masks), dim=3)
        orig_size = combined.shape[:3]
        # comb_flat:
        #  (...) x Channels x Height x Width
        comb_flat = combined.view(-1, *combined.shape[3:])
        # Apply convolution
        x = comb_flat
        for depth_conv_block in self.depth_convs:
            x = depth_conv_block(x)
        # Flatten output
        out_flat = x.view(x.shape[0], -1)
        out_comp = self.depth_compress(out_flat)
        out = out_comp.view(*orig_size, out_comp.shape[-1])
        return out

    def flatten_info(self, objs, ts, mask, obj_ids):
        exp_ts = ts.unsqueeze(1).expand(-1, objs.size(1), -1)
        exp_mask = mask.unsqueeze(1).expand(-1, objs.size(1), -1)
        flat_ts = exp_ts.reshape(exp_ts.size(0), -1)
        flat_mask = exp_mask.reshape(exp_mask.size(0), -1)
        flat_objs = objs.view(objs.size(0), -1, objs.size(-1))
        exp_ids = torch.tensor(obj_ids).to(objs.device).view(1, -1, 1).expand(objs.size(0), -1, objs.size(2))
        flat_ids = exp_ids.reshape(exp_ids.size(0), -1)
        return flat_objs, flat_ts, flat_mask, flat_ids

    def obj_encode(self, flat_objs, flat_ts, flat_mask, flat_ids):
        in_feats = flat_objs
        time_dist_fn = partial(time_dist, flat_ids, flat_mask, flat_objs, self.time_encode)
        for time_pc, combine_mlp in zip(self.time_convs, self.combine_mlps):
            time_nei = time_pc(flat_ts, flat_ts, in_feats, time_dist_fn)
            combine_in = torch.cat([in_feats, time_nei], dim=-1)
            next_feats = combine_mlp(combine_in)
            in_feats = next_feats
        return next_feats

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
        params = [p0, v0, a]
        opt = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)
        for _ in range(100):
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
