import torch
import numpy as np
import torch.nn.functional as F

import functools
import math

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def separate_obj_masks(mask, all_ids=None):
    if all_ids is None:
        all_ids = [x.item() for x in mask.unique() if x != -1]
    obj_masks = [mask == obj_id for obj_id in all_ids]
    return obj_masks, all_ids

@functools.lru_cache(None)
def circle_mask(hsize):
    x, y = np.ogrid[-hsize:hsize+1, -hsize:hsize+1]
    mask = x**2+y**2 <= hsize**2
    m = torch.tensor(mask).float()
    m = m.view(1, 1, *m.size())
    return m

def explode_masks(mask, count):
    half_size = 8
    f = circle_mask(half_size).to(mask.device)
    return _apply_explode_masks(mask, half_size, f, count)

def _apply_explode_masks(mask, half_size, f, count):
    flat_mask = mask.view(-1, mask.size(-2), mask.size(-1))
    flat_mask = flat_mask.unsqueeze(1)
    conv_results = []
    x = flat_mask.float()
    for _ in torch.arange(count):
        next_mask = x.bool().view(*mask.shape[:2], 1, *x.shape[-2:])
        conv_results.append(next_mask)
        x = F.conv2d(x, weight=f, padding=half_size)
    tensor_results = torch.cat(conv_results, dim=2)
    return tensor_results

def test_plot(depth, mask, x, oid, i, j):
    fig = plt.figure(figsize=(5,15), dpi=100)
    ax = fig.add_subplot(3, 1, 1, projection='3d')
    x = x.cpu()
    ax.scatter(*zip(*x))
    ax = fig.add_subplot(3, 1, 2)
    ax.imshow(depth.cpu())
    ax = fig.add_subplot(3, 1, 3)
    ax.imshow(mask.cpu())
    plt.show()
    plt.savefig(f'{oid:02d}_{i:02d}_{j:02d}_out.png')

def project_points(depth, masks, camera_info):
    results = []
    bs = depth.size(0)
    ts = depth.size(1)
    obj_count = len(masks)
    height, width = depth.shape[-2:]
    cy, cx = height/2, width/2
    vfov = camera_info['vfov'] * (math.pi/180)
    hfov = 2*math.atan(math.tan(vfov/2) * width/height)
    tans = torch.tensor([np.tan(fov/2) for fov in (vfov, hfov)], device=depth.device)
    cam_pos = torch.tensor(camera_info['pos'], device=depth.device)
    for oid, obj_mask in enumerate(masks):
        nz_tuple = obj_mask.nonzero(as_tuple=True)
        depths = depth[nz_tuple]
        nz = obj_mask.nonzero(as_tuple=False).float()
        batch_time = nz[:, 0:2]
        yx = nz[:, 2:]
        yx[:, 0] -= cy
        yx[:, 0] /= -height/2
        yx[:, 1] -= cx
        yx[:, 1] /= width/2
        yx *= tans
        xy = yx[:, [1, 0]]
        xyz = torch.cat((xy, torch.ones((xy.size(0), 1), device=xy.device)), dim=1)
        xyz *= depths.unsqueeze(1)
        xyz += cam_pos
        pt_pos = torch.zeros((bs, ts, 3), device=depth.device, dtype=torch.float)
        out_mask = torch.zeros((bs, ts), device=depth.device, dtype=torch.bool)
        for b in range(bs):
            for t in range(ts):
                sel_idx = (batch_time[:,0]==b)*(batch_time[:,1]==t)
                sel_pts = xyz[sel_idx]
                if len(sel_pts) > 0:
                    mean = sel_pts.mean(0)
                    pt_pos[b, t] = mean
                    out_mask[b, t] = True
        results.append((pt_pos, out_mask))
    return results

def project_points_frame(depth, masks, camera_info):
    results = []
    obj_count = len(masks)
    height, width = depth.shape[-2:]
    cy, cx = height/2, width/2
    vfov = camera_info['vfov'] * (math.pi/180)
    hfov = 2*math.atan(math.tan(vfov/2) * width/height)
    tans = torch.tensor([np.tan(fov/2) for fov in (vfov, hfov)], device=depth.device)
    cam_pos = torch.tensor(camera_info['pos'], device=depth.device)
    obj_poses = []
    obj_present = []
    for obj_mask in masks:
        nz_tuple = obj_mask.nonzero(as_tuple=True)
        depths = depth[nz_tuple]
        yx = obj_mask.nonzero(as_tuple=False).float()
        yx[:, 0] -= cy
        yx[:, 0] /= -height/2
        yx[:, 1] -= cx
        yx[:, 1] /= width/2
        yx *= tans
        xy = yx[:, [1, 0]]
        xyz = torch.cat((xy, torch.ones((xy.size(0), 1), device=xy.device)), dim=1)
        xyz *= depths.unsqueeze(1)
        xyz += cam_pos
        if len(xyz) > 0:
            obj_poses.append(xyz.mean(0))
            obj_present.append(True)
        else:
            obj_poses.append(torch.zeros(3, device=xyz.device, dtype=torch.float))
            obj_present.append(False)
    return obj_poses, obj_present

def reverse_project(world_pos, out_mask, camera_info):
    rel_v = world_pos - torch.tensor(camera_info['pos'])
    width, height = out_mask.shape
    vfov = camera_info['vfov'] * (math.pi/180)
    hfov = 2*math.atan(math.tan(vfov/2) * width/height)
    return torch.tensor([0, 0])



