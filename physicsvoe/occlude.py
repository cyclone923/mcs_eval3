from . import depthutils as du

import numpy as np
import torch
import torch.nn.functional as F

def detect_occlusions(depth, mask):
    all_ids = [x for x in np.unique(mask) if x != -1]
    return [detect_occlusion(depth, mask, i) for i in all_ids]

def detect_occlusion(depth, mask, id_):
    obj_mask = (mask == id_)
    if at_edge(obj_mask):
        return True
    if behind_object(obj_mask, depth, 1):
        return True
    return False

def at_edge(mask):
    edges = np.zeros_like(mask)
    edges[0, :] = True
    edges[:, 0] = True
    edges[-1, :] = True
    edges[:, -1] = True
    return (edges * mask).sum() > 0

def mask_around(mask):
    orig = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
    f = torch.tensor([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]], dtype=torch.float)
    f = f.unsqueeze(0).unsqueeze(0)
    explode = F.conv2d(orig.float(), f, padding=1).bool()
    edges = explode^orig
    return edges.squeeze(0).squeeze(0)

def behind_object(mask, depth, thresh):
    around = mask_around(mask)
    adj_depths = depth[around]
    obj_depths = depth[mask]
    adj_depth = np.percentile(adj_depths, 0.01)
    obj_depth = np.percentile(obj_depths, 0.1)
    print(adj_depth, obj_depth)
    return (adj_depth+thresh) < obj_depth
