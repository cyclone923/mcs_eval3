from . import depthutils as du

import numpy as np
import torch
import torch.nn.functional as F

def detect_occlusions(depth, mask, all_ids):
    return [detect_occlusion(depth, mask, i) for i in all_ids]

def detect_occlusion(depth, mask, id_):
    obj_mask = (mask == id_)
    if obj_mask.sum() == 0:
        return False
    if at_edge(obj_mask):
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
    obj_depths = depth[mask]
    obj_depth = np.percentile(obj_depths, 0.5)
    adj_depths = depth[around]
    adj_depth = np.percentile(adj_depths, 0.3)
    return obj_depth > adj_depth
