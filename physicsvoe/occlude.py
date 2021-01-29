from . import depthutils as du

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console

console = Console()

def detect_occlusions(depth, mask, all_ids, area_hists):
    return [detect_occlusion(depth, mask, i, area_hists[i]) for i in all_ids]

def detect_occlusion(depth, mask, id_, area_hist):
    obj_mask = (mask == id_)
    if obj_mask.sum() == 0:
        return False
    if at_edge(obj_mask):
        return True
    if smaller_area(area_hist):
        return True
    return False

def smaller_area(hist):
    recent = hist[-1]
    ref = np.median(hist[-5:-1])
    return recent < 0.8 * ref

def at_edge(mask):
    edges = np.zeros_like(mask)
    edges[0, :] = True
    edges[:, 0] = True
    edges[-1, :] = True
    edges[:, -1] = True
    console.print((edges * mask).sum())
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
