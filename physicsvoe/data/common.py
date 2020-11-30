import torch
import torch.nn.functional as F

import itertools

def _goal_sizes(ts, dims):
    return [max(t.size(d) for t in ts) for d in dims]

def _pad_tensor(t, dims, sizes, value):
    pads = [0]*t.dim()
    for i, s in zip(dims, sizes):
        pads[i] = s - t.size(i)
    pads = list(itertools.chain(*reversed([(0, p) for p in pads])))
    return F.pad(t, pads, value=value)

def pad_tensors(ts, dims=None, value=0):
    if dims is None:
        dims = list(range(ts[0].dim()))
    goal_size = _goal_sizes(ts, dims)
    pad_t = [_pad_tensor(t, dims, goal_size, value) for t in ts]
    return torch.stack(pad_t)

def stack_tensors(ts):
    return torch.stack(ts)

