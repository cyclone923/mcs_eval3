from pathlib import Path
import gzip
import pickle
import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    data_paths = ['./data/thor/scenes', './data/thor/scenes/test']
    data_paths = [Path(x) for x in data_paths]
    files = find_files(data_paths)
    all_scenes = {}
    for f in files:
        pretty_name = f.name[:f.name.index('.')]
        print(pretty_name)
        entry = make_entry(f)
        #plot_entry(pretty_name, entry)
        all_scenes[pretty_name] = entry
    with open('scene_objs.pkl', 'wb') as fd:
        pickle.dump(all_scenes, fd)

def find_files(paths):
    all_files = []
    for p in paths:
        fs = p.glob('*.pkl.gz')
        #fs = [f for f in fs if 'aug' not in f.name]
        all_files += fs
    return all_files

def make_entry(path):
    with gzip.open(path, 'rb') as fd:
        data = pickle.load(fd)
    out_frames = []
    for raw_frame in data:
        obj_dict = {o.uuid:o.position for o in raw_frame.obj_data}
        out_frames.append(obj_dict)
    return out_frames

def plot_entry(name, frames):
    pos_tuple = lambda x: (x['x'], x['y'], x['z'])
    all_objs = set(itertools.chain(*[f.keys() for f in frames]))
    for i, uuid in enumerate(all_objs):
        fig = plt.figure(figsize=(5,5), dpi=100)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 5)
        ax.set_zlim(-5, 5)
        poss = [pos_tuple(f[uuid]) for f in frames if uuid in f]
        xs, ys, zs = zip(*poss)
        ax.scatter(xs, ys, zs, zdir='y')
        fig.savefig(f'{name}{i}.png')

if __name__ == '__main__':
    main()
