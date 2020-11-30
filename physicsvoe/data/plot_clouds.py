import pickle
import itertools
from pathlib import Path
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--pack', type=Path, default=Path('test.pkl'))
    return parser

def plot_pts(ax, pts):
    ax.scatter(*zip(*pts), zdir='y', s=0.1, alpha=0.2)

def plot_objs(ax, objs):
    xf = lambda x: (x['x'], x['y'], x['z'])
    pos = [xf(o.position) for o in objs]
    ax.scatter(*zip(*pos), c='red', s=100, zdir='y', marker='x')

def plot_scene(i, pts, objs):
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot_pts(ax, pts)
    plot_objs(ax, objs)
    bound = 6
    ax.set_xlim([-bound/2, bound/2])
    ax.set_ylim([0, bound])
    ax.set_zlim([0, bound])
    fig.savefig(f'{i}_pts.png')
    pass


def main(pack):
    with pack.open('rb') as fd:
        data = pickle.load(fd)
    for i, frame in enumerate(data):
        objs, structs, pts = frame
        plot_scene(i, pts, itertools.chain(objs, structs))

if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args.pack)
