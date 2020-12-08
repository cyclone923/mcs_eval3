import argparse
import pickle
from pathlib import Path
from collections import defaultdict

from physicsvoe.framewisevoe import AppearanceViolation, PositionViolation, PresenceViolation

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=Path)
    return parser

def count_viols(results):
    viol_count = defaultdict(int)
    for name,frames in results.items():
        for frame in frames:
            frame_viols = [type(x) for x in frame]
            if PresenceViolation in frame_viols:
                print(name)
            for v_type in frame_viols:
                viol_count[v_type] += 1
    return viol_count

def main(main_folder):
    results = {}
    subfolders = (f for f in main_folder.iterdir() if f.is_dir())
    for folder in subfolders:
        viol_path = folder/'viols.pkl'
        if not viol_path.exists():
            continue
        with (folder/'viols.pkl').open('rb') as fd:
            viols, errs = pickle.load(fd)
        """ Remove appearance violations
        remove_app = lambda l: [v for v in l if type(v) != AppearanceViolation]
        new_viols = [remove_app(f) for f in viols]
        """
        results[folder] = new_viols
    total = len(results)
    viol_files = [f for f,vs in results.items() if any(len(v)>0 for v in vs)]
    viol = len(viol_files)
    print(f'{viol}/{total}')
    for v in viol_files:
        print(v.name)
    count = count_viols(results)
    print(count)


if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args.dir)

