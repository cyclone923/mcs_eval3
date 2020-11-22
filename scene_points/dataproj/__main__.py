from argparse import ArgumentParser
from pathlib import Path

from .mcs_env import McsEnv
from .data_gen import convert_scenes

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--thor', type=Path, default=Path('data/thor'))
    parser.add_argument('--data', type=Path, default=Path('data/thor/scenes'))
    parser.add_argument('--filter', type=str, default=None)
    return parser

def main(thor_path, data_path, filter):
    env = McsEnv(thor_path, data_path, filter)
    print(f'Found {len(env.all_scenes)} scenes')
    scenes = list(env.all_scenes)
    convert_scenes(env, scenes)

if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args.thor, args.data, args.filter)
