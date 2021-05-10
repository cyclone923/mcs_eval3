from MCS_exploration.gym_ai2thor.envs.mcs_env import McsEnv
#from vision.mcs_base import McsEnv
from MCS_exploration.meta_controller.meta_controller import MetaController
from vision.generateData.frame_collector import Frame_collector

import sys
import argparse, configparser


def make_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--start_scene', default=0)
    parser.add_argument('--end_escene', default=268)
    parser.add_argument('--outdir', default='simple_task_img')
    parser.add_argument('--dir1', default='interaction_scenes')
    parser.add_argument('--dir2', default='retrieval_scenes_e4')

    return parser

if __name__ == "__main__":

    args = make_parser().parse_args()

    config_ini = configparser.ConfigParser()
    config_ini.read("mcs_config.ini")
    level = config_ini['MCS']['metadata']
    
    collector = Frame_collector(scene_dir=args.outdir,
                                start_scene_number=args.start_scene,
                                scene_type='interact',
                                fg_class_en=False)

    env = McsEnv(task=args.dir1,
                 scene_type=args.dir2,
                 seed=50,
                 start_scene_number=args.start_scene,
                 frame_collector=collector,
                 set_trophy=False,
                 trophy_prob=0 # probability for trophy outside the box, that 1-outside the box, 0-inside the box.
    )

    metaController = MetaController(env, level, collector)

    for _ in range(args.end_escene):
        env.reset()
        result = metaController.execute()

        print("final reward {}".format(env.step_output.reward))

        sys.stdout.flush()
        # collector.reset()
        #exit();import time; time.sleep(30)
