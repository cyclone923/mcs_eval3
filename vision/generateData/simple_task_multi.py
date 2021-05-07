from MCS_exploration.gym_ai2thor.envs.mcs_env import McsEnv
#from vision.mcs_base import McsEnv
from MCS_exploration.meta_controller.meta_controller import MetaController
from vision.generateData.frame_collector import Frame_collector

import sys
import configparser

DEBUG = False



if __name__ == "__main__":
    start_scene_number = int(sys.argv[1]) or 0
    number_scenes = int(sys.argv[2]) or 1

    config_ini = configparser.ConfigParser()
    config_ini.read("mcs_config.ini")
    level = config_ini['MCS']['metadata']

    # collector = Frame_collector(scene_dir="simple_task_img", scene_type='interact', start_scene_number=start_scene_number)
    
    collector = Frame_collector(scene_dir="simple_task_img",
                                start_scene_number=start_scene_number,
                                scene_type='interact',
                                fg_class_en=False)

    env = McsEnv(task="interaction_scenes",
                 scene_type="e4_container",
                 seed=50,
                 start_scene_number=start_scene_number,
                 frame_collector=collector,
                 set_trophy=False,
                 trophy_prob=0 # probability for trophy outside the box, that 1-outside the box, 0-inside the box.
    )

    metaController = MetaController(env, level, collector)

    #while env.current_scene < start_scene_number + number_scenes:
    for _ in range(number_scenes):
        env.reset()
        result = metaController.execute()

        print("final reward {}".format(env.step_output.reward))

        sys.stdout.flush()
        collector.reset()
        #exit();import time; time.sleep(30)
