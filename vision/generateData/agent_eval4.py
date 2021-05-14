from vision.mcs_base import McsEnv
from vision.generateData.frame_collector import Frame_collector
import configparser
import sys


if __name__ == "__main__":
    start_scene_number = int(sys.argv[1]) or 0
    number_scenes = int(sys.argv[2]) or 1

    config_ini = configparser.ConfigParser()
    config_ini.read("mcs_config.ini")
    level = config_ini['MCS']['metadata']
    
    collector = Frame_collector(scene_dir="../mcs-scene-generator/eval4_agents_train/",
                                start_scene_number=start_scene_number,
                                scene_type='agents',
                                fg_class_en=False)

    env = McsEnv(task="agents",
                 scene_type="eval4",
                 seed=50,
                 start_scene_number=start_scene_number,
                 frame_collector=collector,
                 set_trophy=False,
                 trophy_prob=0 
    )

    for _ in range(number_scenes):
        env.reset(random_init=False)
        
        for x in env.scene_config['goal']['action_list']:
            env.step(action=x[0][0])
            if env.step_output is None:
                break

        collector.reset()
