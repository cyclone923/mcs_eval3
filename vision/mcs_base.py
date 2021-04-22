"""
Base class implementation for ai2thor environments wrapper, which adds an openAI gym interface for
inheriting the predefined methods and can be extended for particular tasks.
"""

import os
import platform
import random
import machine_common_sense as mcs
from exploration.mcs_env.trophy import set_goal_with_trophy
import shutil
import json


class McsEnv:
    """
    Wrapper base class
    """
    def __init__(self, task=None, scene_type=None, seed=None, start_scene_number=0, frame_collector=None, set_trophy=False):

        if platform.system() == "Linux":
            app = "unity_app/MCS-AI2-THOR-Unity-App-v0.4.1.1.x86_64"
        elif platform.system() == "Darwin":
            app = "unity_app/MCSai2thor.app/Contents/MacOS/MCSai2thor"
        else:
            app = None

        self.trophy_config = None
        if set_trophy:
            goal_dir = os.path.join(task, "eval3")
            all_scenes = sorted(os.listdir(goal_dir))
            all_scenes = [os.path.join(goal_dir, one_scene) for one_scene in all_scenes]
            assert len(all_scenes) == 1
            self.trophy_config, _ = mcs.load_config_json_file(all_scenes[0])
            self.debug_dir = os.path.join(task, "debug")
            try:
                shutil.rmtree(self.debug_dir)
            except:
                pass
            os.makedirs(self.debug_dir, exist_ok=True)

        os.environ['MCS_CONFIG_FILE_PATH'] = os.path.join(os.getcwd(), "mcs_config.ini")

        self.controller = mcs.create_controller(
            os.path.join(app),
            config_file_path = os.environ['MCS_CONFIG_FILE_PATH']
        )

        if task and scene_type:
            goal_dir = os.path.join(task, scene_type)
            all_scenes = sorted(os.listdir(goal_dir))
            self.all_scenes = [os.path.join(goal_dir, one_scene) for one_scene in all_scenes]
        else:
            self.all_scenes = [os.path.join("scenes", "playroom.json")]

        self.current_scene = start_scene_number - 1

        if seed:
            random.seed(seed)

        self.frame_collector = frame_collector
        print("Frame collector: {}".format(self.frame_collector))

    def step(self, **kwargs):
        self.step_output = self.controller.step(**kwargs)

        if self.frame_collector:
            self.frame_collector.save_frame(self.step_output)

        return self.step_output

    def reset(self, random_init=False, scene_number=None):
        if scene_number:
            print(self.all_scenes[scene_number])
            self.scene_config, status = mcs.load_config_json_file(self.all_scenes[scene_number])
        else:
            if not random_init:
                self.current_scene += 1
            else:
                self.current_scene = random.randint(0, len(self.all_scenes) - 1)
            self.scene_config, status = mcs.load_config_json_file(self.all_scenes[self.current_scene])
            print(self.all_scenes[self.current_scene])

        if self.trophy_config:
            self.scene_config = set_goal_with_trophy(self.scene_config, self.trophy_config, only_trophy=False)
            with open(os.path.join(self.debug_dir, 'box_trophy_{0:0=4d}.json'.format(self.current_scene + 1)), 'w') as fp:
                json.dump(self.scene_config, fp, indent=4)

        self.step_output = self.controller.start_scene(self.scene_config)






if __name__ == '__main__':
    McsEnv()
