"""
Base class implementation for ai2thor environments wrapper, which adds an openAI gym interface for
inheriting the predefined methods and can be extended for particular tasks.
"""

import os
import platform
import random
import machine_common_sense as mcs
from MCS_exploration.gym_ai2thor.envs.trophy import set_goal_with_trophy
import shutil
import json


class McsEnv:
    """
    Wrapper base class
    """
    def __init__(self, task=None, scene_type=None, seed=None, start_scene_number=0, frame_collector=None, set_trophy=False, trophy_prob=1):

        if platform.system() == "Linux":
            app = "unity_app/MCS-AI2-THOR-Unity-App-v0.3.4.x86_64"
        elif platform.system() == "Darwin":
            app = "unity_app/MCSai2thor.app/Contents/MacOS/MCSai2thor"
        else:
            app = None

        self.trophy_config = None
        self.trophy_prob = None
        if set_trophy:
            goal_dir = os.path.join(task, "eval3")
            all_scenes = sorted(os.listdir(goal_dir))
            all_scenes = [os.path.join(goal_dir, one_scene) for one_scene in all_scenes]
            print (len(all_scenes))
            assert len(all_scenes) == 1
            self.trophy_config, _ = mcs.load_config_json_file(all_scenes[0])
            self.debug_dir = os.path.join(task, "debug")
            self.trophy_prob = trophy_prob
            try:
                shutil.rmtree(self.debug_dir)
            except:
                pass
            os.makedirs(self.debug_dir, exist_ok=True)

        os.environ['MCS_CONFIG_FILE_PATH'] = os.path.join(os.getcwd(), "mcs_config.yaml")

        self.controller = mcs.create_controller(
            os.path.join(app)
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

        self.add_obstacle_func = None
        self.frame_collector = frame_collector
        print("Frame collector: {}".format(self.frame_collector))

    def step(self, **kwargs):
        self.step_output = self.controller.step(**kwargs)

        if self.add_obstacle_func:
            self.add_obstacle_func(self.step_output)
        # print(self.step_output.return_status)
        if self.frame_collector:
            self.frame_collector.save_frame(self.step_output)

        return self.step_output

    def reset(self, random_init=False, repeat_current=False):
        if not repeat_current:
            if not random_init:
                self.current_scene += 1
                # print(self.all_scenes[self.current_scene])
                self.scene_config, status = mcs.load_config_json_file(self.all_scenes[self.current_scene])
            else:
                self.current_scene = random.randint(0, len(self.all_scenes) - 1)
                self.scene_config, status = mcs.load_config_json_file(self.all_scenes[self.current_scene])

        # if "goal" in self.scene_config:
        #     print(self.scene_config['goal']["description"])
        # if "answer" in self.scene_config:
        #     print(self.scene_config['answer']["choice"])

        if self.trophy_config:
            self.scene_config = set_goal_with_trophy(self.scene_config, self.trophy_config, trophy_prob=self.trophy_prob)
            with open(os.path.join(self.debug_dir, 'box_trophy_{0:0=4d}.json'.format(self.current_scene)), 'w') as fp:
                json.dump(self.scene_config, fp, indent=4)

        self.step_output = self.controller.start_scene(self.scene_config)
        # self.step_output = self.controller.step(action="Pass")





if __name__ == '__main__':
    McsEnv()
