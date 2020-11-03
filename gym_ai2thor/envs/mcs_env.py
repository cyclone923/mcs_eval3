"""
Base class implementation for ai2thor environments wrapper, which adds an openAI gym interface for
inheriting the predefined methods and can be extended for particular tasks.
"""

import os
import platform
import random
import machine_common_sense as mcs
from MCS_exploration.navigation.visibility_road_map import ObstaclePolygon
from shapely.geometry import MultiPolygon
from descartes import PolygonPatch
import matplotlib.pyplot as plt
import numpy as np

def set_goal_with_trophy(scene_config, trophy_config, plot=False):
    obstacles = []

    for obj in scene_config['objects']:
        x_list = []
        y_list = []
        for i in range(4):
            x_list.append(obj['shows'][0]['bounding_box'][i]['x'])
            y_list.append(obj['shows'][0]['bounding_box'][i]['z'])
        obstacles.append(ObstaclePolygon(x_list, y_list))

    x_list = np.array([5.5, 5.5, -5.5, -5.5])
    y_list = np.array([5.5, 5, 5, 5.5])
    for i in [1, -1]:
        obstacles.append(ObstaclePolygon(x_list*i, y_list*i))
        obstacles.append(ObstaclePolygon(y_list*i, x_list*i))

    x, z = scene_config['performerStart']['position']['x'], scene_config['performerStart']['position']['z']
    agent_radious = 0.22
    trophy_radious = 0.1

    x_list = [x+agent_radious, x+agent_radious, x-agent_radious, x-agent_radious]
    y_list = [z+agent_radious, z-agent_radious, z-agent_radious, z+agent_radious]
    obstacles.append(ObstaclePolygon(x_list, y_list))
    all_obstacles = MultiPolygon(obstacles)

    while True:
        x, z = random.random() * 10 - 5, random.random() * 10 - 5
        x_list = [x+trophy_radious, x+trophy_radious, x-trophy_radious, x-trophy_radious]
        y_list = [z+trophy_radious, z-trophy_radious, z-trophy_radious, z+trophy_radious]
        trophy = ObstaclePolygon(x_list, y_list)
        if not all_obstacles.intersection(trophy):
            trophy_x, trophy_z = x, z
            break

    if plot:
        plt.cla()
        plt.xlim((-7, 7))
        plt.ylim((-7, 7))
        plt.gca().set_xlim((-7, 7))
        plt.gca().set_ylim((-7, 7))

        patch1 = PolygonPatch(all_obstacles, fc="green", ec="black", alpha=0.2, zorder=1)
        plt.gca().add_patch(patch1)
        trophy.plot('blue')
        plt.pause(0.1)

    assert len(trophy_config['objects']) == 1
    trophy_obj = trophy_config['objects'][0].copy()
    trophy_obj['shows'][0]['position']['x'] = trophy_x
    trophy_obj['shows'][0]['position']['z'] = trophy_z

    new_scene_config = scene_config.copy()
    new_scene_config['objects'].append(trophy_obj)
    new_scene_config['goal'] = trophy_config['goal'].copy()

    return new_scene_config


class McsEnv:
    """
    Wrapper base class
    """
    def __init__(self, task=None, scene_type=None, seed=None, start_scene_number=0, frame_collector=None, set_trophy=False):

        if platform.system() == "Linux":
            app = "unity_app/MCS-AI2-THOR-Unity-App-v0.3.1.x86_64"
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
            self.scene_config = set_goal_with_trophy(self.scene_config, self.trophy_config)

        self.step_output = self.controller.start_scene(self.scene_config)
        # self.step_output = self.controller.step(action="Pass")





if __name__ == '__main__':
    McsEnv()
