from exploration.agent import ExploreAgent2D
from exploration.mcs_env.trophy import set_goal_with_trophy
import machine_common_sense as mcs
import os
import json
import shutil


class ControllerEnv:

    def __init__(self, controller):
        self.controller = controller
        self.scene_config = None

    def set_scene(self, config):
        self.scene_config = config

    def step(self, **kwargs):
        self.step_output = self.controller.step(**kwargs)

    def reset(self):
        self.step_output = self.controller.start_scene(self.scene_config)


class ExploreAgent:

    def __init__(self, controller, level):
        env = ControllerEnv(controller)
        self.agent = ExploreAgent2D(env, level)

        self.debug_dir = os.path.join("interaction_scenes", "debug")
        try:
            shutil.rmtree(self.debug_dir)
        except:
            pass
        os.makedirs(self.debug_dir, exist_ok=True)

    def run_scene(self, config):
        trophy_config, _ = mcs.load_config_json_file("interaction_scenes/eval3/hinged_container_example.json")
        new_config = set_goal_with_trophy(config, trophy_config, only_trophy=False)

        self.agent.env.set_scene(new_config)
        self.agent.reset()
        self.agent.pick_trophy()

    def dump_scene_for_debug(self, config):
        with open(os.path.join(self.debug_dir, 'box_trophy_current_fail'), 'w') as fp:
            json.dump(config, fp, indent=4)