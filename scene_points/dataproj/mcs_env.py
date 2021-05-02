"""
Base class implementation for ai2thor environments wrapper, which adds an openAI gym interface for
inheriting the predefined methods and can be extended for particular tasks.

Modified from Chengxi's McsEnv wrapper
"""

from pathlib import Path
import os
import machine_common_sense as mcs
import pickle

class McsEnv:
    def __init__(self, base, scenes, filter=None):
        base = Path(base)
        scenes = Path(scenes)
        os.environ['MCS_CONFIG_FILE_PATH'] = str(base/'mcs_config.yaml')
        app = base/'MCS-AI2-THOR-Unity-App-v0.3.3.x86_64'
        self.controller = mcs.create_controller(str(app), depth_maps=True,
                                                object_masks=True)
        self.read_scenes(scenes, filter)

    def read_scenes(self, scenedir, filter):
        _scenegen = scenedir.glob('*.json')
        if filter is None:
            self.all_scenes = list(_scenegen)
        else:
            self.all_scenes = [s for s in _scenegen if filter in s.name]

    def run_scene(self, scene_path):
        scene_config, _ = mcs.load_scene_json_file(scene_path)
        step_output = self.controller.start_scene(scene_config)
        for action in scene_config['goal']['action_list']:
            step_output = self.controller.step(action=action[0])
            yield step_output

if __name__ == '__main__':
    env = McsEnv(base='data/thor', scenes='data/thor/scenes')
    print(len(env.all_scenes))
    for i, scene_path in enumerate(env.all_scenes):
        outs = []
        print(scene_path)
        for step_output in env.run_scene(scene_path):
            outs.append(step_output)
    print('Done!')
