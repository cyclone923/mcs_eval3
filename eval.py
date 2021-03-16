import machine_common_sense as mcs
import yaml
import pickle
import os
import random
import argparse, configparser
from exploration.controller_agent import ExploreAgent
from MCS_exploration.sequence_generator import SequenceGenerator
from voe.voe_agent import VoeAgent
from voe.agency_voe_agent import AgencyVoeAgent
import physics_voe_agent
import gravity_agent

class Evaluation3_Agent:

    def __init__(self, unity_path, config_path, prefix, scene_type, seed=-1):

        try:
            assert "unity_path.yaml" in os.listdir(os.getcwd())
            assert "mcs_config.ini" in os.listdir(os.getcwd())
        except:
            raise FileNotFoundError("You might not set up mcs config and unity path yet. Please run 'bash setup_unity.sh'.")

        with open("./unity_path.yaml", 'r') as config_file:
            config = yaml.safe_load(config_file)

        config_ini = configparser.ConfigParser()
        config_ini.read(config_path)

        self.controller = mcs.create_controller(
            os.path.join(config['unity_path']),
            config_file_path=config_path
        )

        self.level = config_ini['MCS']['metadata']
        assert self.level in ['oracle', 'level1', 'level2']

        self.exploration_agent = SequenceGenerator(None, self.controller, self.level)
        self.scene_type = scene_type
        # self.agency_voe_agent = AgencyVoeAgent(self.controller, self.level)
        self.gravity_agent = gravity_agent.GravityAgent(self.controller, self.level)
        self.phys_voe = physics_voe_agent.VoeAgent(self.controller, self.level, prefix)

        if seed != -1:
            random.seed(seed)

    def run_scene(self, one_scene):
        scene_config, status = mcs.load_scene_json_file(one_scene)
        if scene_config == {}:
            raise ValueError("Scene Config is Empty", one_scene)
        goal_type = scene_config['goal']['category']
        if goal_type == "intuitive physics":
            if 'gravity' in scene_config['name'] or self.scene_type == 'gravity':
                return self.gravity_agent.run_scene(scene_config, one_scene)
            else:
                return self.phys_voe.run_scene(scene_config, one_scene)
        elif goal_type == "agents":
            if self.level == "level1":
                print("Agency task cannot be run in level1. Exiting")
                return
            self.agency_voe_agent.run_scene(scene_config)
        elif goal_type == "retrieval":
            self.exploration_agent.run_scene(scene_config)
        else:
            print("Current goal type: {}".format(goal_type))
            raise ValueError("Goal type not clear! It should be either: , 'intuitive physics', 'agents' or 'retrieval'")


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unity-path', default='unity_path.yaml')
    parser.add_argument('--config', default='mcs_config.ini')
    parser.add_argument('--prefix', default='out')
    parser.add_argument('--scenes', default='different_scenes')
    parser.add_argument('--scene-type', default='gravity')
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    agent = Evaluation3_Agent(args.unity_path, args.config, args.prefix, args.scene_type)
    goal_dir = args.scenes
    all_scenes = [
        os.path.join(goal_dir, one_scene)
        for one_scene in sorted(os.listdir(goal_dir))
        if one_scene.endswith(".json")  # All scene config files are JSON files
    ]
    random.shuffle(all_scenes)

    results = {}
    for one_scene in all_scenes:
        voe = agent.run_scene(one_scene)
