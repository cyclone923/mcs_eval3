from MCS_exploration.frame_processing import convert_observation
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
import numpy as np
import cv2
import json

class Evaluation4_Agent:

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
        self.phys_voe = physics_voe_agent.PhysicsVoeAgent(self.controller, self.level, prefix)

        if seed != -1:
            random.seed(seed)

    def run_scene(self, one_scene):
        scene_config, status = mcs.load_scene_json_file(one_scene)
        if scene_config == {}:
            raise ValueError("Scene Config is Empty", one_scene)
        goal_type = scene_config['goal']['category']
        if goal_type == "intuitive physics":
            return self.phys_voe.run_scene(scene_config, one_scene)
        elif goal_type == "agents":
            if self.level == "level1":
                print("Agency task cannot be run in level1. Exiting")
                return
            self.agency_voe_agent.run_scene(scene_config)
        elif goal_type == "retrieval":
            print("\nPLAYROOM SCENE...\n")
            self.exploration_agent.run_scene(scene_config)
        else:
            print("Current goal type: {}".format(goal_type))
            raise ValueError("Goal type not clear! It should be either: , 'intuitive physics', 'agents' or 'retrieval'")

    def collect_scene_data(self, one_scene):
        scene_config, status = mcs.load_scene_json_file(one_scene)
        
        # make a folder for scene data
        if not os.path.exists("./{}/".format(scene_config["name"])):
            os.mkdir("./{}/".format(scene_config["name"]))
        self.controller.start_scene(scene_config)
        
        for i, pos in enumerate(scene_config['goal']['action_list']):
            step_output = self.controller.step(action=pos[0][0])  # Get the step output
            depth_map = step_output.depth_map_list[0]

            rgb_image = step_output.image_list[0]
            rgb_image = np.array(rgb_image)[:, :, ::-1]
            mask_image = step_output.object_mask_list[0]
            mask_image = np.array(mask_image)[:, :, ::-1]
            
            if not os.path.exists("./{}/RGB".format(scene_config["name"])):
                os.mkdir("./{}/RGB/".format(scene_config["name"]))
            cv2.imwrite("./{}/RGB/{}.jpg".format(scene_config["name"], i), rgb_image)
            
            if not os.path.exists("./{}/Depth".format(scene_config["name"])):
                os.mkdir("./{}/Depth/".format(scene_config["name"]))
            cv2.imwrite("./{}/Depth/{}.jpg".format(
                scene_config["name"], i), depth_map * 255. / depth_map.max())
            
            if not os.path.exists("./{}/Mask".format(scene_config["name"])):
                os.mkdir("./{}/Mask/".format(scene_config["name"]))
            cv2.imwrite("./{}/Mask/{}.jpg".format(scene_config["name"], i), mask_image)

            if not os.path.exists("./{}/Step_Output/".format(scene_config["name"])):
                os.mkdir("./{}/Step_Output/".format(scene_config["name"]))
            with open("./{}/Step_Output/step_{}.json".format(scene_config["name"], i), "w+") as json_file:
                json.dump(dict(step_output), json_file, indent = 4)

            # Bounding Boxes for MOT tracking
            obj_mask = np.prod(mask_image, axis=2)
            uniq_obj_colors = np.unique(obj_mask)
            rgb_image = cv2.imread("./{}/RGB/{}.jpg".format(scene_config["name"], i))

            with open("./{}/gt.txt".format(scene_config["name"]), "a+") as gt_file:
                for color in uniq_obj_colors:
                    mask = np.where(obj_mask == color, 1.0, 0.0)
                    mask = np.expand_dims(mask, axis=2)
                    mask = (np.squeeze(mask)).astype("uint8")

                    MM = cv2.moments(mask)
                    cX = int(MM["m10"] / MM["m00"])
                    cY = int(MM["m01"] / MM["m00"])

                    w = mask.sum(axis=1).max()
                    h = mask.sum(axis=0).max()

                    top_left = (max(cX - w/2, 0), cY + h/2)
                    top_left = (np.int(top_left[0]), np.int(top_left[1]))

                    bottom_right = (cX + w/2, max(cY - h/2, 0))
                    bottom_right = (np.int(bottom_right[0]), np.int(bottom_right[1]))

                    rgb = (255, 0, 0)
                    rgb_image = cv2.rectangle(rgb_image, top_left, bottom_right, rgb, thickness=2)

                    gt_file.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(
                        i, color, top_left[0], top_left[1], w, h, -1, -1, -1, -1)
                    )
        self.controller.end_scene("Plausible", 1.0)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unity-path', default='unity_path.yaml')
    parser.add_argument('--config', default='mcs_config.ini')
    parser.add_argument('--prefix', default='out')
    parser.add_argument('--scenes', default='different_scenes')
    parser.add_argument('--scene-type', default='')
    parser.add_argument('--data-collection', default='false')
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    agent = Evaluation4_Agent(args.unity_path, args.config, args.prefix, args.scene_type)
    goal_dir = args.scenes
    all_scenes = [
        os.path.join(goal_dir, one_scene)
        for one_scene in sorted(os.listdir(goal_dir))
        if one_scene.endswith(".json")  # All scene config files are JSON files
    ]
    random.shuffle(all_scenes)

    results = {}
    for one_scene in all_scenes:
        if args.data_collection == 'false':
            voe = agent.run_scene(one_scene)
        elif args.data_collection == "true":
            agent.collect_scene_data(one_scene)
