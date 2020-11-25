import machine_common_sense as mcs
import yaml
import os
from exploration.controller_agent import ExploreAgent


class Evaluation3_Agent:

    def __init__(self, level):
        with open("./unity_path.yaml", 'r') as config_file:
            config = yaml.safe_load(config_file)

        self.controller = mcs.create_controller(
            os.path.join(config['unity_path'])
        )

        self.level = level

        self.exploration_agent = ExploreAgent(self.controller, self.level)



    def run_scene(self, one_scene):
        scene_config, status = mcs.load_config_json_file(one_scene)
        goal_type = scene_config['goal']['category']
        if goal_type == "intuitive physics":
            pass
        elif goal_type == "agents":
            pass
        elif goal_type == "transferral":
            self.exploration_agent.run_scene(scene_config)


if __name__ == "__main__":

    agent = Evaluation3_Agent('oracle')

    goal_dir = "different_scenes"
    all_scenes = [os.path.join(goal_dir, one_scene) for one_scene in sorted(os.listdir(goal_dir))]

    for one_scene in all_scenes:
        agent.run_scene(one_scene)





