from gym_ai2thor.envs.mcs_env import McsEnv
from meta_ontroller.meta_controller import MetaController
import sys
from frame_collector import Frame_collector



if __name__ == "__main__":
    start_scene_number = 0
    env = McsEnv(
        task="eval3_dataset", scene_type="TRAINING_SINGLE_OBJECT", seed=50,
        start_scene_number=start_scene_number, frame_collector=None, set_trophy=False
    )
    env.reset()
