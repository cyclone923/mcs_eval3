from exploration.mcs_env.mcs_base import McsEnv
from exploration.agent import ExploreAgent2D
import sys

DEBUG = False


if __name__ == "__main__":
    start_scene_number = 1
    env = McsEnv(
        task="interaction_scenes", scene_type="transferral" if not DEBUG else 'debug', seed=50,
        start_scene_number=start_scene_number, frame_collector=None, set_trophy=True if not DEBUG else False
    )
    metaController = ExploreAgent2D(env, level='oracle')

    while env.current_scene < len(env.all_scenes) - 1:
        metaController.reset()
        result = metaController.pick_trophy()
        sys.stdout.flush()