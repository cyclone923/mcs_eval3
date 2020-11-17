from gym_ai2thor.envs.mcs_env import McsEnv
from meta_controller.meta_controller import MetaController
from frame_collector import Frame_collector
import sys



if __name__ == "__main__":
    start_scene_number = 0
    collector = Frame_collector(scene_dir="simple_task_img", start_scene_number=start_scene_number)
    env = McsEnv(
        task="interaction_scenes", scene_type="transferral", seed=50,
        start_scene_number=start_scene_number, frame_collector=None, set_trophy=True, trophy_prob=1
    ) # trophy_prob=1 mean the trophy is 100% outside the box, trophy_prob=0 mean the trophy is 100% inside the box,
    metaController = MetaController(env)

    while env.current_scene < len(env.all_scenes) - 1:
        env.reset()
        result = metaController.excecute()
        sys.stdout.flush()
        collector.reset()