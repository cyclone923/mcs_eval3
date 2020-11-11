from gym_ai2thor.envs.mcs_env import McsEnv
from meta_ontroller.meta_controller import MetaController
import sys
from tracker.frame_collector import Frame_collector



if __name__ == "__main__":
    start_scene_number, end_scene_number = 0, 100
    task, scene_type = "interaction_scenes", "transferral"
    collector = Frame_collector(scene_dir="instSeg/"+task+"/"+scene_type, start_scene_number=start_scene_number)
    env = McsEnv(
        task=task, scene_type=scene_type, seed=50,
        start_scene_number=start_scene_number, frame_collector=collector, set_trophy=False
    )
    metaController = MetaController(env)

    if end_scene_number == -1:
        end_scene_number = len(env.all_scenes)
    while env.current_scene < end_scene_number - 1:
        env.reset()
        result = metaController.excecute()
        sys.stdout.flush()
        collector.reset()
