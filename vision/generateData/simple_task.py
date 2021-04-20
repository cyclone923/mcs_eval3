from MCS_exploration.gym_ai2thor.envs.mcs_env import McsEnv
#from vision.mcs_base import McsEnv
from MCS_exploration.meta_controller.meta_controller import MetaController
from vision.generateData.frame_collector import Frame_collector
import sys



if __name__ == "__main__":
    scene_list       = ["transferral", "retrieval"]
    task, scene_name = "interaction_scenes", scene_list[1]
    scene_dir        = "instSeg/" + "interaction_scenes/" + scene_name

    start_scene_number, end_scene_number = 0, -1
    collector = Frame_collector(scene_dir=scene_dir,
                                start_scene_number=start_scene_number,
                                scene_type='interact',
                                fg_class_en=False)
    env = McsEnv(task=task,
                 scene_type=scene_name,
                 seed=50,
                 start_scene_number=start_scene_number,
                 frame_collector=collector,
                 set_trophy=True,
                 trophy_prob=0 # probability for trophy outside the box, that 1-outside the box, 0-inside the box.
    )
    level = 'oracle'
    metaController = MetaController(env, level)

    if end_scene_number == -1:
        end_scene_number = len(env.all_scenes)
    else:
        end_scene_number = min(end_scene_number, len(env.all_scenes))

    while env.current_scene < end_scene_number - 1:
        env.reset()
        try:
            result = metaController.excecute()
        except:
            pass
        print("final reward {}".format(env.step_output.reward))
        sys.stdout.flush()
        collector.reset()
