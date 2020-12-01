from vision.mcs_base import McsEnv
from vision.generateData.frame_collector import Frame_collector


task       = "evaluation3Training"
scene_list = ["shape_constancy", "spatio_temporal_continuity", "object_permanence"]
scene_name = scene_list[0]

scene_dir  = "instSeg/" + "intphy_scenes/" + scene_name
start_scene_number = 0
collector = Frame_collector(scene_dir=scene_dir,
                            start_scene_number=start_scene_number,
                            scene_type='voe',
                            fg_class_en=False)
env = McsEnv(task=task, scene_type=scene_name, start_scene_number=start_scene_number, frame_collector=collector)


for scene in range(len(env.all_scenes) - start_scene_number):
    env.reset(random_init=False)
    print("\n")
    for i, x in enumerate(env.scene_config['goal']['action_list']):
        env.step(action=x[0])
        if env.step_output is None:
            break

    collector.reset()









