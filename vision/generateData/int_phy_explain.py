from gym_ai2thor.envs.mcs_env import McsEnv
from int_phy.scene_state import SceneState
from int_phy.checker import ApearanceModel, LocomotionModel
from vision.generateData.frame_collector import Frame_collector


task = "evaluation3Training"
scene_name = "shape_constancy"
#scene_name = "spatio_temporal_continuity"
scene_dir  = "instSeg/" + "intphy_scenes/" + scene_name
start_scene_number = 65
collector = Frame_collector(scene_dir=scene_dir, start_scene_number=start_scene_number)
env = McsEnv(task=task, scene_type=scene_name, start_scene_number=start_scene_number, frame_collector=collector)

appearance_checker = ApearanceModel()
locomotion_checker = LocomotionModel()

for scene in range(len(env.all_scenes) - start_scene_number):
    env.reset(random_init=False)
    # scene_state = SceneState(env.step_output, plot=False)
    print("\n")
    # print(scene + start_scene_number, env.scene_config['answer']['choice'])
    for i, x in enumerate(env.scene_config['goal']['action_list']):
        env.step(action=x[0])
        if env.step_output is None:
            break
        # scene_state.update(env.step_output, appearance_checker, locomotion_checker)

    # print("Scene appearance score: {}".format(scene_state.get_scene_appearance_scrore()))
    # print("Scene locomotion score: {}".format(scene_state.get_scene_locomotion_score()))

    # final_score = scene_state.get_final_score()
    # if final_score == 1:
    #     choice = "plausible"
    #     confidence = final_score
    # else:
    #     choice = "implausible"
    #     confidence = 1 - final_score
    # env.controller.end_scene(choice, confidence)
    collector.reset()
    # print(choice)









