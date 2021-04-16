from MCS_exploration.gym_ai2thor.envs.mcs_env import McsEnv
from MCS_exploration.meta_controller.meta_controller import MetaController
from MCS_exploration.frame_collector import Frame_collector
import sys
import yaml
import os
import configparser

DEBUG = False

if __name__ == "__main__":
    start_scene_number = 1
    config_ini = configparser.ConfigParser()
    config_ini.read("mcs_config.ini")
    level = config_ini['MCS']['metadata']

    collector = Frame_collector(scene_dir="simple_task_img", start_scene_number=start_scene_number)

    env = McsEnv(
        task = "interaction_scenes", # first dir
        scene_type = "debug", # second nested dir
        seed = 50,
        start_scene_number = start_scene_number, 
        frame_collector = None, 
        set_trophy = False, #True if not DEBUG else False, 
        trophy_prob = 1 # trophy_prob=1 mean the trophy is 100% outside the box, trophy_prob=0 mean the trophy is 100% inside the box,
    )

    metaController = MetaController(env, level)
    result_total = 0
    number_tasks_attempted = 0
    total_actions = 0
    failure_data = {}
    exploration_success = 0
    negative_reward = 0
    number_tasks_success = 0 
    number_scenes = 1
    negative_rewards = 0
    failure_return_status = {}
    number_crash = 0
    print ("Start scene number = ", start_scene_number)
    print ("number of scenes to run = ", number_scenes)




    while env.current_scene < start_scene_number + number_scenes:
        env.reset()
        result = metaController.excecute()
        curr_scene_reward = metaController.sequence_generator_object.agent.game_state.step_output.reward 
        return_status = metaController.sequence_generator_object.agent.game_state.step_output.return_status
        if curr_scene_reward > 0 :
            number_tasks_success +=1
            result_total += curr_scene_reward
            print("SUCCESS PICKUP")
        else :
            negative_rewards += curr_scene_reward
            failure_return_status[env.current_scene] = return_status
            print ("SCENE FAIL : Action status return status", return_status)

        sys.stdout.flush()
        collector.reset()

        print ("reward from current scene = ",curr_scene_reward)
        number_tasks_attempted +=1
        game_state = metaController.sequence_generator_object.agent.game_state
        total_actions += game_state.number_actions
        if game_state.goals_found:
            exploration_success += 1

    print ("Total Success envs", number_tasks_success)
    print ("Total rewards gained from successful scenes",result_total)
    print ("Failure return status",failure_return_status)
    print ("Total crashes", number_crash)