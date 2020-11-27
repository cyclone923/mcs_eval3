from gym_ai2thor.envs.mcs_env import McsEnv
from meta_controller.meta_controller import MetaController
from frame_collector import Frame_collector
import sys



if __name__ == "__main__":
    start_scene_number = 0
    collector = Frame_collector(scene_dir="simple_task_img", start_scene_number=start_scene_number)
    env = McsEnv(
        task="interaction_scenes", scene_type="retrieval", seed=50,
        #task="interaction_scenes", scene_type="traversal", seed=50,
        #task="interaction_scenes", scene_type="transferral", seed=50,
        #task="interaction_scenes", scene_type="experiment", seed=50,
        start_scene_number=start_scene_number, frame_collector=None, set_trophy=True, trophy_prob=1
        #start_scene_number=start_scene_number, frame_collector=None, set_trophy=False, trophy_prob=1
    ) # trophy_prob=1 mean the trophy is 100% outside the box, trophy_prob=0 mean the trophy is 100% inside the box,
    metaController = MetaController(env)
    result_total = 0
    number_tasks_attempted = 0
    total_actions = 0
    failure_data = {}
    exploration_success = 0
    negative_reward = 0
    number_tasks_success = 0 
    number_scenes = 10
    negative_rewards = 0
    failure_return_status = {}
    print ("Start scene number = ", start_scene_number)
    print ("number of scenes to run=" ,number_scenes )

    #while env.current_scene < len(env.all_scenes) - 1:
    while env.current_scene < start_scene_number + number_scenes:
        env.reset()
        #env.reset()
        #env.reset()
        #env.reset()
        #env.reset()
        result = metaController.excecute()
        sys.stdout.flush()
        collector.reset()

        #result = metaController.excecute()
        if env.step_output.reward > 0 :
            number_tasks_success +=1
            result_total += env.step_output.reward
        else :
            negative_rewards += env.step_output.reward
            failure_return_status[env.current_scene] = env.step_output.return_status
            print ("Action status return  :", env.step_output.return_status)
            
        number_tasks_attempted +=1
        game_state = metaController.sequence_generator_object.agent.game_state
        total_actions += game_state.number_actions
        if game_state.goals_found:
            exploration_success += 1
        #f.write("scene,"+str(env.current_scene)+ ",reward,"+str(env.step_output.reward)+
        #        ",actions," + str(game_state.number_actions) + ",explore_success," + str(game_state.goals_found) + "\n")
        #f.write("scene,"+str(env.current_scene)+ ",reward,"+str(env.step_output.reward)+ ",actions," +
        #        str(game_state.number_actions)+ ",exploration_success," + str(game_state.goals_found) + ",total success,"+
        #        str(result_total)+",exploration success until now,"+ str(exploration_success)  +"\n")
        #f.close()
        #print ("scene number completed = ", env.current_scene)
        break

    #print ("Number tasks attempted" , number_tasks_attempted)
    print ("Total Success envs", number_tasks_success)
    print ("Total rewards gained from successful scenes",result_total)
    print ("Failure return status",failure_return_status)
    #print ("Total actions taken ", total_actions)
    # print(len(c.frames))
    # write_gif(c.frames, 'original.gif', fps=5)
