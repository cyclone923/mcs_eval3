import numpy as np
import os
import threading
import time
import json

#from networks.free_space_network import FreeSpaceNetwork
#from supervised.sequence_generator import SequenceGenerator
from sequence_generator import SequenceGenerator

#from utils import tf_util
#from utils import game_util

import constants
#import mcs.cover_floor
import cover_floor

data_buffer = []
data_counts = np.full(constants.REPLAY_BUFFER_SIZE, 9999)
os.environ["CUDA_VISIBLE_DEVICES"] = str(constants.GPU_ID)

'''
Function to create scene numbers from 0 to Max scene number

TODO : change from min scene number to max scene number 
'''
def create_scene_numbers(min_scene_number , max_scene_number):
    scene_numbers = []
    for i in range (int(min_scene_number), int(max_scene_number)+1):
        scene_numbers.append('{0:04}'.format(i))
    return scene_numbers

def explore_scene(sequence_generator,event, scene_type=None, scene_number=None, frame_collector=None):
    '''
    if scene_type != None and scene_number != None :
        sequence_generator.explore_3d_scene(event, str(scene_type)+ scene_number + ".json")
    else :
        sequence_generator.explore_3d_scene(event)
    '''

    if scene_type != None and scene_number != None :
        print("MS 1111 ")
        sequence_generator.explore_scene_view(event, str(scene_type)+ scene_number + ".json", frame_collector=frame_collector)
    else :
        print("MS 2222 ")
        sequence_generator.explore_scene_view(event, frame_collector=frame_collector)

    config_data = {}
    
    config_data['performerStart'] = {}
    config_data['performerStart']['position'] = {}
    config_data['performerStart']['rotation'] = {}

    config_data['performerStart']['position']['x']=sequence_generator.agent.game_state.position['x']
    config_data['performerStart']['position']['y']=sequence_generator.agent.game_state.position['y']
    config_data['performerStart']['position']['z']=sequence_generator.agent.game_state.position['z']

    config_data['performerStart']['rotation']['y']=sequence_generator.agent.game_state.rotation
    config_data['objects'] = []

    current_explored_objects = sequence_generator.agent.game_state.discovered_objects

    for elem in current_explored_objects:
        config_data['objects'].append(elem)
        config_data['objects'][-1]['id'] = config_data['objects'][-1]['uuid']
        # print (elem)
    
    config_data['goal'] = sequence_generator.event.goal.__dict__
    #print (config_data['goal'])
    config_data['goal']['category'] = sequence_generator.event.goal.metadata['category']
    config_data['obstacles'] = sequence_generator.agent.nav.scene_obstacles_dict
    config_data['goals_found'] = sequence_generator.agent.game_state.goals_found

    if sequence_generator.agent.game_state.goal_in_hand == True:
        config_data['goal_in_hand'] = True
        config_data['id_goal_in_hand'] = sequence_generator.agent.game_state.id_goal_in_hand
    else :
        config_data['goal_in_hand'] = False

    # print ("Agent data after end of exploration : x,y,rotation", config_data['performerStart']['position']['x'], config_data['performerStart']['position']['z'], config_data['performerStart']['rotation']['y'] )
    print ("Goal object found status : ", sequence_generator.agent.game_state.goals_found)

    return config_data


# def explore_all_scenes():
#     #sess = tf_util.Session()
#     sequence_generator = SequenceGenerator(None,None)

#     #sess.run(tf.global_variables_initializer())

#     all_scene_types = ['retrieval_goal-', 'traversal_goal-', 'transferral_goal-']
#     #scene_types = ['retrieval_goal-', 'traversal_goal-', 'transferral_goal-']
#     #scene_types = ['retrieval_goal-', 'traversal_goal-']#, 'transferral_goal-']
#     scene_types = ['transferral_goal-']
#     #scene_types = ['traversal_goal-']
#     #scene_types = ['retrieval_goal-']


#     #scene_numbers = ['0933','0934','0935']
#     #scene_numbers = ['0058']#,'0934','0935']
#     scene_numbers = ['0007']
#     #scene_numbers = create_scene_numbers('0100', '0201')
#     # print (scene_numbers)
#     #exit()
#     #scene_number = [i]
#     all_data = {}
#     training_data = {}
#     exploration_data = {}
#     actual_count_of_explored_scenes = {}
#     total_goal_objects_found = {}
#     actual_goal_data = {}
#     for elem in scene_types :
#         all_data[elem] = {"explored": [], "actual":[], 'explored_total':0, 'actual_total':0}
#         training_data[elem] = {}
#         exploration_data[elem] = {}
#         actual_count_of_explored_scenes[elem] = 0
#         total_goal_objects_found[elem] = 0
#         actual_goal_data[elem] = 0


#     total_actions = 0
#     #env = game_util.create_ai2thor_env()


#     for scene_type in scene_types :
#         for scene_number in scene_numbers :
            
#             current_explored = 0
#             #new_data, bounds, goal_pose = sequence_generator.explore_scene(str(scene_type)+ scene_number + ".json")
#             explore_scene(sequence_generator,event=None,scene_type=scene_type,scene_number=scene_number)
#             #sequence_generator.explore_3d_scene(str(scene_type)+ scene_number + ".json",event=None)
#             #exit()
#             current_explored_objects = sequence_generator.agent.game_state.discovered_objects
#             current_explored_uuids = sequence_generator.agent.game_state.discovered_explored
#             current_explored = len(current_explored_objects)
            
#             total_actions += sequence_generator.agent.game_state.number_actions
#             #sequence_generator.agent.game_state.env.end_scene('', 0.0) 
#             goal = sequence_generator.agent.game_state.event.goal
#             goal_objects = []
            
            
#             # print (type(goal))
#             #for key,value in sequence_generator.agent.game_state.goal.__dict__.items():
#             for key,value in goal.metadata.items():
#                 if key == "target" or key == "target_1" or key == "target_2":
#                     goal_objects.append(goal.metadata[key]["id"])
#                     actual_goal_data[scene_type] += 1 
#                     #goal_objects.append(goal.metadata['target_2']["id"])
#                     #print (key, type(value))

#             #sequence_generator.agent.game_state.discovered_objects = []
#             # print ("Total objects discovered = " ,current_explored )
#             #with open("discovered_data.json","w") as fp:   
#             #    print ("number of objects discovered until now : ",len(sequence_generator.agent.game_state.discovered_objects))  
#             #    json.dump(sequence_generator.agent.game_state.discovered_objects,fp,indent=1)  
#             # for elem in current_explored_uuids:
#             #     print (elem)#current_explored_objects
#             #
#             # # print ("explored objects over, goal next")
#             #
#             # for elem in goal_objects:
#             #     print (elem)#current_explored_objects
#             #

#             for elem in goal_objects :
#                 if elem in current_explored_uuids:
#                     total_goal_objects_found[scene_type] += 1 

#             '''
#             Checking for number of objects by using AIthor controller
#             current_actual = 0
#             event = game_util.reset_ai2thor_env(env,str(scene_type)+ scene_number + ".json")
#             current_actual = len(event.metadata['objects'])
#             '''
           
#             #all_data[scene_type]['explored'].append(current_explored)
#             #all_data[scene_type]['actual'].append(current_actual)
#             all_data[scene_type]['explored_total'] += current_explored
#             #all_data[scene_type]['actual_total'] += current_actual
            
#             #training_data[scene_type][scene_number] = current_actual
#             exploration_data[scene_type][scene_number] = current_explored
#             print ("Total actions until now= ", total_actions)
            
#         #for key,items in all_data.items():
#             #print ("Explored total= ", items['explored_total'])
#             #print ("Actual", items['actual_total'])

#     actual_data = json.load(open('training_total_objects_data.json'))

#     for key,value in exploration_data.items() :
#         for key2, value2 in value.items() :
#             actual_count_of_explored_scenes[key] += actual_data[key][key2] 

#     #print ("Total explored = " , all_data.items)
#     for key,items in all_data.items():
#         print ("Total explored     for scenes in {} is {}".format(key, items['explored_total']))
#         print ("Total actual       for scenes in {} is {}".format( key, actual_count_of_explored_scenes[key]))
#         print ("Total goals found  for scenes in {} is {}".format( key, total_goal_objects_found[key]))
#         print ("Total goal actual  for scenes in {} is {}".format( key, actual_goal_data[key]))


#     print ("Total actions = ", total_actions)

#     '''
#     with open("training_total_objects_data.json","w") as fp:   
#         json.dump(training_data,fp,indent=1)  
#         #Actual 2105 - retrieval 
#         #Actual 3670 - traversal
#         #Actual 3480 - transferal
#     '''


if __name__ == '__main__':
    print("Empty main!.")
    pass
    # explore_all_scenes()