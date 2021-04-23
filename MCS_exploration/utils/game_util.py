# -*- coding: utf-8 -*-
import pdb
import numpy as np
import cv2
from ai2thor import controller
#from utils import bb_util
import machine_common_sense as mcs
import machine_common_sense
import math
import json
import platform


import constants

GRID_SIZE = 0.1
MAX_REACH_DISTANCE = 1.0

def wrap_step( **kwargs):
    # Create the step data dict for the AI2-THOR step function.
    step_data = dict(
        continuous=True,
        gridSize=GRID_SIZE,
        logs=True,
        renderImage=True,
        renderDepthImage=False,
        renderClassImage=False,
        renderObjectImage=False,
        #    renderImage=render_image,
        #    renderDepthImage=render_depth_image,
        #    renderClassImage=render_class_image,
        #    renderObjectImage=render_object_image))
        # Yes, in AI2-THOR, the player's reach appears to be governed by the "visibilityDistance", confusingly...
        visibilityDistance=MAX_REACH_DISTANCE,
        **kwargs
    )

    return step_data

RUN_MCS = 1

unity_app_file_path_linux = "unity_app/MCS-AI2-THOR-Unity-App-v0.4.1.1.x86_64"
config_file_folder_linux = "/home/rajesh/rajesh/mcs_data/interaction_scenes/"
unity_app_file_path_mac = "/Users/rajesh/Rajesh/Subjects/Research/aiThor/mcs_playroom_old/algorithms/a3c/gym_ai2thor/MCSai2thor.app/Contents/MacOS/MCSai2thor"
config_file_folder_mac = "/Users/rajesh/Rajesh/Subjects/Research/aiThor/interaction_scenes/"

def create_env(x_display=constants.X_DISPLAY,
               quality='MediumCloseFitShadows',
               player_screen_height=constants.SCREEN_HEIGHT,
               player_screen_width=constants.SCREEN_WIDTH):
    print('Creating env')

    if platform.system() == "Linux":
        unity_app_file_path = unity_app_file_path_linux
    elif platform.system() == "Darwin":
        unity_app_file_path = unity_app_file_path_mac
    else:
        app = None
        
    if RUN_MCS != 1 :
        env = controller.Controller(quality=quality)
        env.start(x_display=x_display,
                  player_screen_height=player_screen_height,
                  player_screen_width=player_screen_width)
    
    
    else : 
        #unity_app_file_path = "/home/rajesh/rajesh/mcs_data/MCS-AI2-THOR-Unity-App-v0.0.6.x86_64" 
        #unity_app_file_path = "/Users/rajesh/Rajesh/Subjects/Research/aiThor/mcs_playroom_old/algorithms/a3c/gym_ai2thor/MCSai2thor.app/Contents/MacOS/MCSai2thor"

        '''
        env = controller.Controller(
                quality='Medium',
                fullscreen=False,
                # The headless flag does not work for me
                headless=False,
                local_executable_path=unity_app_file_path,
                width=player_screen_width,
                height=player_screen_height,
                # Set the name of our Scene in our Unity app
                scene='MCS',
                logs=True,
                #quiality=quality,
                # This constructor always initializes a scene, so add a scene config to ensure it doesn't error
                sceneConfig={
                    "objects": []
                }
            )
        '''
        #env = machine_common_sense.MCS_Controller_AI2THOR(
        # env = mcs.MCS_Controller_AI2THOR(unity_app_file_path)

        os.environ['MCS_CONFIG_FILE_PATH'] = os.path.join(os.getcwd(), "mcs_config.ini")
        
        env = mcs.create_controller(
            "unity_app/MCS-AI2-THOR-Unity-App-v0.4.1.1.x86_64",
            config_file_path = os.environ['MCS_CONFIG_FILE_PATH']
        )

        
        
        #                 "/Users/rajesh/Rajesh/Subjects/Research/aiThor/mcs_playroom_old/algorithms/a3c/gym_ai2thor/MCSai2thor.app/Contents/MacOS/MCSai2thor")
    #env.start(x_display=x_display)
    
    print('Starting env, if this takes more than a few seconds (except for downloading the build), the display is not set correctly')

    print('Done starting env')
    print ("type of env", type(env))
    return env


def create_ai2thor_env(x_display=constants.X_DISPLAY,
               quality='MediumCloseFitShadows',
               player_screen_height=constants.SCREEN_HEIGHT,
               player_screen_width=constants.SCREEN_WIDTH):
    print('Creating env')
        
    if platform.system() == "Linux":
        unity_app_file_path = unity_app_file_path_linux
    elif platform.system() == "Darwin":
        unity_app_file_path = unity_app_file_path_mac
    else:
        app = None
    #unity_app_file_path = "/Users/rajesh/Rajesh/Subjects/Research/aiThor/mcs_playroom_old/algorithms/a3c/gym_ai2thor/MCSai2thor.app/Contents/MacOS/MCSai2thor"
    
    env = controller.Controller(
            quality='Medium',
            fullscreen=False,
            # The headless flag does not work for me
            headless=False,
            local_executable_path=unity_app_file_path,
            width=player_screen_width,
            height=player_screen_height,
            # Set the name of our Scene in our Unity app
            scene='MCS',
            logs=True,
            #quiality=quality,
            # This constructor always initializes a scene, so add a scene config to ensure it doesn't error
            sceneConfig={
                "objects": []
            }
        )

    return env

def reset_ai2thor_env(env , config_filename):
    
    if platform.system() == "Linux":
        config_file_folder = config_file_folder_linux
    elif platform.system() == "Darwin":
        config_file_folder = config_file_folder_mac
    else:
        app = None

    #config_json_file_path = "/Users/rajesh/Rajesh/Subjects/Research/aiThor/interaction_scenes/" + config_filename
    config_json_file_path = config_file_folder + config_filename
    #traversal_goal-0004.json"
    config_data , status = MCS.load_scene_json_file(config_json_file_path)

    event =env.step(wrap_step(action="Initialize", sceneConfig=config_data))
    #event = env.start_scene(config_data)

    return event

def reset(env, scene_name_or_num, config_filename="",
        grid_size=constants.AGENT_STEP_SIZE,
        camera_y=constants.CAMERA_HEIGHT_OFFSET,
        render_image=constants.RENDER_IMAGE,
        render_depth_image=constants.RENDER_DEPTH_IMAGE,
        render_class_image=constants.RENDER_CLASS_IMAGE,
        render_object_image=constants.RENDER_OBJECT_IMAGE):
    if type(scene_name_or_num) == str:
        scene_name = scene_name_or_num
    else:
        scene_name = 'FloorPlan%d' % scene_name_or_num

    if platform.system() == "Linux":
        config_file_folder = config_file_folder_linux
    elif platform.system() == "Darwin":
        config_file_folder = config_file_folder_mac
    else:
        app = None

    if RUN_MCS == 1 :
    
        #config_json_file_path = "/home/rajesh/rajesh/mcs_data/playroom.json"
        #config_json_file_path = "/home/rajesh/rajesh/mcs_data/interaction_scenes/traversal_goal-0004.json"
        #config_json_file_path = "/Users/rajesh/Rajesh/Subjects/Research/aiThor/interaction_scenes/traversal_goal-0004.json"
        #config_json_file_path = "/Users/rajesh/Rajesh/Subjects/Research/aiThor/interaction_scenes/" + config_filename
        config_json_file_path = config_file_folder + config_filename
        print (config_json_file_path)
        config_data , status = MCS.load_scene_json_file(config_json_file_path)

        #event =env.step(wrap_step(action="Initialize", sceneConfig=config_data))
        event = env.start_scene(config_data)
    else :
        env.reset(scene_name)
        event = env.step(dict(
            action='Initialize',
            gridSize=grid_size,
            cameraY=camera_y,
            renderImage=render_image,
            renderDepthImage=render_depth_image,
            renderClassImage=render_class_image,
            renderObjectImage=render_object_image))
    
    #print ("event type", type(event))
    return event


def distance(state1, state2):
    # Manhattan distance plus rotational equality.
    rot_diff_y = abs(state1[1]['y'] - state2[1]['y']) / 90.0
    if rot_diff_y > 2:
        rot_diff_y = 1 # 270 - 0
    return (abs(state1[0]['x'] - state2[0]['x']) +
            abs(state1[0]['z'] - state2[0]['z']) +
            abs(state1[1]['x'] - state2[1]['x']) / 15 +
            rot_diff_y)


def get_pose(event):
    
    if RUN_MCS != 1 :
        pose = event.pose
        pose_1 =  (int(np.round(pose[0] / (1000 * constants.AGENT_STEP_SIZE))),
                int(np.round(pose[1] / (1000 * constants.AGENT_STEP_SIZE))),
                int(np.round(pose[2] / (1000 * 90))),
                int(np.round(pose[3] / (1000))))
        return pose_1
    
    else :
        '''
        agent_data = event.metadata['agent']
        #x = int(agent_data['position']['x'] / constants.AGENT_STEP_SIZE)
        #z = int(agent_data['position']['z'] / constants.AGENT_STEP_SIZE)
        x = math.floor(agent_data['position']['x'] / constants.AGENT_STEP_SIZE)
        z = math.floor(agent_data['position']['z'] / constants.AGENT_STEP_SIZE)
        rotation = int((agent_data['rotation']['y']/90))
        horizon = int(agent_data['cameraHorizon'])
        '''
        x = math.floor(event.position['x']/constants.AGENT_STEP_SIZE)
        z = math.floor(event.position['z']/constants.AGENT_STEP_SIZE)

        rotation = int(event.rotation/90)
        horizon = int(event.head_tilt)


        pose_2 =  (x,z,rotation,horizon)
        #print ("actual x and z : ", agent_data['position']['x'], agent_data['position']['z'])
        #print ("calculated x and z poses", x,z)
        return pose_2


def pretty_action(action):
    import copy
    action_dict = copy.deepcopy(action)
    if 'x' in action_dict:
        action_dict['x'] = '%0.6f' % action_dict['x']
    if 'z' in action_dict:
        action_dict['z'] = '%0.6f' % action_dict['z']
    return str(action_dict)


def unique_rows(arr, return_index=False, return_inverse=False):
        arr = np.ascontiguousarray(arr).copy()
        b = arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
        if return_inverse:
            _, idx, inv = np.unique(b, return_index=True, return_inverse=True)
        else:
            _, idx = np.unique(b, return_index=True)
        unique = arr[idx]
        if return_index and return_inverse:
            return unique, idx, inv
        elif return_index:
            return unique, idx
        elif return_inverse:
            return unique, inv
        else:
            return unique


def choose_action(pi_values):
    # In rare cases, the pis are small and don't get properly normalized. This should handle that.
    pi_values /= np.sum(pi_values)
    return np.random.choice(len(pi_values), 1, p=pi_values)[0]


def choose_action_q(q_values):
    if constants.EVAL:
        return np.argmax(q_values)
    else:
        pi_values = np.exp(q_values)
        pi_values /= np.sum(pi_values)
        return np.random.choice(len(pi_values), 1, p=pi_values)[0]


def imresize(image, size, rescale=True):
    if image is None:
        #print ("image is none")
        return None
    #print ("image not none")
    if image.shape[0] != size[0] or image.shape[1] != size[1]:
        image = cv2.resize(image, size)
    if rescale:
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        image /= 255.0
    #print (len(image),type(image))
    return image


def depth_imresize(image, size, rescale=True, max_depth=constants.MAX_DEPTH):
    if image is None:
        return None
    if image.shape[0] != size[0] or image.shape[1] != size[1]:
        image = cv2.resize(image, size)
    image[image > max_depth] = max_depth
    if rescale:
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        image /= max_depth
    return image


def get_rotation_matrix(pose):
    sinX = np.sin(-pose[3] * np.pi / 180)
    cosX = np.cos(-pose[3] * np.pi / 180)
    xRotation = np.matrix([
        [1, 0, 0],
        [0, cosX, -sinX],
        [0, sinX, cosX]])
    sinY = np.sin((-pose[2] % 4) * 90 * np.pi / 180)
    cosY = np.cos((-pose[2] % 4) * 90 * np.pi / 180)
    yRotation = np.matrix([
        [cosY, 0, sinY],
        [0, 1, 0],
        [-sinY, 0, cosY]])
    rotation_matrix = np.matmul(xRotation, yRotation)
    return rotation_matrix


def depth_to_world_coordinates(depth, pose, camera_height):
    x_points = np.arange(-constants.SCREEN_WIDTH / 2, constants.SCREEN_WIDTH / 2)
    x_vals = (depth * x_points / constants.FOCAL_LENGTH)

    y_points = np.arange(constants.SCREEN_HEIGHT / 2, -constants.SCREEN_HEIGHT / 2, -1)
    y_vals = (depth.T * y_points / constants.FOCAL_LENGTH).T

    z_vals = depth
    xyz = np.stack((x_vals, y_vals, z_vals), axis=2) / (1000 * constants.AGENT_STEP_SIZE)
    rotation_matrix = np.linalg.inv(get_rotation_matrix(pose))
    xyz = np.array(np.dot(rotation_matrix, xyz.reshape(-1, 3).T).T).reshape(constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH, 3)
    xzy = xyz[:, :, [0, 2, 1]]
    xzy += np.array([pose[0], pose[1], camera_height])
    return xzy


def get_action_str(action):
    if action['action'] == 'Teleport':
        action_str = 'Goto'
    elif action['action'] == 'OpenObject':
        if 'objectId' not in action:
            action['objectId'] = 'None'
        action_str = 'Open %s' % action['objectId'].split('|')[0]
    elif action['action'] == 'CloseObject':
        if 'objectId' not in action:
            action['objectId'] = 'None'
        action_str = 'Close %s' % action['objectId'].split('|')[0]
    elif action['action'] == 'RotateByDegree':
        action_str = 'RotateByDegree %d' % action['rotation']
    elif action['action'] == 'LookByDegree':
        action_str = 'LookByDegree %d' % action['rotation']
    else:
        action_str = action['action']
    return action_str


def get_object(object_id, metadata):
    for obj in metadata['objects']:
        if obj['objectId'] == object_id:
            return obj
    return None


def get_objects_of_type(object_type, metadata):
    return [obj for obj in metadata['objects'] if obj['objectType'] == object_type]


def get_object_bounds(obj, scene_bounds):
    obj_bounds = np.array(obj['bounds3D'])[[0, 2, 3, 5]] # Get X and Z out
    obj_bounds /= constants.AGENT_STEP_SIZE
    obj_bounds = np.round(obj_bounds).astype(np.int32)
    obj_bounds[[2, 3]] = np.maximum(obj_bounds[[2, 3]], obj_bounds[[0, 1]] + 1)
    obj_bounds[[0, 2]] = np.clip(obj_bounds[[0, 2]], scene_bounds[0], scene_bounds[0] + scene_bounds[2])
    obj_bounds[[1, 3]] = np.clip(obj_bounds[[1, 3]], scene_bounds[1], scene_bounds[1] + scene_bounds[3])
    obj_bounds -= np.array(scene_bounds)[[0, 1, 0, 1]]
    return obj_bounds


def get_object_point(obj, scene_bounds):
    obj_point = np.array([obj['position']['x'], obj['position']['z']])
    obj_point /= constants.AGENT_STEP_SIZE
    obj_point = np.round(obj_point).astype(np.int32)
    obj_point[0] = np.clip(obj_point[0], scene_bounds[0], scene_bounds[0] + scene_bounds[2] - 1)
    obj_point[1] = np.clip(obj_point[1], scene_bounds[1], scene_bounds[1] + scene_bounds[3] - 1)
    obj_point -= np.array(scene_bounds)[[0, 1]]
    return obj_point


def check_object_size(bounds):
    if len(bounds) == 0:
        return False
    bounds = np.array(bounds)
    bounds[[0, 2]] = np.clip(bounds[[0, 2]], 0, constants.SCREEN_WIDTH)
    bounds[[1, 3]] = np.clip(bounds[[1, 3]], 0, constants.SCREEN_HEIGHT)
    area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    return min(abs(bounds[2] - bounds[0]), abs(bounds[3] - bounds[1])) > constants.MIN_DETECTION_LEN


def object_size(obj):
    return np.array(obj['bounds3D'][3:]) - np.array(obj['bounds3D'][:3])


def object_center_position(obj):
    return np.array([obj['position']['x'], obj['position']['y'], obj['position']['z']])


def get_question_str(question_type_ind, question_object_ind, question_container_ind=None):
    object_article = 'a'
    if constants.OBJECTS_SINGULAR[question_object_ind][0] in {'a', 'e', 'i', 'o', 'u'}:
        object_article = 'an'
    container_article = 'a'
    if question_container_ind is not None and constants.OBJECTS_SINGULAR[question_container_ind][0] in {'a', 'e', 'i', 'o', 'u'}:
        container_article = 'an'
    if question_container_ind is not None and constants.OBJECTS_SINGULAR[question_container_ind] in {'fridge', 'microwave', 'sink'}:
        container_article = 'the'

    if question_type_ind == 0:
        return 'Is there %s %s in the room?' % (object_article, constants.OBJECTS_SINGULAR[question_object_ind])
    elif question_type_ind == 1:
        return 'How many %s are there in the room?' % constants.OBJECTS_PLURAL[question_object_ind]
    elif question_type_ind == 2:
        preposition = 'in'
        if constants.OBJECTS[question_container_ind] in {'StoveBurner', 'TableTop'}:
            preposition = 'on'
        return ('Is there %s %s %s %s %s?' % (
            object_article, constants.OBJECTS_SINGULAR[question_object_ind],
            preposition,
            container_article,
            constants.OBJECTS_SINGULAR[question_container_ind]))

'''
def set_open_close_object(action, last_event):
    # The object nearest the center of the screen is open/closed if none is provided.
    if 'objectId' not in action:
        openable = [obj for obj in last_event.metadata['objects']
                    if (obj['visible'] and obj['openable'] and
                        (obj['isopen'] == (action['action'] == 'CloseObject')) and
                        obj['objectId'] in last_event.instance_detections2D)]
        if len(openable) > 0:
            boxes = np.array([last_event.instance_detections2D[obj['objectId']] for obj in openable])
            boxes_xywh = bb_util.xyxy_to_xywh(boxes.T).T
            mids = boxes_xywh[:, :2]
            dists = np.sqrt(np.sum(np.square(
                (mids - np.array([constants.SCREEN_WIDTH / 2, constants.SCREEN_HEIGHT / 2]))), axis=1))
            obj_ind = int(np.argmin(dists))
            action['objectId'] = openable[obj_ind]['objectId']
        else:
            # Nothing to open
            action['objectId'] = ''
    return action
'''
