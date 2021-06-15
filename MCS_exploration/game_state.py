import time
import random
import numpy as np

#from utils import game_util
#from utils import action_util
from MCS_exploration.utils import game_util
#from MCS_exploration.utils import action_util
#from darknet_object_detection import detector
from machine_common_sense import StepMetadata
from machine_common_sense import ObjectMetadata
from machine_common_sense import Util
from cover_floor import *
import shapely.geometry.polygon as sp
from MCS_exploration.navigation.visibility_road_map import ObstaclePolygon,IncrementalVisibilityRoadMap
from MCS_exploration.frame_processing import *
from shapely.geometry import Point, MultiPoint
from MCS_exploration.obstacle import Obstacle
import copy
from vision.instSeg.inference import MaskAndClassPredictor
from vision.instSeg.data.config_mcsVideo3_inter import MCSVIDEO_INTER_CLASSES_BG, MCSVIDEO_INTER_CLASSES_FG
from vision.generateData.frame_collector import Frame_collector

TROPHY_INDEX = MCSVIDEO_INTER_CLASSES_FG.index('trophy') + 1
BOX_INDEX = MCSVIDEO_INTER_CLASSES_FG.index('box') + 1


import constants

assert(constants.SCENE_PADDING == 5)

def wrap_output( scene_event):

    step_output = StepMetadata(
        object_list=retrieve_object_list(scene_event),
    )

    return step_output

def retrieve_object_colors( scene_event):
    # Use the color map for the final event (though they should all be the same anyway).
    return scene_event.events[len(scene_event.events) - 1].object_id_to_color

def retrieve_object_list( scene_event):
    return sorted([retrieve_object_output(object_metadata, retrieve_object_colors(scene_event)) for \
            object_metadata in scene_event.metadata['objects']
                if object_metadata['visible'] or object_metadata['isPickedUp']], key=lambda x: x.uuid)

def retrieve_object_output( object_metadata, object_id_to_color):
    material_list = list(filter(Util.verify_material_enum_string, [material.upper() for material in \
            object_metadata['salientMaterials']])) if object_metadata['salientMaterials'] is not None else []

    rgb = object_id_to_color[object_metadata['objectId']] if object_metadata['objectId'] in object_id_to_color \
            else [None, None, None]

    bounds = object_metadata['objectBounds'] if 'objectBounds' in object_metadata and \
        object_metadata['objectBounds'] is not None else {}

    return ObjectMetadata(
        uuid=object_metadata['objectId'],
        color={
            'r': rgb[0],
            'g': rgb[1],
            'b': rgb[2]
        },
        #dimensions=(bounds['objectBoundsCorners'] if 'objectBoundsCorners' in bounds else None),
        position = object_metadata['position'],
        visible=(object_metadata['visible'] or object_metadata['isPickedUp'])

    )


def retrieve_position(scene_event):
    return scene_event.metadata['agent']['position']


class GameState(object):
    def __init__(self, env=None,level="level1", depth_scope=None, frame_collector=None):
        if env == None:
            self.env = game_util.create_env()
        else :
            self.env = env
        #print ("game state init")
        #self.action_util = action_util.ActionUtil()
        self.local_random = random.Random()
        self.im_count = 0
        self.times = np.zeros((4, 2))
        self.discovered_explored = {} 
        self.discovered_objects = []
        self.number_actions = 0
        self.add_obstacle_func = None
        self.add_obstacle_func_eval3 = None
        self.goals_found = False
        self.goals = []
        self.world_poly = None
        self.new_found_objects = []
        self.new_object_found = False
        self.goal_in_hand = False
        self.id_goal_in_hand = None
        self.get_obstacles = None
        self.goal_object = None
        self.goal_bounding_box = None
        self.goal_calculated_points = None
        self.goal_object_visible = False
        self.grid_size = 0.1 
        #self.grid_size = 1 
        self.map_width = 48
        self.map_length = 48                                                                     
        self.displacement = 12
        self.occupancy_map = self.occupancy_map_init() #* unexplored
        self.object_mask = None
        self.goal_id = None
        self.goal_ids = []
        self.pose_estimate = np.zeros((3,1),dtype = np.float64)
        self.global_obstacles = []
        self.current_frame_obstacles = []
        self.objs = 0
        self.oracle_position = None
        self.position = None
        self.rotation = None
        self.head_tilt = None
        self.trophy_location = None #[Trophy location in the img seg list] 
        self.trophy_mask = None
        self.trophy_obstacle = None
        self.trophy_picked_up = False
        self.trophy_prob_threshold = 0.3
        #self.level = "oracle"
        #self.level = "level1"
        #self.level = "level2"
        self.level= level
        self.trophy_visible_current_frame = False
        self.img_seg_occupancy_map_points = {}
        self.current_frame_img_obstacles = []
        self.img_channels = None
        self.mask_predictor = MaskAndClassPredictor(dataset='mcsvideo3_inter',
                                                   config='plus_resnet50_config_depth_MC',
                                                   weights='./vision/instSeg/dvis_resnet50_mc.pth')

        self.collector = frame_collector


    def occupancy_map_init(self):
        #rows = int(self.map_width//self.grid_size)
        #cols = int(self.map_length//self.grid_size)
        rows = int(self.map_width//self.grid_size)
        cols = int(self.map_length//self.grid_size)
                                                                           
        unexplored = 0 
        occupancy_map = np.zeros((rows,cols)) 
        return occupancy_map

    def process_frame(self, run_object_detection=False):
        self.im_count += 1
        self.pose = game_util.get_pose(self)
        i = 0
        return

    def reset(self, scene_name=None, use_gt=True, seed=None, config_filename= "",event=None):

        if self.collector:
            self.collector.reset()
                
        if scene_name is None:
            # Do half reset
            action_ind = self.local_random.randint(0, constants.STEPS_AHEAD ** 2 - 1)
            action_x = action_ind % constants.STEPS_AHEAD - int(constants.STEPS_AHEAD / 2)
            action_z = int(action_ind / constants.STEPS_AHEAD) + 1
            x_shift = 0
            z_shift = 0
            if self.pose[2] == 0:
                x_shift = action_x
                z_shift = action_z
            elif self.pose[2] == 1:
                x_shift = action_z
                z_shift = -action_x
            elif self.pose[2] == 2:
                x_shift = -action_x
                z_shift = -action_z
            elif self.pose[2] == 3:
                x_shift = -action_z
                z_shift = action_x
            action_x = self.pose[0] + x_shift
            action_z = self.pose[1] + z_shift
            self.end_point = (action_x, action_z, self.pose[2])
            #print ("in the game state reset end point is : ", self.end_point

        else:
            # Do full reset
            #self.world_poly = fov.FieldOfView([0, 0, 0], 0, [])
            self.world_poly = sp.Polygon()
            self.scene_name = scene_name
            self.number_actions = 0
            self.id_goal_in_hand = None
            #print ("Full reset - in the first time of load")
            self.graph = None
            self.goal_in_hand = False
            if seed is not None:
                self.local_random.seed(seed)
            lastActionSuccess = False
            self.discovered_explored = {}
            self.discovered_objects = []
            self.occupancy_map = self.occupancy_map_init() #* unexplored
            self.object_mask = None
            self.goal_id = None
            self.goal_ids = []
            self.goals_found = False
            self.pose_estimate = np.zeros((3,1),dtype = np.float64)
            self.global_obstacles = []
            self.goal_object_visible = False
            self.trophy_picked_up = False
            self.position = None
            self.rotation = None
            self.head_tilt = None
            self.bounds = None
            self.objs = 0
            self.trophy_visible_current_frame = False
            self.img_seg_occupancy_map_points = {}
            self.current_frame_img_obstacles = []
            self.img_channels = None

            #while True :
            #self.event = self.event.events[0]
            if event != None :
                self.event = event
            else :
                self.event = game_util.reset(self.env, self.scene_name,config_filename)
            self.goals = []

            '''
            Oracle data being used (eval 2)
            '''

            '''
            for key,value in self.event.goal.metadata.items():
                if key == "target" or key == "target_1" or key == "target_2":
                    self.goals.append(self.event.goal.metadata[key]["id"])

            for obj in self.event.object_list:
                if obj.uuid not in self.discovered_explored and obj.visible:
                    # print("uuid : ", obj.uuid)
                    self.discovered_explored[obj.uuid] = {0: obj.position}
                    self.discovered_objects.append(obj.__dict__)
                    self.discovered_objects[-1]['locationParent'] = None
                    self.discovered_objects[-1]['explored'] = 0
                    self.discovered_objects[-1]['openable'] = None
                    #self.discovered_objects[-1]['agent_position'] = None
            '''
            '''
            Need to change below statements to 0,0,0 for local coordinates
            and obj_mask should either come from level 2 or Jay's code

            Oracle data being used (eval 3 )
            '''
        
            if self.level == "oracle":# or self.level == "level1" or self.level =="level2":
                position = self.event.position
                rotation = math.radians(self.event.rotation)
                #rotation = self.event.rotation
                tilt = self.event.head_tilt
                self.pose_estimate =np.array([float(position['x']),float(position['z']),rotation]).reshape(3, 1)
               
                for elem in self.event.object_list:
                    if self.event.goal.metadata['target']['id'] == elem.uuid :
                        self.goal_object = elem

                dimensions = self.goal_object.dimensions
                bd_point = set()
                for i in range(0, 8):
                    x, z = dimensions[i]['x'], dimensions[i]['z']
                    if (x, z) not in bd_point:
                        x = x - position['x']
                        z = z - position['z']
                        new_pt_x = (x * math.cos(rotation)) - (z * math.sin(rotation)) 
                        new_pt_z = (x * math.sin(rotation)) + (z * math.cos(rotation))
                        bd_point.add((new_pt_x, new_pt_z))
                        #bd_point.add((x,z))

                poly = MultiPoint(sorted(bd_point)).convex_hull
                x_list, z_list = poly.exterior.coords.xy
                self.goal_bounding_box = ObstaclePolygon(x_list, z_list)

            self.step_output = self.event
            if self.level == "level1" or self.level == "level2":
                self.img_channels = self.prediction_level1()
            '''
            Level 2 code :
                if  at level 2 or oracle :
                    do the following
            '''
            if self.level == "level2" or self.level == "oracle":
                arr_mask = np.array(self.event.object_mask_list[-1])
                self.obj_mask = arr_mask.reshape(-1, arr_mask.shape[-1])
                #print ("shape of obj mask", self.obj_mask.shape)

            if self.level == "level1":
                self.obj_mask = self.img_channels['net-mask'].flatten()
                #print ("shape of obj mask", self.obj_mask.shape)

            '''
            Local coordinate system init 
            '''
            #self.obj_mask = img_channels['net-mask']
            self.camera_height = self.event.camera_height
            #print ("starting tilt", self.event.head_tilt)
            #self.pose_estimate = np.array([0,0,math.radians(self.event.rotation)]).reshape(3,1)
            self.pose_estimate = np.array([0.0,0.0,0.0]).reshape(3,1)
            self.position = {'x': self.pose_estimate[0][0],'y': self.camera_height, 'z':self.pose_estimate[1][0]}
            self.rotation = math.degrees(self.pose_estimate[2][0])
            self.head_tilt = self.event.head_tilt
            #print ("pose estimate from dead reckoning : ", self.position, self.rotation)
            #print ("Actual pose estimate from simulat : ", self.event.position, self.event.rotation)
            bounding_boxes,current_frame_occupancy_points = convert_observation(self,self.number_actions,self.position,self.rotation) 
            if self.level == "oracle":
                #print ("in oracle mode in reset")
                self.create_current_frame_obstacles(current_frame_occupancy_points)
                self.current_frame_obstacles = self.merge_obstacles(self.current_frame_obstacles,"current")
                self.update_goal_data_oracle()
                self.update_global_obstacles()
                self.global_obstacles = self.merge_obstacles(self.global_obstacles,"global")
                #self.merge_global_obstacles()
            elif self.level == "level1" or self.level == "level2":
                #print ("in level1_2 mode in reset")
                self.calculate_img_obstacles()
                self.create_current_frame_obstacles_level_1_2(current_frame_occupancy_points)
                self.current_frame_obstacles = self.merge_obstacles(self.current_frame_obstacles,"current")
                self.update_global_obstacles_level_1_2()
                self.global_obstacles = self.merge_obstacles(self.global_obstacles,"global")
                #self.merge_global_obstacles()
                self.update_goal_object_from_obstacle_prob()
            self.add_obstacle_func(bounding_boxes)
            lastActionSuccess = self.event.return_status

        self.process_frame()
        self.board = None
        #print ("end of reset in game state function")

    def step(self, action_or_ind):
        # print("MS 3 step ", action_or_ind)
        #print ("game state head tilt",self.head_tilt)
        #print ("event head tilt", self.event.head_tilt)
        self.new_found_objects = []
        self.new_object_found = False
        if self.level != "oracle":
            self.goal_ids = []  
        self.trophy_visible_current_frame = False
        #if type(action_or_ind) == int:
        #    action = self.action_util.actions[action_or_ind]
        #else:
        action = action_or_ind
        t_start = time.time()

        #print (action)
        # The object nearest the center of the screen is open/closed if none is provided.
        vel = 0
        ang_rate = 0

        if action['action'] == 'RotateRight':
            action = "RotateLeft"
            ang_rate = math.radians(float(-10.0))
        elif action['action'] == 'RotateLeft':
            action = "RotateRight"
            ang_rate = math.radians(float(10.0))
        elif action['action'] == 'LookDown':
            action = "LookDown"
        elif action['action'] == 'LookUp':
            action = "LookUp"
        elif action['action'] == 'MoveAhead':
            vel = 0.1
            action =  'MoveAhead'
        elif action['action'] == 'MoveBack':
            vel = -0.1
            action =  'MoveBack'
        elif action['action'] == 'MoveLeft':
            action =  'MoveLeft'
        elif action['action'] == 'MoveRight':
            action =  'MoveRight'
        elif action['action'] == 'OpenObject':
            #action = "OpenObject,objectId="+ str(action["objectId"])
            #print ("constructed action for open object", action)
            action = "OpenObject,objectImageCoordsX="+str(int(action['x']))+",objectImageCoordsY="+str(int(action['y']))
        
        elif action['action'] == 'PickupObject':
            #action = "PickupObject,objectId=" + str(action['objectId'])
            action = "PickupObject,objectImageCoordsX="+str(int(action['x']))+",objectImageCoordsY="+str(int(action['y']))
        elif action['action'] == "DropObject" :
            action = "DropObject" 
        elif action['action'] == 'PickupObject':
            action = "PickupObject,objectId=" + str(action['objectId'])

        '''
        '''
        #print (action)
        #print ("rewards", self.event.reward)
        end_time_1 = time.time()
        action_creation_time = end_time_1 - t_start
        #print ("action creating time",action_creation_time)

        start_2 = time.time()
        self.event = self.env.step(action=action)
        if self.collector:
            try:
                # print("calling save_frame() in game_state.py!")
                self.collector.save_frame(self.event)
            except Exception as e:
                print("EEException during save_frame(): ", e)
                pass
        end_2 = time.time()
        action_time = end_2-start_2
        # print(f"MS action time {action_time}")

        if self.level == "oracle" :#or self.level == "level1" or self.level == "level2":
            for elem in self.event.object_list:
                if self.event.goal.metadata['target']['id'] == elem.uuid :
                    self.goal_object = elem
                    #self.goal_object_visible = elem.visible 

        if self.level == "level1" or self.level == "level2":
            self.img_seg_occupancy_map_points = {}
            self.img_channels = self.prediction_level1()
            #print ("obj classcore in step", self.img_channels['obj_class_score'])


        '''
        Level 2 code :
            if  at level 2 or oracle :
                do the following
        '''

        agent_movement = np.array([vel, ang_rate],dtype=np.float64).reshape(2, 1)
        if self.event.return_status != "OBSTRUCTED":
            self.pose_estimate = self.motion_model(self.pose_estimate,agent_movement) 
        self.step_output = self.event

        if self.level == "level2" or self.level == "oracle":
            arr_mask = np.array(self.event.object_mask_list[-1])
            self.obj_mask = arr_mask.reshape(-1, arr_mask.shape[-1])

        if self.level == "level1":
            self.obj_mask = self.img_channels['net-mask'].flatten()
            #print ("shape of obj mask", self.obj_mask.shape)


        self.position = {'x': self.pose_estimate[0][0], 'y': self.camera_height, 'z':self.pose_estimate[1][0]}
        self.rotation = math.degrees(self.pose_estimate[2][0])
        self.head_tilt = self.event.head_tilt
        start_time = time.time()
        bounding_boxes,current_frame_occupancy_points = convert_observation(self,self.number_actions,self.position,self.rotation) 
        self.number_actions += 1

        #print ("after updating and getting the steps img channel data")

        if self.level == "oracle":
            #print ("in oracle mode in step")
            self.create_current_frame_obstacles(current_frame_occupancy_points)
            self.current_frame_obstacles = self.merge_obstacles(self.current_frame_obstacles,"current")
            self.update_goal_data_oracle()
            self.update_global_obstacles()
            self.global_obstacles = self.merge_obstacles(self.global_obstacles,"global")
            #self.merge_global_obstacles()
        elif self.level == "level1" or self.level == "level2":
            self.goals_found = False
            self.calculate_img_obstacles()
            self.create_current_frame_obstacles_level_1_2(current_frame_occupancy_points)
            self.current_frame_obstacles = self.merge_obstacles(self.current_frame_obstacles,"current")
            self.update_global_obstacles_level_1_2()
            self.global_obstacles = self.merge_obstacles(self.global_obstacles,"global")
            self.update_goal_object_from_obstacle_prob()
        #print ("time taken to update global obstacle list", time.time()-start_time)
        self.add_obstacle_func(bounding_boxes)
        
        if self.event.return_status :
            self.process_frame()
        else :
            pass
            #print ("Failed status : ",self.event.return_status )

        #for obstacle1 in self.current_frame_obstacles:
        #    for obstacle2 in self.current_frame_obstacles:  
        #        if obstacle1.id == obstacle2.id :
        #            continue
        #        '''
        #        object inside container
        #        '''
        #        if obstacle1.get_bounding_box().convex_hull.contains(obstacle2.get_bounding_box()) :
       #             print ("one object containing the other inside a single frame")
       #             continue
        SHOW_ANIMATION = False
        if SHOW_ANIMATION:
            plt.cla()
            plt.xlim((-7, 7))
            plt.ylim((-7, 7))
            plt.gca().set_xlim((-7, 7))
            plt.gca().set_ylim((-7, 7))

            curr_frame_obstacles = True
            global_obstacles = True
            curr_frame_img_obstacles = False
    
            if global_obstacles :
                for obstacle in self.global_obstacles:
                    if obstacle.id in self.goal_ids :
                        patch1 = PolygonPatch(obstacle.get_bounding_box(), fc='red', ec="black", alpha=0.2, zorder=1)
                    else :
                        patch1 = PolygonPatch(obstacle.get_bounding_box(), fc='green', ec="black", alpha=0.2, zorder=1)
                    plt.gca().add_patch(patch1)

            if curr_frame_img_obstacles == True :
                for obstacle in self.current_frame_img_obstacles:
                    patch1 = PolygonPatch(obstacle.get_bounding_box(), fc='blue', ec="black", alpha=0.2, zorder=1)
                    plt.gca().add_patch(patch1)

            if curr_frame_obstacles == True:
                for obstacle in self.current_frame_obstacles:
                    patch1 = PolygonPatch(obstacle.get_bounding_box(), fc='blue', ec="black", alpha=0.2, zorder=1)
                    plt.gca().add_patch(patch1)
            if self.goal_bounding_box != None :
                patch1 = PolygonPatch(self.goal_bounding_box, fc='yellow', ec="black", alpha=0.2, zorder=1)
                plt.gca().add_patch(patch1)
            #if self.trophy_obstacle != None :
            #    patch1 = PolygonPatch(self.trophy_obstacle.get_bounding_box(),fc='red',ec="black", alpha=0.2, zorder=1)

            plt.axis("equal")
            plt.pause(0.001)
            #plt.show()

        self.trophy_location = None

    def update_global_obstacles(self):
        #for key,values in self.current_frame_obstacles.items():

        for curr_frame_obstacle in self.current_frame_obstacles:
            flag = 0 
            #obj_occ_map = get_occupancy_from_points( values,self.occupancy_map.shape)   
            #obj_polygon = polygon_simplify(occupancy_to_polygons(obj_occ_map,self.grid_size,self.displacement ))
            #print (values)
            for i,obstacle in enumerate(self.global_obstacles) :
                containment = False

                '''
                if obstacle.get_bounding_box().contains(curr_frame_obstacle.get_bounding_box()) and \
                    curr_frame_obstacle.is_goal == True and obstacle.is_goal != True :
                    print ("contains is true for some obstacle and trophy most likely")
                    if self.goals_found != True :
    
                        #print ("in contains making a separate object ")
                        #print ("occupancy map size b4", len(curr_frame_obstacle.get_occupancy_map_points()))
                        curr_frame_obstacle.expand_obstacle(obstacle.get_occupancy_map_points(),self.occupancy_map.shape,self.grid_size,self.displacement)
                        curr_frame_obstacle.id = self.objs
                        self.global_obstacles.append(copy.deepcopy(curr_frame_obstacle))
                        #print ("occupancy map after b4", len(curr_frame_obstacle.get_occupancy_map_points()))
                        #print ("occupancy map size after", len(self.global_obstacles[-1].get_occupancy_map_points()))
                        if self.global_obstacles[-1].is_goal == True :
                            self.goal_ids = [self.objs]
                            #print ("goal obj id being set to(in contains) = ", self.goal_id)
                            self.goals_found = True
                        self.objs += 1
                        self.parent_id = obstacle.id 
                        #print ("current frame Id of the goal", curr_frame_obstacle.current_frame_id)
                    else :
                        for i,obstacle in enumerate(self.global_obstacles) :
                            if obstacle.id == self.goal_ids[0] :
                                self.global_obstacles[i].current_frame_id = curr_frame_obstacle.current_frame_id
                    flag = 1
                    break
                '''
                if obstacle.get_bounding_box().contains(curr_frame_obstacle.get_bounding_box()):
                    if curr_frame_obstacle.is_contained == False:
                        #print ("containment = True in update global obstacles")
                        containment = True
                    else :
                        curr_frame_obstacle.parent_bounding_box = obstacle.get_bounding_box()
                        continue

                intersect_area = curr_frame_obstacle.get_bounding_box().intersection(obstacle.get_bounding_box()).area
                height_ratio = abs(curr_frame_obstacle.height-obstacle.height ) / (max(obstacle.height,curr_frame_obstacle.height))
                if (intersect_area > 0.00001 and height_ratio < 0.7) or containment:
                    self.global_obstacles[i].expand_obstacle(curr_frame_obstacle.get_occupancy_map_points(),self.occupancy_map.shape,self.grid_size,self.displacement)
                    self.global_obstacles[i].current_frame_id = curr_frame_obstacle.current_frame_id
                    self.global_obstacles[i].is_goal =  curr_frame_obstacle.is_goal or self.global_obstacles[i].is_goal
                    self.global_obstacles[i].set_height(max(self.global_obstacles[i].height,curr_frame_obstacle.height))
                    if self.global_obstacles[i].is_goal :
                        #print ("goal obj id being set to(in intersect) = ", self.goal_id)
                        self.goal_ids = [self.global_obstacles[i].id]
                        self.goals_found = True
                    flag = 1
                    break
            if flag == 0 :
                #self.global_obstacles.append(Obstacle(self.objs, values,self.occupancy_map.shape,self.grid_size,self.displacement))
                self.global_obstacles.append(copy.deepcopy(curr_frame_obstacle))
                self.global_obstacles[-1].id = self.objs
                #print ("obj height being added" ,curr_frame_obstacle.height)
                if self.global_obstacles[-1].is_goal == True :
                    self.goal_ids = [self.objs]
                    #print ("goal obj id being set to(in non intersect) = ", self.goal_id)
                    self.goals_found = True
                self.objs += 1

    def merge_global_obstacles(self):
        #print ("in merge before ", len(self.global_obstacles))
        elem_to_pop = []
        for i,obstacle1 in enumerate(self.global_obstacles):
            if obstacle1 in elem_to_pop :
                continue
            
            for j in range(i+1,len(self.global_obstacles)):
                obstacle2 = self.global_obstacles[j]
                if obstacle2.get_bounding_box().contains(obstacle1.get_bounding_box()) or obstacle1.get_bounding_box().contains(obstacle2.get_bounding_box()) :
                    if obstacle1.is_goal == True or obstacle2.is_goal == True: 
                        break
                intersect_area = obstacle1.get_bounding_box().intersection(obstacle2.get_bounding_box()).area
                height_ratio = abs(obstacle1.height-obstacle2.height ) / (max(obstacle1.height,obstacle2.height))
                if intersect_area > 0.00001 and height_ratio < 0.7:
                    obstacle1.expand_obstacle(obstacle2.get_occupancy_map_points(),self.occupancy_map.shape,self.grid_size,self.displacement)
                    obstacle1.set_height(max(obstacle1.height,obstacle2.height))
                    if len(obstacle2.trophy_prob_per_frame) != 0 : 
                        for trophy_prob in obstacle2.trophy_prob_per_frame :
                                self.global_obstacles[i].trophy_prob_per_frame.append(trophy_prob)  

                        self.global_obstacles[i].calculate_trophy_prob()
                        '''
                        if self.current_frame_obstacles[i].trophy_prob > 0.1 :
                            self.current_frame_obstacles[i].is_goal = True
                            self.goals_found = True
                        '''
                    elem_to_pop.append(obstacle2) 
            
        for elem in elem_to_pop:
            #print ("global merge happening ID being popped",elem.id)
            self.global_obstacles.remove(elem)        


    def merge_obstacles(self,obstacle_list,obs_list_scope):
        #print ("in merge before ", len(self.global_obstacles))
        elem_to_pop = []
        for i,obstacle1 in enumerate(obstacle_list):
            if obstacle1 in elem_to_pop :
                continue
            
            for j in range(0,len(obstacle_list)):
                obstacle2 = obstacle_list[j]
                if obstacle1.id == obstacle2.id :   
                    continue
                containment = False
                height_ratio =abs(obstacle1.height-obstacle2.height ) / (max(obstacle1.height,obstacle2.height))
    
                '''
                if self.goals_found == True : 

                    print ("obs_list_scope ", obs_list_scope)
                    print ("when goals found, normal containment", obstacle1.get_bounding_box().contains(obstacle2.get_bounding_box()))
                    print ("height ratio", height_ratio)
                '''
                intersect_area = obstacle1.get_bounding_box().intersection(obstacle2.get_bounding_box()).area

                #if intersect_area > 0.0001 :
                #    if obstacle1.is_contained == True and obstacle2.is_contained == True :
                #        break

                if obstacle1.get_bounding_box().contains(obstacle2.get_bounding_box()) :
                    if obstacle2.is_contained == False:
                        containment = True
                        #print ("containment = True, one frame is contained is false")
                        
                    else :
                        obstacle2.parent_bounding_box = obstacle1.get_bounding_box()
                        continue

                #if containment == True :
                #    print ("intersect area,height_ratio ", intersect_area,height_ratio)
                
                if (intersect_area > 0.00001 and height_ratio < 0.7) or containment == True:
                    obstacle1.expand_obstacle(obstacle2.get_occupancy_map_points(),self.occupancy_map.shape,self.grid_size,self.displacement)
                    obstacle1.set_height(max(obstacle1.height,obstacle2.height))
                    obstacle1.is_goal = obstacle1.is_goal or obstacle2.is_goal
                    obstacle1.is_contained = obstacle1.is_contained or obstacle2.is_contained
                    if len(obstacle2.trophy_prob_per_frame) != 0 : 
                        for trophy_prob in obstacle2.trophy_prob_per_frame :
                            #self.global_obstacles[i].trophy_prob_per_frame.append(trophy_prob)  
                            obstacle1.trophy_prob_per_frame.append(trophy_prob)
                        obstacle1.calculate_trophy_prob()
                        #self.global_obstacles[i].calculate_trophy_prob()
                    if obstacle2 not in elem_to_pop:
                        elem_to_pop.append(obstacle2) 
                    continue
    

                if obs_list_scope == "current" :
                    #print ("obs list scope :", obs_list_scope)
                    #print ("height ratio ",height_ratio)
                    #print ("Convex containment" ,obstacle1.get_bounding_box().convex_hull.contains(obstacle2.get_bounding_box()) )                
                    #print ("normal containment", containment)

                    if obstacle1.get_bounding_box().convex_hull.contains(obstacle2.get_bounding_box()) \
                        and containment == False and height_ratio < 0.89:
                        #print ("obstacle height")
                        if obstacle1.height > 1.4 : 
                            continue                
                        #obstacle1_poly_exterior_coords = obstacle1.get_convex_polygon_coords()
                        #obstacle1_exterior_poly = Polygon(obstacle1_poly_exterior_coords)
                        #if obstacle1_exterior_poly.contains(obstacle2.get_bounding_box()) \
                        obstacle2.is_contained = True
                        #print ("is contained is true")
                        obstacle2.parent_bounding_box = obstacle1.get_bounding_box().convex_hull
                        SHOW_ANIMATION = (not containment and False)
                        if SHOW_ANIMATION:
                            plt.cla()
                            plt.xlim((-7, 7))
                            plt.ylim((-7, 7))
                            plt.gca().set_xlim((-7, 7))
                            plt.gca().set_ylim((-7, 7))

                            #for obstacle in self.current_frame_obstacles:
                            patch1 = PolygonPatch(obstacle1.get_bounding_box().convex_hull, fc='green', ec="black", alpha=0.2, zorder=1)
                            plt.gca().add_patch(patch1)
                            patch1 = PolygonPatch(obstacle2.get_bounding_box(), fc='red', ec="black", alpha=0.2, zorder=1)
                            plt.gca().add_patch(patch1)
                            plt.axis("equal")
                            plt.pause(0.001)
                            plt.show()
            
        for elem in elem_to_pop:
            #print ("merginng obstacles happening ID being popped",elem.id)
            obstacle_list.remove(elem)        


        return obstacle_list[:]

    def create_current_frame_obstacles(self, current_frame_obstacles_dict):
        def intersect2D(a, b):
            return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])
        obj_id =  1000
        i = 0
        self.current_frame_obstacles = []
        goal_index = -1

        structure = np.ones((3, 3), dtype=np.int)
        current_frame_obstacles_dict_divided = {}
        for key,values in current_frame_obstacles_dict.items():
            #print ("per obj ")
            obj_heights = values[1]
            obj_occ_map_points = values[0]

            new_occ_map = np.zeros(self.occupancy_map.shape)
            new_height_map = np.zeros(self.occupancy_map.shape)
            k = 0

            for position in obj_occ_map_points :
                new_occ_map[position[0]][position[1]] = 1     
                new_height_map[position[0]][position[1]] = obj_heights[k]     
                k += 1
            
            labeled,n_components = label(new_occ_map, structure)     

            #print (n_components,np.amax(labeled))

            for i in range(1,n_components+1):
                object_occ_map = np.where(labeled==i)
                new_map_points = np.array([np.array([x,y]) for x,y in zip(object_occ_map[0],object_occ_map[1])] )
                obj_height = np.amax(new_height_map[object_occ_map])
                current_frame_obstacles_dict_divided[obj_id] = (new_map_points,obj_height,key)
                obj_id += 1

        for key,values in current_frame_obstacles_dict_divided.items():
            obj_height = values[1]
            obj_occ_map_points = values[0]
            curr_frame_id = values[2]
            self.current_frame_obstacles.append(Obstacle(key,obj_height,  obj_occ_map_points,self.occupancy_map.shape,self.grid_size,self.displacement))
            self.current_frame_obstacles[-1].current_frame_id = curr_frame_id
            #obj_id += 1


    def update_goal_data_oracle(self):
        i = 0
    
        if not self.goal_object_visible :
            return 
        max_intersect_area = 0.001
        goal_index = -1

        for current_frame_obstacle in self.current_frame_obstacles:
            intersect_area = current_frame_obstacle.get_bounding_box().intersection(self.goal_bounding_box).area
            #intersect_area = self.current_frame_obstacles[-1].get_bounding_box().intersection(self.trophy_obstacle.get_bounding_box()).area
            #print ("Intersection area : ", intersect_area)
            if intersect_area > max_intersect_area :
                goal_index = i
                self.goal_calculated_points = current_frame_obstacle.get_bounding_box() 
                max_intersect_area = intersect_area
            i += 1

        #This is disabled when we do not want to recognize trophy(disabled only for testing containers)
        if goal_index != -1:
            #print ("goal found")
            self.current_frame_obstacles[goal_index].is_goal = True

            #print ("number of obstacles seen in this frame", len(self.current_frame_obstacles))

            SHOW_ANIMATION = False
            if SHOW_ANIMATION:
                plt.cla()
                plt.xlim((-7, 7))
                plt.ylim((-7, 7))
                plt.gca().set_xlim((-7, 7))
                plt.gca().set_ylim((-7, 7))

                #for obstacle in self.current_frame_obstacles:
                for obstacle in self.global_obstacles:
                    #obstacle_convex = Polygon(obstacle.get_convex_polygon_coords())
                    patch1 = PolygonPatch(obstacle.get_bounding_box(), fc='green', ec="black", alpha=0.2, zorder=1)
                    plt.gca().add_patch(patch1)
                patch1 = PolygonPatch(self.current_frame_obstacles[goal_index].get_bounding_box(), fc='red', ec="black", alpha=0.2, zorder=1)
                plt.gca().add_patch(patch1)
                #patch1 = PolygonPatch(curr_frame_obstacle.get_bounding_box(), fc='red', ec="black", alpha=0.2, zorder=1)
                #plt.gca().add_patch(patch1)
                plt.axis("equal")
                #plt.pause(0.001)
                plt.show()

        '''
        for obstacle1 in self.current_frame_obstacles:
            for obstacle2 in self.current_frame_obstacles:  
                if obstacle1.id == obstacle2.id :
                    continue
                #if obstacle1.get_bounding_box().convex_hull.contains(obstacle2.get_bounding_box()) :
                if obstacle1.get_bounding_box().contains(obstacle2.get_bounding_box()) :
                    print ("one object containing the other inside a single frame")
                    continue
                    parent_gc = obstacle1.get_occupancy_map_points()
                    #parent_gc = np.array([tuple(elem[0],elem[1]) for elem in parent_gc])
                    child_gc = obstacle2.get_occupancy_map_points()
                    #child_gc = np.array([tuple(elem[0],elem[1]) for elem in child_gc])
                    #print ("grid cells by parent size", len(parent_gc))
                    #print ("grid cells by child", len(child_gc))
                    #print ("gc intersection size", len(intersect2D(parent_gc,child_gc)))
                    #print ("gc intersection prop", len(intersect2D(parent_gc,child_gc))/len(child_gc))
                    #print ("area proportion", curr_frame_obstacle.get_bounding_box().area/obstacle.get_bounding_box().area)
        '''
        #print ("current frame obstacles size", len(self.current_frame_obstacles))

    def create_current_frame_obstacles_level_1(self, current_frame_obstacles_dict):
        #print ("in calculate current obstacle level 1_2 ")
        obj_id =  1000
        max_intersect_area = 0.001
        i = 0
        self.current_frame_obstacles = []
        goal_index = -1
        for key,values in current_frame_obstacles_dict.items():
            obj_height = values[1]
            obj_occ_map_points = values[0]
            self.current_frame_obstacles.append(Obstacle(obj_id,obj_height,  obj_occ_map_points,self.occupancy_map.shape,self.grid_size,self.displacement))
            self.current_frame_obstacles[-1].current_frame_id = key


            #print ("len of current img frame obs", len(self.current_frame_img_obstacles))
            for img_obs in self.current_frame_img_obstacles : 

                intersect_area = self.current_frame_obstacles[-1].get_bounding_box().intersection(img_obs.get_bounding_box()).area
                #print ("Intersection area : ", intersect_area)
                if intersect_area > 0.0001 :
                    #print (" Has to come here exactly once every frame")
                    self.current_frame_obstacles[-1].trophy_prob_per_frame.append(img_obs.trophy_prob_per_frame[0])
                    self.current_frame_obstacles[-1].calculate_trophy_prob()
                    #if self.current_frame_obstacles[-1].trophy_prob > 0.1 :
                    #    self.current_frame_obstacles[-1].is_goal = True
            obj_id += 1

        SHOW_ANIMATION = False
        if SHOW_ANIMATION:
            plt.cla()
            plt.xlim((-7, 7))
            plt.ylim((-7, 7))
            plt.gca().set_xlim((-7, 7))
            plt.gca().set_ylim((-7, 7))

            for obstacle in self.current_frame_obstacles:
                #obstacle_convex = Polygon(obstacle.get_convex_polygon_coords())
                patch1 = PolygonPatch(obstacle.get_bounding_box().convex_hull, fc='green', ec="black", alpha=0.2, zorder=1)
                #patch1 = PolygonPatch(obstacle_convex, fc='green', ec="black", alpha=0.2, zorder=1)
                plt.gca().add_patch(patch1)
            #patch1 = PolygonPatch(curr_frame_obstacle.get_bounding_box(), fc='red', ec="black", alpha=0.2, zorder=1)
            #plt.gca().add_patch(patch1)
            plt.axis("equal")
            plt.pause(0.001)
            #plt.show()

        '''
        for obstacle1 in self.current_frame_obstacles:
            for obstacle2 in self.current_frame_obstacles:  
                if obstacle1.id == obstacle2.id :
                    continue
                if obstacle1.get_bounding_box().convex_hull.contains(obstacle2.get_bounding_box()) :
                    print ("one object containing the other inside a single frame")
                    continue
                    parent_gc = obstacle1.get_occupancy_map_points()
                    #parent_gc = np.array([tuple(elem[0],elem[1]) for elem in parent_gc])
                    child_gc = obstacle2.get_occupancy_map_points()
                    #child_gc = np.array([tuple(elem[0],elem[1]) for elem in child_gc])
                    #print ("grid cells by parent size", len(parent_gc))
                    #print ("grid cells by child", len(child_gc))
                    #print ("gc intersection size", len(intersect2D(parent_gc,child_gc)))
                    #print ("gc intersection prop", len(intersect2D(parent_gc,child_gc))/len(child_gc))
                    #print ("area proportion", curr_frame_obstacle.get_bounding_box().area/obstacle.get_bounding_box().area)
        '''
    def create_current_frame_obstacles_level_1_2(self, current_frame_obstacles_dict):
        #print ("in calculate current obstacle level 1_2 ")
        obj_id =  1000
        max_intersect_area = 0.001
        i = 0
        self.current_frame_obstacles = []
        goal_index = -1        
        structure = np.ones((3, 3), dtype=np.int)
        current_frame_obstacles_dict_divided = {}
        for key,values in current_frame_obstacles_dict.items():
            #print ("per obj ")
            obj_heights = values[1]
            obj_occ_map_points = values[0]

            new_occ_map = np.zeros(self.occupancy_map.shape)
            new_height_map = np.zeros(self.occupancy_map.shape)
            k = 0

            for position in obj_occ_map_points :
                new_occ_map[position[0]][position[1]] = 1     
                new_height_map[position[0]][position[1]] = obj_heights[k]     
                k += 1
            
            labeled,n_components = label(new_occ_map, structure)     

            #print (n_components,np.amax(labeled))

            for i in range(1,n_components+1):
                object_occ_map = np.where(labeled==i)
                new_map_points = np.array([np.array([x,y]) for x,y in zip(object_occ_map[0],object_occ_map[1])] )
                obj_height = np.amax(new_height_map[object_occ_map])
                current_frame_obstacles_dict_divided[obj_id] = (new_map_points,obj_height,key)
                obj_id += 1
            

        #for key,values in current_frame_obstacles_dict.items():
        for key,values in current_frame_obstacles_dict_divided.items():
            obj_height = values[1]
            obj_occ_map_points = values[0]
            curr_frame_id = values[2]
            self.current_frame_obstacles.append(Obstacle(key,obj_height,  obj_occ_map_points,self.occupancy_map.shape,self.grid_size,self.displacement))
            self.current_frame_obstacles[-1].current_frame_id = curr_frame_id

            #print ("len of current img frame obs", len(self.current_frame_img_obstacles))
            for img_obs in self.current_frame_img_obstacles : 

                intersect_area = self.current_frame_obstacles[-1].get_bounding_box().intersection(img_obs.get_bounding_box()).area
                #print ("Intersection area : ", intersect_area)
                if intersect_area > 0.0001 :
                    #print (" Has to come here exactly once every frame")
                    self.current_frame_obstacles[-1].trophy_prob_per_frame.append(img_obs.trophy_prob_per_frame[0])
                    self.current_frame_obstacles[-1].calculate_trophy_prob()
                    #if self.current_frame_obstacles[-1].trophy_prob > 0.1 :
                    #    self.current_frame_obstacles[-1].is_goal = True
            obj_id += 1

        SHOW_ANIMATION = False
        if SHOW_ANIMATION:
            plt.cla()
            plt.xlim((-7, 7))
            plt.ylim((-7, 7))
            plt.gca().set_xlim((-7, 7))
            plt.gca().set_ylim((-7, 7))

            for obstacle in self.current_frame_obstacles:
                #obstacle_convex = Polygon(obstacle.get_convex_polygon_coords())
                patch1 = PolygonPatch(obstacle.get_bounding_box().convex_hull, fc='green', ec="black", alpha=0.2, zorder=1)
                #patch1 = PolygonPatch(obstacle_convex, fc='green', ec="black", alpha=0.2, zorder=1)
                plt.gca().add_patch(patch1)
            #patch1 = PolygonPatch(curr_frame_obstacle.get_bounding_box(), fc='red', ec="black", alpha=0.2, zorder=1)
            #plt.gca().add_patch(patch1)
            plt.axis("equal")
            plt.pause(0.001)
            #plt.show()

    def update_global_obstacles_level_1_2(self):

        #print ("in calculate global obstacle level 1_2 ")
        #for key,values in self.current_frame_obstacles.items():
        for curr_frame_obstacle in self.current_frame_obstacles:
            flag = 0 
            for i,obstacle in enumerate(self.global_obstacles) :
                '''
                if obstacle.get_bounding_box().contains(curr_frame_obstacle.get_bounding_box()) and \
                    curr_frame_obstacle.is_goal == True and obstacle.is_goal != True :

                        if self.global_obstacles[-1].is_goal == True :
                            self.goal_id = self.objs
                            #print ("goal obj id being set to(in contains) = ", self.goal_id)
                            self.goals_found = True
                        self.objs += 1
                        self.parent_id = obstacle.id 
                        #print ("current frame Id of the goal", curr_frame_obstacle.current_frame_id)
                    else :
                        for i,obstacle in enumerate(self.global_obstacles) :
                            if obstacle.id == self.goal_id :
                                self.global_obstacles[i].current_frame_id = curr_frame_obstacle.current_frame_id
                    flag = 1
                    break
                '''
                if obstacle.get_bounding_box().contains(curr_frame_obstacle.get_bounding_box()):
                    if curr_frame_obstacle.is_contained == False:
                        containment = True
                    else :
                        continue
                intersect_area = curr_frame_obstacle.get_bounding_box().intersection(obstacle.get_bounding_box()).area
                height_ratio = abs(curr_frame_obstacle.height-obstacle.height ) / (max(obstacle.height,curr_frame_obstacle.height))
                if intersect_area > 0.00001 and height_ratio < 0.7:
                #print ("Intersection area : ", intersect_area)
                #if intersect_area > 0.00001 :
                    self.global_obstacles[i].expand_obstacle(curr_frame_obstacle.get_occupancy_map_points(),self.occupancy_map.shape,self.grid_size,self.displacement)
                    self.global_obstacles[i].current_frame_id = curr_frame_obstacle.current_frame_id
                    #self.global_obstacles[i].is_goal =  curr_frame_obstacle.is_goal
                    self.global_obstacles[i].set_height(max(self.global_obstacles[i].height,curr_frame_obstacle.height))
                    self.global_obstacles[i].current_frame_id = curr_frame_obstacle.current_frame_id
                    if curr_frame_obstacle.trophy_prob != 0 :
                        #print ("curr frame obstacle prob not zero ", curr_frame_obstacle.trophy_prob)#, curr_frame_obstacle.get_centre())
                        for trophy_prob in curr_frame_obstacle.trophy_prob_per_frame :
                            self.global_obstacles[i].trophy_prob_per_frame.append(trophy_prob)  

                        self.global_obstacles[i].calculate_trophy_prob()
                        #print ("updated global obstacle prob", self.global_obstacles[i].trophy_prob)#, self.global_obstacles[i].get_centre())
                        '''
                        if self.global_obstacles[i].trophy_prob > 0.3 :
                            print ("found trophy finally",self.global_obstacles[i].id)
                            self.global_obstacles[i].is_goal = True
                            self.goals_found = True
                            self.goal_id = self.global_obstacles[i].id 
                            self.trophy_obstacle = self.global_obstacles[i]
                        '''
                    flag = 1
                    break
            if flag == 0 :
                #self.global_obstacles.append(Obstacle(self.objs, values,self.occupancy_map.shape,self.grid_size,self.displacement))
                self.global_obstacles.append(copy.deepcopy(curr_frame_obstacle))
                self.global_obstacles[-1].id = self.objs
                #print ("obj height being added" ,curr_frame_obstacle.height)
                '''
                if self.global_obstacles[-1].is_goal == True :
                    self.goal_id = self.objs
                    #print ("goal obj id being set to(in non intersect) = ", self.goal_id)
                    self.goals_found = True
                '''
                self.objs += 1

    def calculate_img_obstacles(self):
        #print ("in calculate img obstacles")
        obj_id =  500
        max_intersect_area = 0.001
        i = 0
        obj_class_score = self.img_channels['obj_class_score']
        #print (obj_class_score.shape)
        self.current_frame_img_obstacles = []
        for key,values in self.img_seg_occupancy_map_points.items():
            #print ("object location" , key)
            obj_height = values[1]
            obj_occ_map_points = values[0]
            min_prob,max_prob = values[2]
            #max_prob = np.amax(self.img_channels['mask_prob'][key+4])
            #print ("max prob for this mask" , max_prob)
            trophy_prob = obj_class_score[key][2] * max_prob
            self.current_frame_img_obstacles.append(Obstacle(obj_id,obj_height,  obj_occ_map_points,self.occupancy_map.shape,self.grid_size,self.displacement,trophy_prob))
            #print ("curr img seg prob ", self.current_frame_img_obstacles[-1].trophy_prob)#, self.current_frame_img_obstacles[-1].get_centre())
            self.current_frame_img_obstacles[-1].current_frame_id = obj_id
            obj_id += 1
            i += 1 
        SHOW_ANIMATION = False
        if SHOW_ANIMATION:
            plt.cla()
            plt.xlim((-7, 7))
            plt.ylim((-7, 7))
            plt.gca().set_xlim((-7, 7))
            plt.gca().set_ylim((-7, 7))

            for obstacle in self.current_frame_img_obstacles:
                #patch1 = PolygonPatch(self.current_frame_img_obstacles[-1].get_bounding_box(), fc='green', ec="black", alpha=0.2, zorder=1)
                patch1 = PolygonPatch(obstacle.get_bounding_box(), fc='green', ec="black", alpha=0.2, zorder=1)
                plt.gca().add_patch(patch1)

            plt.axis("equal")
            plt.pause(0.1)
        
    def update_goal_object_from_obstacle_prob(self):
        for i,obstacle in enumerate(self.global_obstacles) :
            self.global_obstacles[i].calculate_trophy_prob() 
            if self.global_obstacles[i].is_possible_trophy():
                if self.global_obstacles[i].trophy_prob > self.trophy_prob_threshold :
                    self.global_obstacles[i].is_goal = True
                    self.goals_found = True
                    self.goal_ids.append(self.global_obstacles[i].id)
  
    def prediction_level1(self):
        rgbI = np.array(self.event.image_list[-1])
        bgrI = rgbI[:, :, [2,1,0]]
        depthI = np.uint8(self.step_output.depth_map_list[-1] / self.step_output.camera_clipping_planes[1] * 255)

        ret = self.mask_predictor.step(bgrI, depthI)
        return ret
        '''
        cls = np.argmax(ret['obj_class_score'], axis=1)
        print("all object scores : ", ret["obj_class_score"])
        n_th_obj = None
        if TROPHY_INDEX in cls:
            n_th_obj = np.where(cls == TROPHY_INDEX)[0]
            print("find trophy in {}th fg object".format(n_th_obj))

        # self.debug_out(bgrI, depthI, ret)
        return ret , n_th_obj
        '''

    def motion_model(self, x, u):
        DT = 1.0
        #u = np.array([vel, ang_rate]).reshape(2, 1)
        F = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])

        B = np.array([[DT * math.sin(x[2, 0] ), 0.0],
                      [DT * math.cos(x[2, 0] ), 0.0],
                      [0.0, DT]])

        x = F @ x + B @ u

        #print ('x',x)
        #print ('B @ u', B @ u)

        #print ('x after update ',x, "\n")
        x[2, 0] = self.pi_2_pi(x[2, 0])

        return x

    def pi_2_pi(self, angle):
        #return ((angle + math.pi) % (2 * math.pi)) - math.pi
        return (angle) % (2 * math.pi)
