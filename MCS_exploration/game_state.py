import time
import random
import numpy as np

#from utils import game_util
#from utils import action_util
from MCS_exploration.utils import game_util
from MCS_exploration.utils import action_util
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
    def __init__(self, env=None, depth_scope=None):
        if env == None:
            self.env = game_util.create_env()
        else :
            self.env = env
        #print ("game state init")
        self.action_util = action_util.ActionUtil()
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
        self.map_width = 36
        self.map_length = 36                                                                     
        self.displacement = 12
        self.occupancy_map = self.occupancy_map_init() #* unexplored
        self.object_mask = None
        self.goal_id = None
        self.pose_estimate = np.zeros((3,1),dtype = np.float64)
        self.global_obstacles = []
        self.current_frame_obstacles = []
        self.objs = 0
        self.oracle_position = None
        self.position = None
        self.rotation = None
        self.head_tilt = None
        self.mask_predictor = MaskAndClassPredictor()
        self.trophy_location = None #[Trophy location in the img seg list] 
        self.trophy_mask = None
        self.trophy_obstacle = None
        self.trophy_picked_up = False
        self.mode = "oracle"
        self.trophy_visible_current_frame = False

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
            #print ("in the game state reset end point is : ", self.end_point)

        else:
            # Do full reset
            #self.world_poly = fov.FieldOfView([0, 0, 0], 0, [])
            self.world_poly = sp.Polygon()
            self.goals_found = False
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
        
            if self.mode == "oracle":
                position = self.event.position
                rotation = math.radians(self.event.rotation)
                #rotation = self.event.rotation
                tilt = self.event.head_tilt
                #self.pose_estimate =np.array([float(position['x']),float(position['z']),rotation]).reshape(3, 1)

                for elem in self.event.object_list:
                    if self.event.goal.metadata['target']['id'] == elem.uuid :
                        self.goal_object = elem
            
                #print ("goal object",self.goal_object)
                #print ("orig starting pos", position )
                #print ("starting rotation ", self.event.rotation)

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

                '''
                Level 2 code :
                    if  at level 2 or oracle :
                        do the following
                '''
                self.obj_mask = self.event.object_mask_list[-1]
            '''
            Local coordinate system init 
            '''
            self.step_output = self.event
            #img_channels, self.trophy_location = self.prediction_level1()
            #self.obj_mask = img_channels['net-mask']
            self.camera_height = self.event.camera_height
            self.pose_estimate = np.array([0.0,0.0,0.0]).reshape(3,1)
            #self.pose_estimate = np.array([0,0,math.radians(self.event.rotation)]).reshape(3,1)
            #self.pose_estimate = np.array([0,0,math.radians(0)]).reshape(3,1)
            self.position = {'x': self.pose_estimate[0][0], 'y': self.camera_height, 'z':self.pose_estimate[1][0]}
            self.rotation = math.degrees(self.pose_estimate[2][0])
            self.head_tilt = self.event.head_tilt
            bounding_boxes,current_frame_occupancy_points = convert_observation(self,self.number_actions,self.position,self.rotation) 
            self.create_current_frame_obstacles(current_frame_occupancy_points)
            self.update_global_obstacles()
            self.merge_global_obstacles()
            self.add_obstacle_func(bounding_boxes)
            #self.add_obstacle_func(self.global_obstacles)
            lastActionSuccess = self.event.return_status

        self.process_frame()
        self.board = None
        #print ("end of reset in game state function")

    def step(self, action_or_ind):
        self.new_found_objects = []
        self.new_object_found = False
        self.trophy_visible_current_frame = False
        if type(action_or_ind) == int:
            action = self.action_util.actions[action_or_ind]
        else:
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
        elif action['action'] == 'OpenObject':
            #action = "OpenObject,objectId="+ str(action["objectId"])
            #print ("constructed action for open object", action)
            action = "OpenObject,objectImageCoordsX="+str(int(action['x']))+",objectImageCoordsY="+str(int(action['y']))
        
        elif action['action'] == 'PickupObject':
            #action = "PickupObject,objectId=" + str(action['objectId'])
            action = "PickupObject,objectImageCoordsX="+str(int(action['x']))+",objectImageCoordsY="+str(int(action['y']))
        elif action['action'] == 'PickupObject':
            action = "PickupObject,objectId=" + str(action['objectId'])

        '''
        '''
        #print (action)
        end_time_1 = time.time()
        action_creation_time = end_time_1 - t_start
        #print ("action creating time",action_creation_time)

        start_2 = time.time()
        self.event = self.env.step(action=action)
        end_2 = time.time()
        action_time = end_2-start_2

        for elem in self.event.object_list:
            if self.event.goal.metadata['target']['id'] == elem.uuid :
                self.goal_object = elem
                self.goal_object_visible = elem.visible 

        #print ("action time", action_time)
        # lastActionSuccess = self.event.return_status

        #print ("self event goal visible : ", self.goal_object_visible)
        #print ("self objects" , self.event.object_list[0])
        #exit()
        agent_movement = np.array([vel, ang_rate],dtype=np.float64).reshape(2, 1)
        if self.event.return_status != "OBSTRUCTED":
            self.pose_estimate = self.motion_model(self.pose_estimate,agent_movement) 
        #else :
        #    print ("return status from step " , self.event.return_status)
        '''
        Level 2 code :
            if  at level 2 or oracle :
                do the following
        '''

        self.step_output = self.event
        #img_channels, self.trophy_location = self.prediction_level1()
        #print ("trophy position in channels", self.trophy_location)
        if self.trophy_location != None :
            #print ("channel data number of masks", img_channels['mask_prob'].shape )
            #print ("channels data mask prob",img_channels['mask_prob'][self.trophy_location[0]+4].shape)
            #print ("channel data",img_channels[trophy_location[0]+4].shape)
            print ("all class scores", img_channels['obj_class_score']) 
            self.trophy_mask = img_channels['mask_prob'][self.trophy_location[0]+4].flatten()
            #print ("flattened mask ", trophy_mask.shape)
            #trophy_mask = np.where(trophy_mask >= 0.75 , 1 ,0 ) 
            #print (len(np.where(trophy_mask!=0)[0]))
            #unique_elem,unique_indices,unique_counts = np.unique(trophy_mask,return_index=True,return_counts=True)
            #print ("number of unique elem", min(unique_elem), max(unique_elem), unique_counts)


        self.obj_mask = self.event.object_mask_list[-1]
        #self.obj_mask = img_channels['net-mask']
        self.position = {'x': self.pose_estimate[0][0], 'y': self.camera_height, 'z':self.pose_estimate[1][0]}
        self.rotation = math.degrees(self.pose_estimate[2][0])
        self.head_tilt = self.event.head_tilt
        start_time = time.time()
        #bounding_boxes = convert_observation(self,self.number_actions,self.position,self.rotation) 
        bounding_boxes,current_frame_occupancy_points = convert_observation(self,self.number_actions,self.position,self.rotation) 
        #print ("Frame processing time" , time.time()- start_time)
        #self.add_obstacle_func(self.global_obstacles)
        self.number_actions += 1

        obj_id = 10000
        if self.trophy_location != None :
            if len(self.trophy_occupancy_map_points) != 0 :
                self.trophy_obstacle = Obstacle(obj_id,0.2,self.trophy_occupancy_map_points ,self.occupancy_map.shape,self.grid_size,self.displacement)
                self.trophy_obstacle.is_goal = True
                self.trophy_visible_current_frame = True
                print ("setting current frame to true")
                #self.current_frame_obstacles.append(self.trophy_obstacle)
            else :
                self.trophy_location = None
            #print ("intersection1between ground truth bb and as seen from image predictions" , self.trophy_obstacle.get_bounding_box().intersection(self.goal_bounding_box).area)
            #print ("total area of ground truth bb", self.goal_bounding_box.area)
            #print ("total area of mask based trophy", self.trophy_obstacle.get_bounding_box().area)
            #print ("exterior coords", self.trophy_obstacle.get_bounding_box().exterior.coords.xy)

        self.create_current_frame_obstacles(current_frame_occupancy_points)
        self.update_global_obstacles()
        self.merge_global_obstacles()
        #print ("time taken to update global obstacle list", time.time()-start_time)
        self.add_obstacle_func(bounding_boxes)
        
        if self.event.return_status :
            self.process_frame()
        else :
            print ("Failed status : ",self.event.return_status )

    
        SHOW_ANIMATION = True
            

        #print (self.global_obstacles)

        if SHOW_ANIMATION:
            plt.cla()
            plt.xlim((-7, 7))
            plt.ylim((-7, 7))
            plt.gca().set_xlim((-7, 7))
            plt.gca().set_ylim((-7, 7))

            for obstacle in self.global_obstacles:
                patch1 = PolygonPatch(obstacle.get_bounding_box(), fc='green', ec="black", alpha=0.2, zorder=1)
                plt.gca().add_patch(patch1)
                centre_x,centre_y,centre_z = obstacle.get_centre()
                #plt.plot(centre_x, centre_z, "x")
            #for obstacle in [self.goal_bounding_box,self.trophy_obstacle.get_bounding_box()]:
            if self.goal_bounding_box != None :
                patch1 = PolygonPatch(self.goal_bounding_box, fc='blue', ec="black", alpha=0.2, zorder=1)
                plt.gca().add_patch(patch1)
            if self.trophy_location != None :
                patch1 = PolygonPatch(self.trophy_obstacle.get_bounding_box(),fc='red',ec="black", alpha=0.2, zorder=1)
                plt.gca().add_patch(patch1)

            '''
            fig, ax = plt.subplots(1,2)
            #ax[0,0].imshow(bgrI[..., [2,1,0]])
            #ax[0,0].set_title('RGB image')
            #ax[0,1].imshow(depthI, cmap='gray')
            #ax[0,1].set_title('depth image')
            ax[0].imshow(img_channels['net-mask'])
            ax[0].set_title('net predict mask')
            ax[1].imshow(img_channels['mask_prob'].argmax(axis=0))
            ax[1].set_title('final mask (with cls-score)')
            '''
            #plt.show()
            plt.axis("equal")
            plt.pause(0.001)
            #plt.pause(1)
        #print ("self goals found",self.goals_found)

        self.trophy_location = None

        #if len(self.goals) == 0 :
        #    self.goals_found = True

    def update_global_obstacles(self):
        #for key,values in self.current_frame_obstacles.items():
        for curr_frame_obstacle in self.current_frame_obstacles:
            flag = 0 
            #obj_occ_map = get_occupancy_from_points( values,self.occupancy_map.shape)   
            #obj_polygon = polygon_simplify(occupancy_to_polygons(obj_occ_map,self.grid_size,self.displacement ))
            #print (values)
            for i,obstacle in enumerate(self.global_obstacles) :

                if obstacle.get_bounding_box().contains(curr_frame_obstacle.get_bounding_box()) and \
                    curr_frame_obstacle.is_goal == True and obstacle.is_goal != True :
                    if self.goals_found != True :
                        #print ("in contains making a separate object ")
    
                        #print ("occupancy map size b4", len(curr_frame_obstacle.get_occupancy_map_points()))
                        curr_frame_obstacle.expand_obstacle(obstacle.get_occupancy_map_points(),self.occupancy_map.shape,self.grid_size,self.displacement)
                        curr_frame_obstacle.id = self.objs
                        self.global_obstacles.append(copy.deepcopy(curr_frame_obstacle))
                        #print ("occupancy map after b4", len(curr_frame_obstacle.get_occupancy_map_points()))
                        #print ("occupancy map size after", len(self.global_obstacles[-1].get_occupancy_map_points()))

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

                intersect_area = curr_frame_obstacle.get_bounding_box().intersection(obstacle.get_bounding_box()).area
                #print ("Intersection area : ", intersect_area)
                if intersect_area > 0.00001 :
                    self.global_obstacles[i].expand_obstacle(curr_frame_obstacle.get_occupancy_map_points(),self.occupancy_map.shape,self.grid_size,self.displacement)
                    self.global_obstacles[i].current_frame_id = curr_frame_obstacle.current_frame_id
                    self.global_obstacles[i].is_goal =  curr_frame_obstacle.is_goal
                    if self.global_obstacles[i].is_goal :
                        #print ("goal obj id being set to(in intersect) = ", self.goal_id)
                        self.goal_id = self.global_obstacles[i].id
                        self.goals_found = True
                    flag = 1
                    break
            if flag == 0 :
                #self.global_obstacles.append(Obstacle(self.objs, values,self.occupancy_map.shape,self.grid_size,self.displacement))
                self.global_obstacles.append(copy.deepcopy(curr_frame_obstacle))
                self.global_obstacles[-1].id = self.objs
                if self.global_obstacles[-1].is_goal == True :
                    self.goal_id = self.objs
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
                if intersect_area > 0.00001 :
                    obstacle1.expand_obstacle(obstacle2.get_occupancy_map_points(),self.occupancy_map.shape,self.grid_size,self.displacement)
                    obstacle1.height = max(obstacle1.height,obstacle2.height)
                    elem_to_pop.append(obstacle2) 
            
        #print ("len of elements to pop", len(elem_to_pop))
        for elem in elem_to_pop:
            self.global_obstacles.remove(elem)        
        #print ("in merge end ", len(self.global_obstacles))

    def create_current_frame_obstacles(self, current_frame_obstacles_dict):
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
            #if self.trophy_obstacle != None:
            if self.goal_object_visible:
            #if self.trophy_visible_current_frame:
            #if True:
                intersect_area = self.current_frame_obstacles[-1].get_bounding_box().intersection(self.goal_bounding_box).area
                #intersect_area = self.current_frame_obstacles[-1].get_bounding_box().intersection(self.trophy_obstacle.get_bounding_box()).area
                #print ("Intersection area : ", intersect_area)
                if intersect_area > max_intersect_area :
                    goal_index = i
                    self.goal_calculated_points = self.current_frame_obstacles[-1].get_bounding_box() 
                    max_intersect_area = intersect_area
            i += 1
            obj_id += 1

        #This is disabled when we do not want to recognize trophy(disabled only for testing containers)
        if goal_index != -1:
            self.current_frame_obstacles[goal_index].is_goal = True

        #print ("current frame obstacles size", len(self.current_frame_obstacles))

  
    def prediction_level1(self):
        rgbI = np.array(self.step_output.image_list[-1])
        bgrI = rgbI[:, :, [2,1,0]]
        depthI = np.uint8(self.step_output.depth_map_list[-1] / self.step_output.camera_clipping_planes[1] * 255)

        ret = self.mask_predictor.step(bgrI, depthI)
        cls = np.argmax(ret['obj_class_score'], axis=1)
        #print(cls)
        n_th_obj = None
        if TROPHY_INDEX in cls:
            n_th_obj = np.where(cls == TROPHY_INDEX)[0]
            print("find trophy in {}th fg object".format(n_th_obj))

        # self.debug_out(bgrI, depthI, ret)
        return ret , n_th_obj


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
