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
        self.grid_size = 0.1 
        #self.grid_size = 1 
        self.map_width = 12
        self.map_length = 12                                                                     
        self.displacement = 5.5
        self.occupancy_map = self.occupancy_map_init() #* unexplored
        self.object_mask = None
        self.goal_id = None
        self.pose_estimate = np.zeros((3,1),dtype = np.float64)

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
        self.pose = game_util.get_pose(self.event)
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
            #grid_file = 'layouts/%s-layout_%s.npy' % (scene_name,str(constants.AGENT_STEP_SIZE))
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

            self.bounds = None

            #while True :
            #self.event = self.event.events[0]
            if event != None :
                self.event = event
            else :
                self.event = game_util.reset(self.env, self.scene_name,config_filename)
            self.goals = []
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

            for elem in self.event.object_list:
                if self.event.goal.metadata['target']['id'] == elem.uuid :
                    self.goal_object = elem
            
            #print (self.goal_object)

            dimensions = self.goal_object.dimensions
            bd_point = set()
            for i in range(0, 8):
                x, z = dimensions[i]['x'], dimensions[i]['z']
                if (x, z) not in bd_point:
                    bd_point.add((x, z))

            poly = MultiPoint(sorted(bd_point)).convex_hull
            x_list, z_list = poly.exterior.coords.xy
            self.goal_bounding_box = ObstaclePolygon(x_list, z_list)


            #for i in range(0,8):    
            #    x_list.append(self.goal_object.dimensions[i]['x'])
            #    z_list.append(self.goal_object.dimensions[i]['z'])
            #self.goal_bounding_box = ObstaclePolygon(x_list,z_list)
            
            #print ("self goal object x list , y list", x_list,z_list)
            #print ("self goal ext cooords", self.goal_bounding_box.exterior.coords.xy)
            #exit()

            position = self.event.position
            current_angle = math.radians(self.event.rotation)
            self.pose_estimate = np.array([float(position['x']),float(position['z']),current_angle]).reshape(3, 1)
            #self.pose_estimate = np.array([0,0,0]).reshape(3,1)
            agent_pos = {'x': self.pose_estimate[0][0], 'y': 0.465, 'z':self.pose_estimate[1][0]}
            rotation = math.degrees(self.pose_estimate[2][0])
            self.step_output = self.event
            #if self.number_actions %3 == 0 :
            bounding_boxes =convert_observation(self,self.number_actions,agent_pos, rotation)
            #self.add_obstacle_func(self.event)
            self.add_obstacle_func(bounding_boxes)
            #self.add_obstacle_func_eval3(bounding_boxes)
            #self.add_obstale_func_eval3()#(self.event)
            #print ("type of event 2 : ", type(self.event))
            lastActionSuccess = self.event.return_status
            #print (self.pose_estimate)
            #print (self.step_output.position, math.radians(self.step_output.rotation))
            #break


        self.process_frame()
        self.board = None
        #print ("end of reset in game state function")

    def step(self, action_or_ind):
        self.new_found_objects = []
        self.new_object_found = False
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
            #action =  'MoveAhead, amount=0.5'
            #action =  'MoveAhead, amount=0.2'
        elif action['action'] == 'OpenObject':
            action = "OpenObject,objectId="+ str(action["objectId"])
            #print ("constructed action for open object", action)
        
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

        #print ("action time", action_time)
        # lastActionSuccess = self.event.return_status

        for obj in self.event.object_list :
            # if obj.uuid == "trophy":
            #     if not obj.visible:
            #         print("trophy not visible {}, {}".format(self.event.position, self.event.rotation))
            #     else:
            #         print("trophy is visible {} {}".format(self.event.position, self.event.rotation))

            if obj.uuid not in self.discovered_explored and obj.visible:
                # print ("uuid : ", obj.uuid)
                self.discovered_explored[obj.uuid] = {0:obj.position}
                self.discovered_objects.append(obj.__dict__)
                self.new_object_found = True
                self.new_found_objects.append(obj.__dict__)
                self.discovered_objects[-1]['explored'] = 0
                self.discovered_objects[-1]['locationParent'] = None
                self.discovered_objects[-1]['openable'] = None

        #print ("self event goal", self.event.goal.__dict__)
        #print ("self objects" , self.event.object_list[0])
        #exit()
        agent_movement = np.array([vel, ang_rate],dtype=np.float64).reshape(2, 1)
        if self.event.return_status != "OBSTRUCTED":
            self.pose_estimate = self.motion_model(self.pose_estimate,agent_movement) 
        #agent_pos = self.event.position
        #rotation = self.event.rotation
        agent_pos = {'x': self.pose_estimate[0][0], 'y': 0.465, 'z':self.pose_estimate[1][0]}
        rotation = math.degrees(self.pose_estimate[2][0])
        self.step_output = self.event
        start_time = time.time()
        bounding_boxes = convert_observation(self,self.number_actions,agent_pos, rotation)# ,self.occupancy_map) 
        #bounding_boxes = convert_observation(self,self.number_actions)#,agent_pos, rotation) 
        #print ("Frame processing time" , time.time()- start_time)
        #self.add_obstacle_func(self.event)
        self.add_obstacle_func(bounding_boxes)
        #self.add_obstacle_func_eval3(bounding_boxes)
        self.number_actions += 1
        #print (self.pose_estimate)
        #print (self.step_output.position, math.radians(self.step_output.rotation))

        #self.times[2, 0] += time.time() - t_start
        #self.times[2, 1] += 1
        # if self.times[2, 1] % 100 == 0:
        #     print('env step time %.3f' % (self.times[2, 0] / self.times[2, 1]))

        
        #print ("return status from step " , self.event.return_status)
        #if self.event.metadata['lastActionSuccess']:
        if self.event.return_status :
            self.process_frame()
        else :
            print ("Failed status : ",self.event.return_status )

    
        #goal_occupancy_map = 

        '''
        #print (self.event.object_mask_list)
        min_obstacle_area = 100000
        for obstacle_id,obstacle_polygon in self.get_obstacles().items():
            print ("obstcle poly area",obstacle_polygon.area)
            if obstacle_polygon.area < min_obstacle_area :
                min_area_obstacle = obstacle_id
                min_obstacle_area = obstacle_polygon.area

        epsilon = 1.1* grid_size * grid_size
        if  (min_obstacle_area) < epsilon:
            print ("small area")
            #self.goal_centre_y = 
            #print (self.goal_centre_x)
            #print (self.goal_centre_z)
            #print (agent_pos)  
            #print (self.goal_object_nearest_point)
            self.goals_found = True
        '''
        #for elem in self.discovered_explored:
        #    if elem in self.goals:
        #        #total_goal_objects_found[scene_type] += 1
        #        self.goals.remove(elem)

        #if len(self.goals) == 0 :
        #    self.goals_found = True


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
