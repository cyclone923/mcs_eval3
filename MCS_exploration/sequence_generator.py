import numpy as np
import random
#from utils import drawing
from qa_agents import graph_agent
#import graph_agent

import constants
#from utils import game_util
from MCS_exploration.utils import game_util
#import game_util
#from mcs import cover_floor
import cover_floor
from cover_floor import *
import math
import time
from shapely.geometry import Point, Polygon
from tasks.point_goal_navigation.navigator import NavigatorResNet
from tasks.search_object_in_receptacle.face_turner import FaceTurnerResNet
from MCS_exploration.frame_processing import *
from MCS_exploration.navigation.visibility_road_map import ObstaclePolygon,IncrementalVisibilityRoadMap
from shapely.geometry import Point, MultiPoint

class SequenceGenerator(object):
    def __init__(self,sess, env):
        #print ("seq generator init")
        self.agent = graph_agent.GraphAgent(env, reuse=True)
        self.game_state = self.agent.game_state
        self.action_util = self.game_state.action_util
        self.planner_prob = 0.5
        self.scene_num = 0
        self.count = -1
        self.scene_name = None
        #self.nav =bounding_box_navigator.BoundingBoxNavigator()
        #if isinstance(self.nav, BoundingBoxNavigator):
        #    self.env.add_obstacle_func = self.nav.add_obstacle_from_step_output

    def explore_3d_scene(self,event,config_filename=None):
        number_actions = 0
        success_distance = 0.3
        self.scene_name = 'transferral_data'
        #print('New episode. Scene %s' % self.scene_name)
        self.agent.reset(self.scene_name,config_filename = config_filename,event=event)
        self.graph = graph_2d()
        #self.graph.reset()

        self.event = self.game_state.event
        #self.agent.nav.add_obstacle_from_step_output(self.event)
        #plan, path = self.agent.gt_graph.get_shortest_path(
        #        pose, self.game_state.end_point)
        #print ("optimal path planning done", path, plan)
        num_iters = 0
        exploration_routine = []
        exploration_routine = cover_floor.flood_fill(0,0, cover_floor.check_validity)        
        #print (exploration_routine, len(exploration_routine))
        pose = game_util.get_pose(self.game_state.event)[:3]

        self.graph.update_seen(self.event.position['x'],self.event.position['z'],self.event.rotation,100,self.event.camera_field_of_view,self.agent.nav.scene_obstacles_dict )
        unexplored = self.graph.get_unseen()
        print ("before explore point ", len(unexplored))

        unexplored = self.graph.get_unseen()
        print ("after explore point", len(unexplored))
        self.event = self.agent.game_state.event
        pose = game_util.get_pose(self.game_state.event)[:3]

        while ( len(unexplored) > 35 ) :
            start_time = time.time()

            if self.agent.game_state.goals_found :
                return
            print ("before next best point calculation")
            max_visible = 0
            max_visible_position = []
            processed_points = {}
            start_time = time.time()
            print (exploration_routine)
            min_distance = 20
            while (len(max_visible_position) == 0):
                for elem in exploration_routine:
                    #number_visible_points = points_visible_from_position(exploration_routine[1][0],exploration_routine[1][1], self.event.camera_field_of_view,self.event.camera_clipping_planes[1] )
                    #number_visible_points = points_visible_from_position(self.event.position['x'],self.event.position['z'],self.event.camera_field_of_view,100,self.nav.scene_obstacles_dict,self.graph.graph )
                    distance_to_point = math.sqrt((pose[0] - elem[0])**2 + (pose[1]-elem[1])**2)

                    if distance_to_point > min_distance and elem not in processed_points:
                        new_visible_pts = self.graph.points_visible_from_position(elem[0]*constants.AGENT_STEP_SIZE, elem[1]*constants.AGENT_STEP_SIZE, self.event.camera_field_of_view,100,self.agent.nav.scene_obstacles_dict )
                        processed_points[elem] = new_visible_pts
                        #if max_visible < number_visible_points/math.sqrt((pose[0]-elem[0])**2 + (pose[1]-elem[1])**2):
                        if max_visible < new_visible_pts: #and abs(max_visible_points[-1][0] - elem[0]) > 2 and  :
                            max_visible_position.append(elem)
                            max_visible = new_visible_pts

                    min_distance = min_distance/2
                    #points_visible(elem)
            max_visible_position.append((7,-7))
            end_time = time.time()
            print (max_visible_position)
            print ("time taken to select next position" , end_time-start_time)
            if len(max_visible_position) == 0 :
                return number_actions
            new_end_point = [0]*3
            new_end_point[0] = max_visible_position[-1][0]
            new_end_point[1] = max_visible_position[-1][1]
            new_end_point[2] = pose[2]
            exploration_routine.remove(max_visible_position[-1])

            print ("New goal selected : ", new_end_point)

            number_actions = self.agent.nav.go_to_goal(new_end_point,self.agent,success_distance,self.graph,True)
            if self.agent.game_state.goals_found :
                return
            self.graph.explore_point(self.agent.game_state.event.position['x'],self.agent.game_state.event.position['z'], self.agent, 42.5,self.agent.nav.scene_obstacles_dict)
            if self.agent.game_state.goals_found :
                return

            unexplored = self.graph.get_unseen()
            print (len(unexplored))
            end_time = time.time()
            print ("Time taken for 1 loop run = ", end_time - start_time)
            
        return number_actions

    def explore_scene_view(self, event, config_filename=None, frame_collector=None):
        number_actions = 0
        success_distance = 0.3
        self.scene_name = 'transferral_data'
        # print('New episode. Scene %s' % self.scene_name)
        self.agent.reset(self.scene_name, config_filename=config_filename, event=event)

        self.event = self.agent.game_state.event
        #print ("beginning of explore scene view")

        #rotation = self.agent.game_state.event.rotation / 180 * math.pi
        cover_floor.update_seen(self.event.position['x'],self.event.position['z'],self.agent.game_state,self.event.rotation,self.event.camera_field_of_view,self.agent.nav.scene_obstacles_dict.values())

        cover_floor.explore_initial_point(self.event.position['x'],self.event.position['z'],self.agent,self.agent.nav.scene_obstacles_dict.values())
        exploration_routine = cover_floor.flood_fill(0,0, cover_floor.check_validity)
        pose = game_util.get_pose(self.game_state)[:3]

        #print ("done exploring point and now going to random points")
        #print ("current pose ", pose)
        #self.explore_all_objects()

        if self.agent.game_state.goals_found:
            #print ("Object found returning to main ")
            self.go_to_goal_and_pick()
            return
    
        '''
        new_end_point = [0]*3
        new_end_point[0] = 3.3 #self.agent.game_state.goal_object_nearest_point[0]
        new_end_point[1] = 3.4#self.agent.game_state.goal_object_nearest_point[1]
        new_end_point[2] = pose[2]
        success_distance = 0.2 
        nav_success = self.agent.nav.go_to_goal(new_end_point, self.agent, success_distance) 
        '''
        #print ("beginning of explore scene view 2")

        x_list, y_list = [],[]

        for key,value in self.agent.nav.scene_obstacles_dict.items():
            x_list.append(min(value.x_list))
            x_list.append(max(value.x_list))
            y_list.append(min(value.y_list))
            y_list.append(max(value.y_list))

        #print ("bounds", min(x_list),max(x_list),min(y_list),max(y_list))
        x_min = min(x_list)
        x_max = max(x_list)
        y_min = min(y_list)
        y_max = max(y_list)

        #outer_poly = Polygon(zip([x_min,x_min,x_max,x_max],[y_min,y_max,y_min,y_max]))
        #outer_poly_new = outer_poly.difference(self.agent.game_state.world_poly)
        #print (value.x_list,value.y_list)
        #print (outer_poly_new.area)

        overall_area = abs(x_max-x_min) * abs (y_max-y_min)
        #print ("beginning of explore scene view 3")

        while overall_area * 0.65 >  self.agent.game_state.world_poly.area or len(self.agent.game_state.discovered_objects) == 0 :
            #print ("In the main for loop of executtion")
            points_checked = 0
            #z+=1
            max_visible_position = []
            processed_points = {}
            start_time = time.time()
            #print(exploration_routine)
            min_distance = 20
            while (len(max_visible_position) == 0):
                max_visible = 0
                for elem in exploration_routine:
                    distance_to_point = math.sqrt((pose[0] - elem[0])**2 + (pose[1]-elem[1])**2)

                    if distance_to_point > min_distance and elem not in processed_points:
                        points_checked += 1
                        #for obstacle_key, obstacle in self.agent.nav.scene_obstacles_dict.items():
                        #    if obstacle.contains_goal(elem):
                        #        continue

                        new_visible_area = cover_floor.get_point_all_new_coverage(elem[0]*constants.AGENT_STEP_SIZE, elem[1]*constants.AGENT_STEP_SIZE, self.agent.game_state,self.agent.game_state.event.rotation,self.agent.nav.scene_obstacles_dict.values() )
                        processed_points[elem] = new_visible_area
                        #if max_visible < number_visible_points/math.sqrt((pose[0]-elem[0])**2 + (pose[1]-elem[1])**2):
                        if max_visible < new_visible_area: #and abs(max_visible_points[-1][0] - elem[0]) > 2 and  :
                            max_visible_position.append(elem)
                            max_visible = new_visible_area

                min_distance = min_distance/2
                if min_distance < 1 :
                    break
                #points_visible(elem)
            end_time = time.time()

            #max_visible_position = [(7,-7)]
            #print(max_visible_position)
            time_taken = end_time-start_time
            #print("time taken to select next position", end_time - start_time)
            #print ("points searched with area overlap", points_checked)
            if len(max_visible_position) == 0:
                return
            new_end_point = [0] * 3
            new_end_point[0] = max_visible_position[-1][0] *constants.AGENT_STEP_SIZE
            new_end_point[1] = max_visible_position[-1][1] *constants.AGENT_STEP_SIZE
            new_end_point[2] = pose[2]

            #print("New goal selected : ", new_end_point)

            nav_success = self.agent.nav.go_to_goal(new_end_point, self.agent, success_distance)
            exploration_routine.remove(max_visible_position[-1])

            if nav_success == False :
                continue

            #self.event = self.agent.game_state.event
            if self.agent.game_state.goals_found:
                self.go_to_goal_and_pick()
                return
            cover_floor.explore_point(self.agent.game_state.position['x'], self.agent.game_state.position['z'], self.agent,
                                      self.agent.nav.scene_obstacles_dict.values())
            if self.agent.game_state.goals_found :
                self.go_to_goal_and_pick()
                return
            if self.agent.game_state.number_actions > constants.MAX_STEPS :
                print ("Too many actions performed")
                return
            if len(exploration_routine) == 0:
                #self.go_to_goal_and_pick()
                print ("explored a lot of points but objects not found")
                return

        all_explored = False
        while (all_explored == False):
            current_pos = self.agent.game_state.position
            min_distance = math.inf
            flag = 0
            for object in self.agent.game_state.discovered_objects :
                #distance_to_object =
                #if object['explored'] == 0  and object['locationParent'] == None and len(object['dimensions']) >0:
                if object['explored'] == 0 :#and len(object['dimensions']) > 0:# and object['locationParent'] == None
                    flag = 1
                    distance_to_object = math.sqrt( (current_pos['x'] - object['position']['x'] )** 2 + (current_pos['z']-object['position']['z'])**2)
                else :
                    continue

                if distance_to_object < min_distance:
                    min_distance_obj_id = object['uuid']
                    min_distance = distance_to_object

            if flag == 0 :
                all_explored = True
                break
            self.explore_object(min_distance_obj_id)
            omega = -self.agent.game_state.event.head_tilt
            m = int(omega // 10)

            if omega > 0:
                for _ in range(m):
                    self.agent.step({"action": "LookDown"})
            else:
                for _ in range(m):
                    self.agent.step({"action": "LookUp"})

            if self.agent.game_state.goals_found:
                return
            if self.agent.game_state.number_actions > constants.MAX_STEPS :
                # print ("Too many actions performed")
                return

        #self.explore_object(self.agent.game_state.discovered_objects[0])

    def go_to_goal_and_pick(self):

        #print ("object goal ID = " , self.agent.game_state.goal_id)

        target_obj = self.get_target_obj(self.agent.game_state.goal_id)
        object_nearest_point = self.get_best_object_point(target_obj, 1000, self.nearest )
        agent_pos = self.agent.game_state.position

        success_distance = 0.40 
        #print ("agent position" , agent_pos)
        #print ("goal point : ", object_nearest_point)
        nav_success = self.agent.nav.go_to_goal(object_nearest_point, self.agent, success_distance) 
        self.face_object(target_obj)
        x,y = self.get_obj_pixels(target_obj)
                
        action = {'action':"PickupObject", 'x': x, 'y':y}
        self.agent.game_state.step(action)
        going_closer_counter = 0

        while self.agent.game_state.event.return_status == "OUT_OF_REACH":
            success_distance -= 0.03
            nav_success = self.agent.nav.go_to_goal(object_nearest_point, self.agent, success_distance) 
            #self.update_goal_centre()
            self.face_object()
            x,y = self.get_obj_pixels(target_obj)
            #print (x,y)
            if x == None and y == None :
                continue
            action = {'action':"PickupObject", 'x': x, 'y':y}
            self.agent.game_state.step(action)
            going_closer_counter += 1
            if going_closer_counter > 5:
                break
        

    def get_obj_pixels(self,target_obj):
        arr_mask = np.array(self.agent.game_state.event.object_mask_list[-1])
        reshaped_obj_masks = arr_mask.reshape(-1, arr_mask.shape[-1])
        ar_row_view= reshaped_obj_masks.view('|S%d' % (reshaped_obj_masks.itemsize * reshaped_obj_masks.shape[1]))
        reshaped_obj_masks = ar_row_view.reshape(arr_mask.shape[:2])
        #goal_pixel_coords = np.where(ar_row_view==self.agent.game_state.goal_id )
        goal_pixel_coords = np.where(reshaped_obj_masks==target_obj.current_frame_id )
        #print (len(goal_pixel_coords[0]))  
        if len(goal_pixel_coords[0])==0 :
            return None, None

        #print ("xmax,xmin", np.amax(goal_pixel_coords[0]), np.amin(goal_pixel_coords[0]))
        #print ("ymax,ymin", np.amax(goal_pixel_coords[1]), np.amin(goal_pixel_coords[1]))
        y = ((np.amax(goal_pixel_coords[0]) - np.amin(goal_pixel_coords[0]))/2) + np.amin(goal_pixel_coords[0])
        x = ((np.amax(goal_pixel_coords[1]) - np.amin(goal_pixel_coords[1]))/2) + np.amin(goal_pixel_coords[1])

        goal_pixel_coords_tuples = np.array((goal_pixel_coords[0],goal_pixel_coords[1])).T

        '''
        print (type(goal_pixel_coords_tuples), len(goal_pixel_coords_tuples))
        print ("get goal pixels")
        print ("example,", goal_pixel_coords_tuples[0], (x,y))
        
        if (x,y) in goal_pixel_coords_tuples : 
            print ("goal pixels in the list of possible goal pixels")
        else :
            print ("goal pixels not in list of possible goal pixels")
        '''
        return x,y

    def get_target_obj(self,target_obj_id):
        target_obj = None
        for obstacle in self.agent.game_state.global_obstacles :
            if obstacle.id == target_obj_id:
                target_obj = obstacle
                break
        return target_obj
        

    def face_object(self,target_obj):
            
        goal_object_centre = [0]*3
        goal_object_centre[0] = target_obj.centre_x
        goal_object_centre[1] = target_obj.centre_y
        goal_object_centre[2] = target_obj.centre_z
        #goal_pixel_coords = np.where(self.agent.game_state.object_mask==self.agent.game_state.goal_id )
        
        theta = - NavigatorResNet.get_polar_direction(goal_object_centre, self.agent.game_state.event) * 180/math.pi
        omega = FaceTurnerResNet.get_head_tilt(goal_object_centre, self.agent.game_state.event) - self.agent.game_state.event.head_tilt

        n = int(abs(theta) // 10)
        m = int(abs(omega) // 10)

        #print ("Theta", theta)
        #print ("Omega", omega)
        if theta > 0:
            #action = {'action': 'RotateRight'}
            action = {'action': 'RotateLeft'}
            for _ in range(n):
                self.agent.game_state.step(action)
        else:
            action = {'action': 'RotateRight'}
            #action = {'action': 'RotateLeft'}
            for _ in range(n):
                self.agent.game_state.step(action)

        if omega > 0:
            action = {'action': 'LookDown'}
            for _ in range(m):
                self.agent.game_state.step(action)
        else:
            action = {'action': 'LookUp'}
            for _ in range(m):
                self.agent.game_state.step(action)

    def look_straight(self):
        omega = self.agent.game_state.event.head_tilt
         
        m = int(abs(omega) // 10)
        if omega > 0:
            action = {'action': 'LookUp'}
            for _ in range(m):
                self.agent.game_state.step(action)
        else:
            action = {'action': 'LookDown'}
            for _ in range(m):
                self.agent.game_state.step(action)
        

    def get_best_object_point(self,target_obj,dist_points,dist_func):
        agent_pos  =self.agent.game_state.position
        exterior_coords = target_obj.get_convex_polygon_coords()
        x_list = exterior_coords[0]
        y_list = exterior_coords[1]
        for x in x_list:
            for y in y_list:
                #if math.sqrt( (x-agent_pos['x'])**2 + (y-agent_pos['z'])**2 ) < dist_points :
                if dist_func(math.sqrt( (x-agent_pos['x'])**2 + (y-agent_pos['z'])**2 ),dist_points) :
                    object_point = [x,y]
                    dist_points = math.sqrt( (x-agent_pos['x'])**2 + (y-agent_pos['z'])**2 )

        return object_point

    def nearest(self,a,b):  
        if a < b :
            return True

    def farthest(self,a,b):
        if a > b :
            return True

    def go_to_object_and_open(self,target_obj):
        #target_obj = get_target_obj(target_obj_id)
        object_nearest_point = self.get_best_object_point(target_obj, 1000, self.nearest)
        object_farthest_point = self.get_best_object_point(target_obj, 0.0 , self.farthest)
    
        print ("object nearest point", object_nearest_point)
        print ("object nearest point", object_farthest_point)

        success_distance = 0.40 
        nav_success = self.agent.nav.go_to_goal(object_nearest_point, self.agent, success_distance) 
        self.face_object(target_obj)

        x,y = self.get_obj_pixels (target_obj)
        action = {'action':"OpenObject", 'x': x, 'y':y}
        self.agent.game_state.step(action)

        self.look_straight()
        nav_success = self.agent.nav.go_to_goal(object_farthest_point, self.agent, success_distance) 
        self.face_object(target_obj)
    
    def explore_all_objects(self):
        for obstacle in self.agent.game_state.global_obstacles[2:] :
            self.go_to_object_and_open(obstacle)


    def update_goal_centre(self):
        displacement = self.agent.game_state.displacement
        goal_points = self.agent.game_state.goal_calculated_points
        obj_occ_map = get_occupancy_from_points( goal_points,self.agent.game_state.occupancy_map.shape)   
        self.goal_object = polygon_simplify(occupancy_to_polygons( obj_occ_map,self.agent.game_state.grid_size,displacement ))
        if self.goal_object.geom_type == "MultiPolygon":
            #allparts = [p.buffer(0) for p in .geometry]
            #simensions = self.goal_object.dimensions
            bd_point = set()
            for polygon in self.goal_object :
                x_list, z_list = polygon.exterior.coords.xy
                #print ("each poly exterior pts", x_list,z_list)
                for x,z in zip(x_list,z_list):
                    #x, z = dimensions[i]['x'], dimensions[i]['z']
                    if (x, z) not in bd_point:
                        bd_point.add((x, z))

            #print ("boundary points")
            poly = MultiPoint(sorted(bd_point)).convex_hull
            self.goal_object = poly.simplify(0.0)#MultiPoint(sorted(bd_point)).convex_hull
            #x_list, z_list = poly.exterior.coords.xy
            #self.goal_bounding_box = ObstaclePolygon(x_list, z_list)
            #print ("multi polygon ",self.goal_bounding_box.exterior.coords.xy)    
            #return
        #else: 
        exterior_coords = self.goal_object.exterior.coords.xy
        
        self.goal_object = ObstaclePolygon(exterior_coords[0], exterior_coords[1])
        #print ("exterior coords calculated", exterior_coords)
        #print ("Exterior coords grnd truth x ", self.agent.game_state.goal_bounding_box.x_list)
        #print ("Exterior coords grnd truth z ", self.agent.game_state.goal_bounding_box.y_list)
        self.goal_centre_x = np.mean(np.array(self.goal_object.x_list,dtype=object))
        self.goal_centre_z = np.mean(np.array(self.goal_object.y_list,dtype=object))

    def explore_object(self, object_id_to_search):
        uuid = object_id_to_search
        success_distance = constants.AGENT_STEP_SIZE
        #object_polygon = self.agent.nav.scene_obstacles_dict[uuid]
        current_position = self.agent.game_state.event.position
        current_object_position = 0
        i = 0
        for elem in self.agent.game_state.discovered_objects:
            if uuid == elem['uuid']:
                current_object = elem
                current_object_position = i
                break
            i+= 1
        goal_object_centre = [current_object['position']['x'], current_object['position']['y'],
                              current_object['position']['z']]
        self.agent.game_state.discovered_objects[current_object_position]['explored'] = 1

        number_vertices = 4
        min_distance = math.inf
        goal_poses = []

        if len(current_object['dimensions'])==0:
            #print ("right track")
            if current_object['locationParent'] == None:
                for obstacle_key, obstacle in self.agent.nav.scene_obstacles_dict.items():
                    if obstacle.contains_goal((goal_object_centre[0],goal_object_centre[2])):
                        #goal_inside_obstacle = True
                        current_object['locationParent'] = obstacle_key
                        break

            if current_object['locationParent'] in self.agent.game_state.discovered_explored:
                for elem in self.agent.game_state.discovered_objects:
                    if elem['uuid'] == current_object['locationParent']:
                        dimension_object = elem
                        break
                #dimension_object =self.agent.game_state.discovered_objects[current_object['locationParent']]
            else:
                return
        else:
            dimension_object = current_object

        for i in range(4,4+number_vertices):
            #goal = [current_object['dimensions'][i]['x'],current_object['dimensions'][i]['y'],current_object['dimensions'][i]['z']]
            goal = [dimension_object['dimensions'][i]['x'],dimension_object['dimensions'][i]['y'],dimension_object['dimensions'][i]['z']]

            if len(current_object['dimensions']) == 0:
                distance_to_point = math.sqrt((goal_object_centre[0] - goal[0]) ** 2 + (goal_object_centre[2] - goal[2]) ** 2)
            else:
                distance_to_point = math.sqrt((current_position['x'] - goal[0]) ** 2 + (current_position['z'] - goal[2]) ** 2)
            #if distance_to_point < min_distance :
            #    goal_pose = goal
            #    min_distance = distance_to_point
            goal_poses.append(( goal , distance_to_point))

        #self.agent.nav.go_to_goal(goal_pose,self.agent,success_distance,self.graph,True)

        goal_poses = sorted(goal_poses, key = lambda x: x[1])
        goal_poses_not_visited = goal_poses[:]

        #for goal_pose in goal_poses[:-2]:
        for i in range (len(goal_poses)-1):
            current_position = self.agent.game_state.event.position
            #goal_pose = goal_poses[i]
            #'''
            min_distance_vertex = math.inf
            for j in range(len(goal_poses_not_visited)):
                if len(current_object['dimensions']) == 0:
                    distance_to_point = math.sqrt(
                        (goal_object_centre[0] - goal_poses_not_visited[j][0][0]) ** 2 +
                        (goal_object_centre[2] - goal_poses_not_visited[j][0][2]) ** 2)
                else:
                    distance_to_point = math.sqrt((current_position['x'] - goal_poses_not_visited[j][0][0]) ** 2 +
                                              (current_position['z'] - goal_poses_not_visited[j][0][2]) ** 2)
                if min_distance_vertex > distance_to_point:
                    goal_pose = goal_poses_not_visited[j]
                    min_distance_vertex= distance_to_point
            #'''

            goal_pose_loc = goal_pose[0]
            goal_poses_not_visited.remove(goal_pose)
            goal_pose_x_z =(goal_pose_loc[0],goal_pose_loc[2])
            goal_pose_x_z = cover_floor.get_point_between_points(goal_pose_x_z,[goal_object_centre[0],goal_object_centre[2]],self.agent.nav_radius)

            goal_inside_obstacle = False

            for obstacle_key, obstacle in self.agent.nav.scene_obstacles_dict.items():
                if obstacle.contains_goal(goal_pose_x_z):
                    goal_inside_obstacle = True
                    break

            if goal_inside_obstacle :
                continue



            if len(current_object['dimensions']) == 0 :
                if goal_object_centre[1] > self.agent.game_state.event.position['y']:
                    return
                if current_object['openable'] == True :
                    return
                if i == 2:
                    return

            nav_success = self.agent.nav.go_to_goal(goal_pose_x_z, self.agent, success_distance)
            if nav_success == False:
                continue


            theta = - NavigatorResNet.get_polar_direction(goal_object_centre, self.agent.game_state.event) * 180/math.pi
            omega = FaceTurnerResNet.get_head_tilt(goal_object_centre, self.agent.game_state.event) - self.agent.game_state.event.head_tilt

            n = int(abs(theta) // 10)
            m = int(abs(omega) // 10)
            if theta > 0:
                action = {'action': 'RotateRight'}
                for _ in range(n):
                    self.agent.game_state.step(action)
            else:
                action = {'action': 'RotateLeft'}
                for _ in range(n):
                    self.agent.game_state.step(action)

            if omega > 0:
                action = {'action': 'LookDown'}
                for _ in range(m):
                    self.agent.game_state.step(action)
            else:
                action = {'action': 'LookUp'}
                for _ in range(m):
                    self.agent.game_state.step(action)



            if current_object['openable'] == True:
                if i == 2 :
                    break
            elif current_object['openable'] == False:
                if i == 1 :
                    break


            object_visible = False
            for elem in self.agent.game_state.event.object_list :
                if uuid == elem.uuid and elem.visible:
                    object_visible = True
                    break
            if object_visible == True :
                action = {"action": "OpenObject", "objectId": uuid}
                self.agent.step(action)
                status = self.agent.game_state.event.return_status
                if status == "SUCCESSFUL" or  status == "IS_OPENED_COMPLETELY":
                    #TODO update if new object is seen and return
                    self.agent.game_state.discovered_objects[current_object_position]['opened'] = True
                    self.agent.game_state.discovered_objects[current_object_position]['openable'] = True
                    if self.agent.game_state.new_object_found == True:
                        self.update_object_data(uuid)
                        return
                    for _ in range(3):
                        self.agent.step({"action": "RotateRight"})
                    if self.agent.game_state.new_object_found == True :
                        self.update_object_data(uuid)
                        return
                    for _ in range(3):
                        self.agent.step({"action": "RotateLeft"})
                    if self.agent.game_state.new_object_found == True :
                        self.update_object_data(uuid)
                        return
                elif status == "NOT_OPENABLE" or status == "NOT_INTERACTABLE" or status == "NOT_OBJECT" :
                    self.agent.game_state.discovered_objects[current_object_position]['openable'] = False
                    print ("opening failed")
                    continue
                    #return
                elif status == "OBSTRUCTED" or "OUT_OF_REACH":
                    #TODO Maybe add something different for out of reach later
                    continue

            else :
                continue

    def update_object_data(self,parent_obj_id):
        for i, elem in enumerate(self.agent.game_state.new_found_objects):
            for j, obj in enumerate(self.agent.game_state.discovered_objects):
                if elem['uuid'] == obj['uuid']:
                    self.agent.game_state.discovered_objects[j][
                        'agent_position'] = self.agent.game_state.event.position
                    self.agent.game_state.discovered_objects[j][
                        'agent_tilt'] = self.agent.game_state.event.head_tilt
                    self.agent.game_state.discovered_objects[j][
                        'agent_rotation'] = self.agent.game_state.event.rotation
                    self.agent.game_state.discovered_objects[j]['locationParent'] =parent_obj_id
                    self.agent.game_state.discovered_objects[j]['explored'] =1
                    category = self.agent.game_state.event.goal.metadata['category']
                    target_id = None
                    if category == "transferral":
                        target_id = self.agent.game_state.event.goal.metadata['target_1']['id']
                    if category == "retrieval":
                        target_id = self.agent.game_state.event.goal.metadata['target']['id']

                    if target_id != None :
                        if target_id == elem['uuid']:
                            action = {'action':'PickupObject', 'objectId':elem['uuid']}
                            self.agent.game_state.step(action)
                            if self.agent.game_state.event.return_status == 'SUCCESSFUL':
                                self.agent.game_state.goal_in_hand = True
                                self.agent.game_state.id_goal_in_hand = elem['uuid']
                    break


if __name__ == '__main__':
    pass



