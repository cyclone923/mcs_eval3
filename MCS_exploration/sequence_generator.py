import numpy as np
import random
from qa_agents import graph_agent

import constants
from MCS_exploration.utils import game_util
import cover_floor
from cover_floor import *
import math
import time
from shapely.geometry import Point, Polygon
#from tasks.point_goal_navigation.navigator import NavigatorResNet
#from tasks.search_object_in_receptacle.face_turner import FaceTurnerResNet
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

    def explore_scene_view(self, event, config_filename=None, frame_collector=None):
        number_actions = 0
        success_distance = 0.3
        self.scene_name = 'transferral_data'
        # print('New episode. Scene %s' % self.scene_name)
        self.agent.reset(self.scene_name, config_filename=config_filename, event=event)

        self.position = self.agent.game_state.position
        self.event = self.agent.game_state.event
        #print ("beginning of explore scene view")

        #rotation = self.agent.game_state.event.rotation / 180 * math.pi
        cover_floor.update_seen(self.agent.game_state.position['x'],self.agent.game_state.position['z'],self.agent.game_state,self.agent.game_state.rotation,self.event.camera_field_of_view,self.agent.nav.scene_obstacles_dict.values())

        cover_floor.explore_initial_point(self.agent.game_state.position['x'],self.agent.game_state.position['z'],self.agent,self.agent.nav.scene_obstacles_dict.values())
        exploration_routine = cover_floor.flood_fill(0,0, cover_floor.check_validity)
        pose = game_util.get_pose(self.game_state)[:3]

        #print ("done exploring point and now going to random points")
        #print ("current pose ", pose)
        #self.explore_all_objects()

        if self.agent.game_state.goals_found:
            print ("Object found returning to main ")
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

        print ("object goal ID = " , self.agent.game_state.goal_id)

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
            self.face_object(target_obj)
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
        goal_pixel_coords = np.where(reshaped_obj_masks==target_obj.current_frame_id )
        if len(goal_pixel_coords[0])==0 :
            return None, None

        #print ("xmax,xmin", np.amax(goal_pixel_coords[0]), np.amin(goal_pixel_coords[0]))
        #print ("ymax,ymin", np.amax(goal_pixel_coords[1]), np.amin(goal_pixel_coords[1]))
        y = ((np.amax(goal_pixel_coords[0]) - np.amin(goal_pixel_coords[0]))/2) + np.amin(goal_pixel_coords[0])
        x = ((np.amax(goal_pixel_coords[1]) - np.amin(goal_pixel_coords[1]))/2) + np.amin(goal_pixel_coords[1])

        goal_pixel_coords_tuples = np.array((goal_pixel_coords[0],goal_pixel_coords[1])).T

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
        
        theta = - get_polar_direction(goal_object_centre, self.agent.game_state) * 180/math.pi
        omega = get_head_tilt(goal_object_centre, self.agent.game_state) - self.agent.game_state.head_tilt

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
    
        #print ("object nearest point", object_nearest_point)
        #print ("object nearest point", object_farthest_point)

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

