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
from shapely import affinity
#from tasks.point_goal_navigation.navigator import NavigatorResNet
#from tasks.search_object_in_receptacle.face_turner import FaceTurnerResNet
from MCS_exploration.frame_processing import *
from MCS_exploration.navigation.visibility_road_map import ObstaclePolygon,IncrementalVisibilityRoadMap
from shapely.geometry import Point, MultiPoint
import operator
from functools import reduce

class SequenceGenerator(object):
    def __init__(self, sess, env, level, frame_collector=None):
        #print ("seq generator init")
        self.controller = env
        self.agent = graph_agent.GraphAgent(env ,
                                            level, 
                                            frame_collector=frame_collector, 
                                            reuse=True)
        self.game_state = self.agent.game_state
        #self.action_util = self.game_state.action_util
        self.planner_prob = 0.5
        self.scene_num = 0
        self.count = -1
        self.scene_name = None
        self.outermost_poly = None

    def run_scene(self, scene_config, config_filename=None, frame_collector=None):
        # try :
        self.step_output = self.controller.start_scene(scene_config)
        self.explore_scene_view(self.step_output, config_filename, frame_collector)
        print(f"SCENE NAME {config_filename} OUTCOME {self.agent.game_state.trophy_picked_up}")
        self.controller.end_scene("")
        # except Exception as e :
            # print (e)
            

    def explore_scene_view(self, event, config_filename=None, frame_collector=None):
        number_actions = 0
        success_distance = 0.3
        self.scene_name = 'transferral_data'
        # print('New episode. Scene %s' % self.scene_name)
        self.agent.reset(self.scene_name, config_filename=config_filename, event=event)

        self.position = self.agent.game_state.position
        self.event = self.agent.game_state.event
        # print ("beginning of explore scene view")

        #rotation = self.agent.game_state.event.rotation / 180 * math.pi
        cover_floor.update_seen(self.agent.game_state.position['x'],self.agent.game_state.position['z'],self.agent.game_state,self.agent.game_state.rotation,self.event.camera_field_of_view,self.agent.nav.scene_obstacles_dict.values())

        cover_floor.explore_initial_point(self.agent.game_state.position['x'],self.agent.game_state.position['z'],self.agent,self.agent.nav.scene_obstacles_dict.values())
        #exploration_routine = cover_floor.flood_fill(0,0, cover_floor.check_validity)

        self.outermost_poly = None

        all_coords = []
        bd_point = set()
        for obstacle in self.agent.game_state.global_obstacles :
            current_coords= obstacle.get_convex_polygon_coords()
            for x,z in zip(current_coords[0], current_coords[1]):
                #all_coords.append((x,y))
                if (x, z) not in bd_point:
                    bd_point.add((x, z))

        outermost_poly = MultiPoint(sorted(bd_point)).convex_hull

        outermost_coords_x = outermost_poly.exterior.coords.xy[0]
        outermost_coords_y = outermost_poly.exterior.coords.xy[1]

        x_min = min(outermost_coords_x)
        x_max = max(outermost_coords_x)
        y_min = min(outermost_coords_y)
        y_max = max(outermost_coords_y)
        x_z_range = [x_min+1,x_max-1,y_min+1,y_max-1]
        x_z_range = [(i/constants.AGENT_STEP_SIZE) for i in x_z_range]


        x, y = np.meshgrid(np.arange(x_z_range[0],x_z_range[1]), np.arange(x_z_range[2],x_z_range[3])) # make a canvas with coordinates
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T 

        points = points[::425]
        exploration_routine = []

        start_time = time.time()
        self.outermost_poly = outermost_poly
        outermost_poly = affinity.scale(outermost_poly ,xfact=0.87, yfact=0.87)
        #exit()
        for point in points :
            real_point = Point(point[0]*constants.AGENT_STEP_SIZE,point[1]*constants.AGENT_STEP_SIZE)
            if real_point.within(outermost_poly):
                exploration_routine.append((point[0],point[1]))

        #self.explore_all_objects()
        #self.pick_up_obstacles(all_obstacles=True)

        if self.agent.game_state.trophy_picked_up == True:
            # print("TROPHY PICKED UP, RETURNING")
            return
        if self.agent.game_state.goals_found :
            # print ("Object found returning to main")
            self.pick_up_obstacles(possible_trophy_obstacles=True)

        if self.agent.game_state.trophy_picked_up == True:
            # print("TROPHY PICKED UP, RETURNING")
            return
    
        overall_area = 102
        pose = game_util.get_pose(self.game_state)[:3]
        # print ("overall area",overall_area)
        # print (" poly area " , self.agent.game_state.world_poly.area)
        while overall_area * 0.85 >  self.agent.game_state.world_poly.area or len(self.agent.game_state.global_obstacles) == 0 :
            # print ("In the main for loop of executtion")
            points_checked = 0
            #z+=1
            max_visible_position = []
            processed_points = {}
            start_time = time.time()
            print(exploration_routine)
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

                        new_visible_area = cover_floor.get_point_all_new_coverage(elem[0]*constants.AGENT_STEP_SIZE, elem[1]*constants.AGENT_STEP_SIZE, self.agent.game_state,self.agent.game_state.rotation,self.agent.nav.scene_obstacles_dict.values() )
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

            time_taken = end_time-start_time
            # print("time taken to select next position", end_time - start_time)
            if len(max_visible_position) == 0:
                break
            new_end_point = [0] * 3
            new_end_point[0] = max_visible_position[-1][0] *constants.AGENT_STEP_SIZE
            new_end_point[1] = max_visible_position[-1][1] *constants.AGENT_STEP_SIZE
            new_end_point[2] = pose[2]

            # print("New goal selected : ", new_end_point)

            nav_success = self.agent.nav.go_to_goal(new_end_point, self.agent, success_distance)
            exploration_routine.remove(max_visible_position[-1])

            # if nav_success == False :
            #     print(9)
            #     continue

            if self.agent.game_state.goals_found:
                self.pick_up_obstacles(possible_trophy_obstacles=True)

            if self.agent.game_state.trophy_picked_up == True:
                # print(8)
                return
            cover_floor.explore_point(self.agent.game_state.position['x'], self.agent.game_state.position['z'], self.agent,
                                      self.agent.nav.scene_obstacles_dict.values())
            if self.agent.game_state.trophy_picked_up == True:
                # print(7)
                return
            if self.agent.game_state.goals_found :
                # print(6)
                self.pick_up_obstacles(possible_trophy_obstacles=True)
                #self.go_to_goal_and_pick()
            if self.agent.game_state.trophy_picked_up == True:
                # print(5)
                return

            if self.agent.game_state.number_actions > constants.MAX_STEPS :
                # print ("Too many actions performed")
                return
            if len(exploration_routine) == 0:
                # print(4)
                #self.go_to_goal_and_pick()
                #print ("explored a lot of points but objects not found")
                break

        # if self.agent.game_state.trophy_picked_up == True:
        #     # print(3)
        #     return
        # if self.agent.game_state.goals_found :
        #     self.pick_up_obstacles(possible_trophy_obstacles=True)
        # if self.agent.game_state.trophy_picked_up == True:
        #     # print(1)
        #     return

        # self.explore_all_objects()
        # if self.agent.game_state.trophy_picked_up == True:
        #     # print(2)
        #     return
        self.pick_up_obstacles(all_obstacles=True)
    
        #for obstacle in self.agent.game_state.global_obstacles :
        #    print ("obstacle data")
        #    print ("is contonained" , obstacle.is_contained)
        #    print ("is open ", obstacle.is_opened)
        #    #print ("is not trophy", ob)

        '''
        if self.agent.game_state.trophy_picked_up == True:
            return
        self.pick_up_obstacles(contained_obstacles=True)
        '''
        return

    def go_to_object_and_pick(self,target_obj):


        #print ("in go to goal and pick")
        object_nearest_point = self.get_best_object_point(target_obj, 1000, self.nearest )
        #goal_object_centre = [0]*2
        #goal_object_centre[0] = target_obj.centre_x
        #goal_object_centre[1] = target_obj.centre_z
        #object_nearest_point = get_point_between_points(object_nearest_point, goal_object_centre, self.agent.nav_radius)
        agent_pos = self.agent.game_state.position
        if object_nearest_point == None :
            return

        success_distance = 0.40 
        #print ("goal point : ", object_nearest_point)
        nav_success = self.agent.nav.go_to_goal(object_nearest_point, self.agent, success_distance) 

        #print ("in nav go to goal done moving towards goal")

        self.face_object(target_obj)

        SHOW_ANIMATION = False
        if SHOW_ANIMATION:
            plt.cla()
            plt.xlim((-7, 7))
            plt.ylim((-7, 7))
            plt.gca().set_xlim((-7, 7))
            plt.gca().set_ylim((-7, 7))

            for obstacle in self.agent.game_state.global_obstacles:
                if obstacle.id ==target_obj.id :
                    patch1 = PolygonPatch(obstacle.get_bounding_box(), fc='red', ec="black", alpha=0.2, zorder=1)
                else :
                    patch1 = PolygonPatch(obstacle.get_bounding_box(), fc='green', ec="black", alpha=0.2, zorder=1)
                plt.gca().add_patch(patch1)
            plt.plot(object_nearest_point[0], object_nearest_point[1], "x")

            plt.axis("equal")
            plt.pause(0.001)
            plt.show()

        #print ("target obj height" , target_obj.height)

        x,y = self.get_obj_pixels(target_obj)
        
        if x == None or y == None :
            return 

        #print ("goal pick up coordinates" ,x,y)
        #print ("goal current frame ID",target_obj.current_frame_id)
                
        action = {'action':"PickupObject", 'x': x, 'y':y}
        self.agent.game_state.step(action)

        if self.update_picked_up(target_obj) :        
            return 
                
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

            if self.update_picked_up(target_obj) :        
                return 

            going_closer_counter += 1
            if going_closer_counter > 2:
                break
  

    def update_picked_up(self,target_obj):
        """ 
        Function to update the status of the world after a pick up action
        
        If the pick up was successfull and it was the trophy - we are done
        If the pick up was successfull and it isnt the trophy - we drop it
        If the pick up was not successfull - we update the object with whatever info we get    
        """
                  
        if self.agent.game_state.event.return_status == "SUCCESSFUL" :
            #print ("object picked up rewards attained = " , self.agent.game_state.event.reward)
            if self.agent.game_state.event.reward > -1* (self.agent.game_state.number_actions*0.001) +0.5 :
                self.agent.game_state.trophy_picked_up = True
                self.update_object_picked_up(target_obj,False)
            else :
                self.update_object_picked_up(target_obj,True)
                action = {'action':'DropObject'}
                self.agent.game_state.step(action)
                self.look_straight()
            return True
        elif self.agent.game_state.event.return_status == "NOT_INTERACTABLE" or  \
                self.agent.game_state.event.return_status == "NOT_OBJECT" or  \
                self.agent.game_state.event.return_status == "NOT_PICKUPABLE" :
            self.update_object_picked_up(target_obj,True)
            self.look_straight()
            return True
        else :
            return False

    def update_object_picked_up(self,target_obj,pick_up_status):
        for i,obstacle in enumerate(self.agent.game_state.global_obstacles) :
            if target_obj.id == obstacle.id :
                self.agent.game_state.global_obstacles[i].is_picked_and_not_trophy = pick_up_status
                return 

    def get_obj_pixels(self,target_obj,get_goal_pixels=False):
        #arr_mask = np.array(self.agent.game_state.event.object_mask_list[-1])
        #reshaped_obj_masks = arr_mask.reshape(-1, arr_mask.shape[-1])
        obj_masks = self.agent.game_state.obj_mask
        level = self.agent.game_state.level
        if level == "level2" or level == "oracle":
            ar_row_view = obj_masks.view('|S%d' % (obj_masks.itemsize * obj_masks.shape[1]))
        if level == "level1" :
            ar_row_view = obj_masks[:]
        #ar_row_view= reshaped_obj_masks.view('|S%d' % (reshaped_obj_masks.itemsize * reshaped_obj_masks.shape[1]))
        reshaped_obj_masks = ar_row_view.reshape(400,600)
        goal_pixel_coords = np.where(reshaped_obj_masks==target_obj.current_frame_id)
        if len(goal_pixel_coords[0])==0:
            if get_goal_pixels :
                return None
            else:
                return None, None

        #print ("xmax,xmin", np.amax(goal_pixel_coords[0]), np.amin(goal_pixel_coords[0]))
        #print ("ymax,ymin", np.amax(goal_pixel_coords[1]), np.amin(goal_pixel_coords[1]))
        y = ((np.amax(goal_pixel_coords[0]) - np.amin(goal_pixel_coords[0]))/2) + np.amin(goal_pixel_coords[0])
        x = ((np.amax(goal_pixel_coords[1]) - np.amin(goal_pixel_coords[1]))/2) + np.amin(goal_pixel_coords[1])

        goal_pixel_coords_tuples = np.array((goal_pixel_coords[0],goal_pixel_coords[1])).T

        if get_goal_pixels :
           return goal_pixel_coords
        else :
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

        #print ("in look straight")
         
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
        if target_obj.is_contained == True :
            #print ("object contained true in get best object point")
            exterior_coords =  target_obj.parent_bounding_box.exterior.coords.xy       
        else :
            exterior_coords = target_obj.get_convex_polygon_coords()
        x_list = exterior_coords[0]
        y_list = exterior_coords[1]
        object_point = None
        for x in x_list:
            for y in y_list:
                #if math.sqrt( (x-agent_pos['x'])**2 + (y-agent_pos['z'])**2 ) < dist_points :
                if dist_func(math.sqrt( (x-agent_pos['x'])**2 + (y-agent_pos['z'])**2 ),dist_points) :
                    current_point = Point(x,y) 
                    if current_point.within(self.outermost_poly):
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

        if object_nearest_point == None :
            return

        #goal_object_centre = [0]*2
        #goal_object_centre[0] = target_obj.centre_x
        #goal_object_centre[1] = target_obj.centre_z
        #print ("object nearest point", object_nearest_point)
        #print ("object nearest point", object_farthest_point)
        #print ("goal centre" , goal_object_centre)
        #object_nearest_point = get_point_between_points(object_nearest_point, goal_object_centre, self.agent.nav_radius)
        #object_farthest_point = get_point_between_points(object_farthest_point, goal_object_centre, self.agent.nav_radius)
    
        #print ("object nearest point", object_nearest_point)
        #print ("object nearest point", object_farthest_point)
        success_distance = 0.35 
        nav_success = self.agent.nav.go_to_goal(object_nearest_point, self.agent, success_distance) 
        self.face_object(target_obj)

        x,y = self.get_obj_pixels (target_obj)
        
        if x == None or y == None :
            return
    
        action = {'action':"OpenObject", 'x': x, 'y':y}
        self.agent.game_state.step(action)

        
        #print ("return status from open", self.agent.game_state.event.return_status)
        if  not self.update_opened_up(target_obj):
            return

        #print ("return statys from open" , self.agent.game_state.event.return_status)

        going_closer_counter = 0

        while self.agent.game_state.event.return_status == "OUT_OF_REACH":
            success_distance -= 0.03
            #print ("sucess distance")
            nav_success = self.agent.nav.go_to_goal(object_nearest_point, self.agent, success_distance) 
            #self.update_goal_centre()
            self.face_object(target_obj)
            x,y = self.get_obj_pixels(target_obj)
            #print (x,y)
            if x == None and y == None :
                continue
            action = {'action':"OpenObject", 'x': x, 'y':y}
            self.agent.game_state.step(action)

            if not self.update_opened_up(target_obj) :        
                return 

            going_closer_counter += 1
            if going_closer_counter > 2:
                break

        if  self.agent.game_state.event.return_status == "SUCCESSFUL":  
            self.random_object_pickup(target_obj)

        if self.agent.game_state.trophy_picked_up == True:
            return
  
        #if self.agent.game_state.goal_object_visible: 
        if self.agent.game_state.goals_found : 
            #print ("found goal in the middle of looking into containers")
            return

    
        #print ("done opening object ")
        self.look_straight()
        if object_farthest_point == None :
            return
        nav_success = self.agent.nav.go_to_goal(object_farthest_point, self.agent, success_distance, stepBack=True) 
        self.face_object(target_obj)
        self.random_object_pickup(target_obj)

    def random_object_pickup(self,target_obj):
        goal_pixel_coords = self.get_obj_pixels(target_obj,True)
        if goal_pixel_coords == None :
            return False

        for i in range (1,5):        
            for j in range(1,5):

                y = ((np.amax(goal_pixel_coords[0]) - np.amin(goal_pixel_coords[0]))*j/4) + np.amin(goal_pixel_coords[0])
                x = ((np.amax(goal_pixel_coords[1]) - np.amin(goal_pixel_coords[1]))*i/4) + np.amin(goal_pixel_coords[1])
            
                action = {'action':"PickupObject", 'x': x, 'y':y}
                self.agent.game_state.step(action)

                #print ("return status from random pick ups", self.agent.game_state.event.return_status)

                if self.agent.game_state.event.return_status == "SUCCESSFUL" :
                    if self.agent.game_state.event.reward > -1*(self.agent.game_state.number_actions*0.001) +0.5 :
                        self.agent.game_state.trophy_picked_up = True
                        self.update_object_picked_up(target_obj,False)
                    else :
                        self.update_object_picked_up(target_obj,True)
                        action = {'action':'DropObject'}
                        self.agent.game_state.step(action)
                    self.look_straight()
                    #print ("pick up successful trophy pickup status", self.agent.game_state.trophy_picked_up)
                    return True

        return False
                
        
        


    def update_opened_up(self,target_obj):          
        for i,obstacle in enumerate(self.agent.game_state.global_obstacles) :
            if target_obj.id == obstacle.id :
                if self.agent.game_state.event.return_status == "SUCCESSFUL" :
                    self.agent.game_state.global_obstacles[i].is_opened = True
                    return  True
                if self.agent.game_state.event.return_status == "NOT_OPENABLE":
                    self.agent.game_state.global_obstacles[i].is_container = False
                    return False 

        return True


    def explore_all_objects(self):
        #for i,obstacle in enumerate(self.agent.game_state.global_obstacles[2:]) :
        #for i,obstacle in enumerate(self.agent.game_state.global_obstacles[3:]) :
        for i,obstacle in enumerate(self.agent.game_state.global_obstacles) :
            if obstacle.is_possible_container():
                self.go_to_object_and_open(obstacle)
                #print ("goal object visible",self.agent.game_state.goal_object_visible)
                self.look_straight()
                
                if self.agent.game_state.trophy_picked_up == True:
                    return

                if self.agent.game_state.goals_found : 
                    #print ("goal ids", self.agent.game_state.goal_ids)
                    self.pick_up_obstacles(possible_trophy_obstacles=True)
                    return 
                self.look_straight()

    def pick_up_obstacles(self,obstacle_ids=None,all_obstacles=False,possible_trophy_obstacles=False):

        #print ("In pick up obstacles ", all_obstacles)

        if all_obstacles == True :
            for obstacle in self.agent.game_state.global_obstacles :
                #if self.is_possible_trophy(obstacle):
                #print (" for each obejct height", obstacle.height)
                if obstacle.is_possible_trophy():
                    self.go_to_object_and_pick(obstacle)
                if self.agent.game_state.trophy_picked_up == True:
                    return
            return

        obstacles = []
        if possible_trophy_obstacles == True :
            #print ("trying to go to all possible goal objects")
            for obstacle_id in self.agent.game_state.goal_ids :
                obstacles.append(self.get_target_obj(obstacle_id))
        else :
            for obstacle_id in obstacle_ids :
                obstacles.append(self.get_target_obj(obstacle_id))
        for obstacle in obstacles :
            #if self.is_possible_trophy(obstacle):
            if obstacle.is_possible_trophy():
                self.go_to_object_and_pick(obstacle)
            if self.agent.game_state.trophy_picked_up == True:
                return
            

    def get_outermost_map_sorted_coords(self):
        all_coords = []
        for obstacle in self.agent.game_state.global_obstacles[-1:] :
            current_coords= obstacle.get_convex_polygon_coords()
            for x,y in zip(current_coords[0], current_coords[1]):
                all_coords.append((x,y))

        sorted_by_x = sorted(all_coords,key=lambda x: x[0], reverse=True)
        sorted_by_y = sorted(all_coords,key=lambda x: x[1], reverse=True)
        return sorted_by_x, sorted_by_y

    def get_rotated_boundaries(self):
        sorted_by_x,sorted_by_y = self.get_outermost_map_sorted_coords()
        outermost_orig = []
        outermost_orig.append(sorted_by_y[-1])
        outermost_orig.append(sorted_by_x[0])
        outermost_orig.append(sorted_by_y[0])
        outermost_orig.append(sorted_by_x[-1])

        outermost_pts = []
        outermost_pts.append((0,0))
        outermost_pts.append((sorted_by_x[0][0]-sorted_by_y[-1][0],sorted_by_x[0][1]-sorted_by_y[-1][1]))
        outermost_pts.append((sorted_by_y[0][0]-sorted_by_y[-1][0],sorted_by_y[0][1]-sorted_by_y[-1][1]))
        outermost_pts.append((sorted_by_x[-1][0]-sorted_by_y[-1][0],sorted_by_x[-1][1]-sorted_by_y[-1][1]))

        sorted_coords= self.get_polar_sorted_coords(outermost_pts)
        #print ("sorted coords ",sorted_coords)
    
        x1,y1 = sorted_coords[-2]

        rotation = math.atan2((y1), (x1 ))

        rotated_coords = []
        for elem in outermost_pts : 
            rotated_x = elem[0] *math.cos(rotation) - elem[1] * math.sin(rotation)
            rotated_y = elem[0] *math.sin(rotation) + elem[1] * math.cos(rotation)
        
            rotated_coords.append((rotated_x,rotated_y))
        #for point in 
        #rotated_coords.insert(0,outermost_orig[0])
        final_adjusted_coords = []
        for elem in rotated_coords :
            final_adjusted_coords.append((elem[0] + outermost_orig[0][0],elem[1]+outermost_orig[0][1]))

        sorted_by_x = sorted(final_adjusted_coords,key=lambda x: x[0], reverse=True)
        sorted_by_y = sorted(final_adjusted_coords,key=lambda x: x[1], reverse=True)
        xMax = sorted_by_x[0][0]
        xMin = sorted_by_x[-1][0]
        yMax = sorted_by_y[0][1]
        yMin = sorted_by_y[-1][1]
        return [xMin,xMax,yMin,yMax],rotation

    def get_polar_sorted_coords(self,outermost_pts):
        center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), outermost_pts), [len(outermost_pts)] * 2))
        sorted_coords  = sorted(outermost_pts, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
        return sorted_coords 

    def rotate_exploration_points(self, exploration_routine , angle_rotated_by):

        sorted_by_x,sorted_by_y = self.get_outermost_map_sorted_coords()

        bottom_point_x = sorted_by_y[-1][0]/constants.AGENT_STEP_SIZE
        bottom_point_y = sorted_by_y[-1][1]/constants.AGENT_STEP_SIZE
        
        rotation = -angle_rotated_by#[to rotate the points in the opposite direction they were originally rotated in]
        rotated_search_points = []
        for elem in exploration_routine:
            elem_new = [0] *2
            elem_new[0] = elem[0] - bottom_point_x 
            elem_new[1] = elem[1] - bottom_point_y
            
            rotated_x = elem_new[0] *math.cos(rotation) - elem_new[1] * math.sin(rotation)
            rotated_y = elem_new[0] *math.sin(rotation) + elem_new[1] * math.cos(rotation)

            rotated_x += bottom_point_x
            rotated_y += bottom_point_y

            rotated_search_points.append((rotated_x,rotated_y))

        exploration_routine = rotated_search_points[:]
        return exploration_routine
