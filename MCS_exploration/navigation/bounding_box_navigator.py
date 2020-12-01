#from tasks.bonding_box_navigation_mcs.visibility_road_map import ObstaclePolygon,IncrementalVisibilityRoadMap
from MCS_exploration.navigation.visibility_road_map import ObstaclePolygon,IncrementalVisibilityRoadMap
from MCS_exploration.navigation.discrete_action_planner import DiscreteActionPlanner
import random
import math
import matplotlib.pyplot as plt
from MCS_exploration.navigation.fov import FieldOfView
import cover_floor
import time
from shapely.geometry import Point, MultiPoint, LineString
import numpy as np
from descartes import PolygonPatch

SHOW_ANIMATION = True
LIMIT_STEPS = 350

class BoundingBoxNavigator:

	# pose is a triplet x,y,theta (heading)
	def __init__(self, robot_radius, maxStep=0.1):
		self.agentX = None
		self.agentY = None
		self.agentH = None
		self.epsilon = None

		self.scene_obstacles_dict = {}
		self.scene_obstacles_dict_roadmap = {}
		self.scene_plot = None

		self.radius = robot_radius
		self.maxStep = maxStep
		self.current_nav_steps = 0

	

	def step_towards_point(self, agent, x,y, backwards=False):
		dX = x - self.agentX
		dY = y - self.agentY
		heading = math.atan2(dX, dY)


		rotation_degree = heading / (2 * math.pi) * 360 - agent.game_state.rotation

		if backwards:
			rotation_degree -= 180
		
		if np.abs(rotation_degree) > 360:
			rotation_degree = np.sign(rotation_degree) * (np.abs(rotation_degree) - 360)
		if rotation_degree > 180:
			rotation_degree -= 360
		if rotation_degree < -180:
			rotation_degree += 360

		n = int(abs(round(rotation_degree)) // 10)
		
		action_list = []

		for _ in range(n):
			action_list.append( {'action': 'RotateLeft'} if rotation_degree > 0 else {'action': 'RotateRight'})
		if math.sqrt( dX**2 + dY**2) >= 0.09:
			if backwards:
				action_list.append({'action':"MoveBack"})
			else:
				action_list.append({'action':"MoveAhead"})
		
		for act in action_list:
			agent.game_state.step(act)
			rotation = agent.game_state.rotation
			self.agentX = agent.game_state.position['x']
			self.agentY = agent.game_state.position['z']
			self.agentH = rotation / 360 * (2 * math.pi)
			cover_floor.update_seen(self.agentX, self.agentY, agent.game_state, rotation, 42.5,
								self.scene_obstacles_dict.values())
			self.current_nav_steps += 1
		return agent.game_state.event.return_status != "SUCCESSFUL"

	def clear_obstacle_dict(self):
		self.scene_obstacles_dict = {}

	def reset(self):
		self.scene_obstacles_dict = {}
		self.agentX = None
		self.agentY = None
		self.agentH = None

	def add_obstacle_from_step_output(self, step_output):
		def get_bd_point(dimensions):
			bd_point = set()
			for i in range(0, 8):
				x, z = dimensions[i]['x'], dimensions[i]['z']
				if (x, z) not in bd_point:
					bd_point.add((x, z))

			poly = MultiPoint(sorted(bd_point)).convex_hull
			x_list, z_list = poly.exterior.coords.xy
			return x_list, z_list

		for obj in step_output.object_list:
			if len(obj.dimensions) > 0 and obj.uuid not in self.scene_obstacles_dict and obj.visible:
				x_list, z_list = get_bd_point(obj.dimensions)
				self.scene_obstacles_dict[obj.uuid] = ObstaclePolygon(x_list, z_list)
				self.scene_obstacles_dict_roadmap[obj.uuid] = 0
			if obj.held:
				del self.scene_obstacles_dict[obj.uuid]

		for obj in step_output.structural_object_list:
			if len(obj.dimensions) > 0 and obj.uuid not in self.scene_obstacles_dict and obj.visible:
				if obj.uuid == "ceiling" or obj.uuid == "floor":
					continue
				x_list, z_list = get_bd_point(obj.dimensions)
				self.scene_obstacles_dict[obj.uuid] = ObstaclePolygon(x_list, z_list)
				self.scene_obstacles_dict_roadmap[obj.uuid] = 0

	def add_obstacle_from_bounding_boxes(self, bounding_boxes):
		if bounding_boxes == None  :
			return
		obj_id = int(0)
		self.scene_obstacles_dict = {}
		self.scene_obstacles_dict_roadmap = {}
		#print ("in the new obstacle calculation function", len(list(bounding_boxes)))
		if bounding_boxes.geom_type != "MultiPolygon" :
			self.scene_obstacles_dict[obj_id] = ObstaclePolygon(bounding_boxes.exterior.coords.xy[0], bounding_boxes.exterior.coords.xy[1])
			self.scene_obstacles_dict_roadmap[obj_id] = 0
			return
		for bounding_box in bounding_boxes:
			#print ("in the new obstacle calculation function", len(list(bounding_boxes)))
			#obstacle_polygon = ObstaclePolygon(bounding_box[0],bounding_box[1]).simplify(2)
			self.scene_obstacles_dict[obj_id] = ObstaclePolygon(bounding_box.exterior.coords.xy[0], bounding_box.exterior.coords.xy[1])
			#print (obstacle_polygon.exterior.coords.xy)
			self.scene_obstacles_dict_roadmap[obj_id] = 0
			#print ("time taken till creating FOV after roadmap",time_taken_part_1)


			SHOW_ANIMATION = False
			if SHOW_ANIMATION:
			#if True:
				plt.cla()
				plt.xlim((-7, 7))
				plt.ylim((-7, 7))
				plt.gca().set_xlim((-7, 7))
				plt.gca().set_ylim((-7, 7))

				circle = plt.Circle((self.agentX, self.agentY), radius=self.radius, color='r')
				plt.gca().add_artist(circle)

				for obstacle in self.scene_obstacles_dict.values():
					obstacle.plot("green")
				#self.scene_obstacles_dict[obj_id].plot("green")

				plt.axis("equal")
				plt.pause(1)
				print ("in show animation and trying to save fig")
				#plt.savefig("bounding_box_add_shapely_output.png")
			obj_id += 1
		#print("obj_id = ", obj_id)

	def add_obstacle_from_global_obstacles(self, global_obstacles):
		if len(global_obstacles) == 0  :
			return
		obj_id = int(0)
		self.scene_obstacles_dict = {}
		self.scene_obstacles_dict_roadmap = {}
		for obstacle in global_obstacles :
			bounding_boxes = obstacle.get_bounding_box()
			if bounding_boxes.geom_type != "MultiPolygon" :
				self.scene_obstacles_dict[obj_id] = ObstaclePolygon(bounding_boxes.exterior.coords.xy[0], bounding_boxes.exterior.coords.xy[1])
				self.scene_obstacles_dict_roadmap[obj_id] = 0
				obj_id += 1
			else :
				for bounding_box in bounding_boxes:
					start_time = time.time()
					self.scene_obstacles_dict[obj_id] = ObstaclePolygon(bounding_box.exterior.coords.xy[0], bounding_box.exterior.coords.xy[1])
					self.scene_obstacles_dict_roadmap[obj_id] = 0
					#print ("time taken till creating FOV after roadmap", time.time()-start_time )
					obj_id += 1
        
	#def go_to_goal(self, nav_env, goal, success_distance, epsd_collector=None, frame_collector=None):
	def can_add_obstacle(self, obstacle, goal):
		#return not obstacle.contains_goal(goal) and obstacle.distance(Point(goal[0], goal[1])) > 0.5
		return True

	def get_obstacles(self):
		return self.scene_obstacles_dict

	def go_to_goal(self, goal_pose, agent, success_distance):

		self.current_nav_steps = 0
		self.agentX = agent.game_state.position['x']
		self.agentY = agent.game_state.position['z']
		self.agentH = agent.game_state.rotation / 360 * (2 * math.pi)
		self.epsilon = success_distance

		gx, gy = goal_pose[0], goal_pose[1]
		for obstacle_key, obstacle in self.scene_obstacles_dict.items():
			self.scene_obstacles_dict_roadmap[obstacle_key] = 0

		plan = []
		collision = False
		
		while True:
			start_time = time.time()

			#check if we are close enough
			dis_to_goal = math.sqrt((self.agentX-gx)**2 + (self.agentY-gy)**2)
			if dis_to_goal < self.epsilon:
				break
			obs = []

			#add any new obstacles
			for obstacle_key, obstacle in self.scene_obstacles_dict.items():
				if self.scene_obstacles_dict_roadmap[obstacle_key] == 0:
					#print ("not added obstacle", self.current_nav_steps)
					if self.can_add_obstacle(obstacle, (gx, gy)):
						#print ("adding new obstacles ", self.current_nav_steps)
						self.scene_obstacles_dict_roadmap[obstacle_key] =1
						obs.append(obstacle)
						#roadmap.addObstacle(obstacle)
			roadmap = DiscreteActionPlanner(self.radius+0.05, obs, self.epsilon)
			
			#check if the plan is still valid / exists and replan if not
			if not roadmap.validPlan(plan, (self.agentX, self.agentY)):
				plan_x, plan_y = roadmap.planning(self.agentX, self.agentY, gx, gy)
				plan = list(zip(plan_x, plan_y))
				

			#take action if the plan provides one
			collision = True
			if len(plan) > 0:
				x,y = plan.pop(0)
				collision = self.step_towards_point(agent, x,y)

			
			#if we collide or produced no plan, try to un-stick ourselves
			if collision:
				path_x, path_y = roadmap.getUnstuckPath(self.agentX, self.agentY) 
				for x,y in zip(path_x, path_y):
					self.step_towards_point(agent, x, y, backwards=True)
				plan = []
				
			
			#plot out the state if enabled
			if SHOW_ANIMATION:
				fov = FieldOfView([self.agentX, self.agentY, 0], 42.5 / 180.0 * math.pi, self.scene_obstacles_dict.values())
				fov.agentX = self.agentX
				fov.agentY = self.agentY
				fov.agentH = self.agentH
				poly = fov.getFoVPolygon(15)

				plt.cla()
				plt.xlim((-7, 7))
				plt.ylim((-7, 7))
				plt.gca().set_xlim((-7, 7))
				plt.gca().set_ylim((-7, 7))

				circle = plt.Circle((self.agentX, self.agentY), radius=self.radius, color='r')
				plt.gca().add_artist(circle)
				plt.plot(gx, gy, "x")
				poly.plot("red")

				for obstacle in self.scene_obstacles_dict.values():
					obstacle.plot("green")

				if len(plan) > 1:
					linePlan = LineString( plan ).buffer(self.radius)
					patch1 = PolygonPatch(linePlan,fc='grey', ec="black", alpha=0.2, zorder=1)
					plt.gca().add_patch(patch1)
					
				plt.axis("equal")
				plt.pause(0.001)


			end_time = time.time()

			

			if agent.game_state.number_actions >= 595 :
				print("Reached overall STEPS limit")
				return

			if self.current_nav_steps >= LIMIT_STEPS:
				print("Reach LIMIT STEPS")
				return False


		return True
