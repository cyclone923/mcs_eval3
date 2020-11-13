#from tasks.bonding_box_navigation_mcs.visibility_road_map import ObstaclePolygon,IncrementalVisibilityRoadMap
from MCS_exploration.navigation.visibility_road_map import ObstaclePolygon,IncrementalVisibilityRoadMap
import random
import math
import matplotlib.pyplot as plt
from MCS_exploration.navigation.fov import FieldOfView
import cover_floor
import time
from shapely.geometry import Point, Polygon
import numpy as np

SHOW_ANIMATION = False
LIMIT_STEPS = 350

class BoundingBoxNavigator:

	# pose is a triplet x,y,theta (heading)
	def __init__(self, robot_radius=0.4, maxStep=0.25):
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

	
	def get_one_step_move(self, goal, roadmap):

		try :
			pathX, pathY = roadmap.planning(self.agentX, self.agentY, goal[0], goal[1])
		except ValueError:
			return None,None

		# execute a small step along that plan by
		# turning to face the first waypoint
		if len(pathX) == 1 and len(pathY) == 1:
			i = 0
		else:
			i = 1
		dX = pathX[i]-self.agentX
		dY = pathY[i]-self.agentY
		angleFromAxis = math.atan2(dX, dY)
			
		#taking at most a step of size 0.1
		distToFirstWaypoint = math.sqrt((self.agentX-pathX[i])**2 + (self.agentY-pathY[i])**2)
		stepSize = min(self.maxStep, distToFirstWaypoint)

		return stepSize, angleFromAxis

	def clear_obstacle_dict(self):
		self.scene_obstacles_dict = {}

	def reset(self):
		self.clear_obstacle_dict()
		self.agentX = None
		self.agentY = None
		self.agentH = None

	def add_obstacle_from_step_output(self, step_output):
		# if step_output is None:
		# 	return
		for obj in step_output.object_list:
			if len(obj.dimensions) > 0 and obj.uuid not in self.scene_obstacles_dict and obj.visible:
				x_list = []
				y_list = []
				for i in range(4, 8):
					x_list.append(obj.dimensions[i]['x'])
					y_list.append(obj.dimensions[i]['z'])
				self.scene_obstacles_dict[obj.uuid] = ObstaclePolygon(x_list, y_list)
				self.scene_obstacles_dict_roadmap[obj.uuid] = 0
			if obj.held:
				del self.scene_obstacles_dict[obj.uuid]

		for obj in step_output.structural_object_list:
			if len(obj.dimensions) > 0 and obj.uuid not in self.scene_obstacles_dict and obj.visible:
				if obj.uuid == "ceiling" or obj.uuid == "floor":
					continue
				x_list = []
				y_list = []
				for i in range(4, 8):
					x_list.append(obj.dimensions[i]['x'])
					y_list.append(obj.dimensions[i]['z'])
				self.scene_obstacles_dict[obj.uuid] = ObstaclePolygon(x_list, y_list)
				self.scene_obstacles_dict_roadmap[obj.uuid] = 0

	#def go_to_goal(self, nav_env, goal, success_distance, epsd_collector=None, frame_collector=None):
	def can_add_obstacle(self, obstacle, goal):
		return not obstacle.contains_goal(goal) and obstacle.distance(Point(goal[0], goal[1])) > 0.5

	def go_to_goal(self, goal_pose, agent, success_distance):

		self.current_nav_steps = 0
		self.agentX = agent.game_state.event.position['x']
		self.agentY = agent.game_state.event.position['z']
		self.agentH = agent.game_state.event.rotation / 360 * (2 * math.pi)
		self.epsilon = success_distance

		gx, gy = goal_pose[0], goal_pose[1]
		sx, sy = self.agentX, self.agentY
		roadmap = IncrementalVisibilityRoadMap(self.radius, do_plot=False)
		for obstacle_key, obstacle in self.scene_obstacles_dict.items():
			self.scene_obstacles_dict_roadmap[obstacle_key] = 0

		for obstacle_key, obstacle in self.scene_obstacles_dict.items():
			if self.can_add_obstacle(obstacle, (gx, gy)):
				self.scene_obstacles_dict_roadmap[obstacle_key] = 1
				roadmap.addObstacle(obstacle)

		while True:
			start_time = time.time()

			for obstacle_key, obstacle in self.scene_obstacles_dict.items():
				if self.scene_obstacles_dict_roadmap[obstacle_key] == 0:
					#print ("not added obstacle", self.current_nav_steps)
					if self.can_add_obstacle(obstacle, (gx, gy)):
						#print ("adding new obstacles ", self.current_nav_steps)
						self.scene_obstacles_dict_roadmap[obstacle_key] =1
						roadmap.addObstacle(obstacle)

			goal_obj_bonding_box = None
			for id, box in self.scene_obstacles_dict.items():
				if box.contains_goal((gx,gy)):
					goal_obj_bonding_box = box.get_goal_bonding_box_polygon()
					break
			if not goal_obj_bonding_box:
				dis_to_goal = math.sqrt((self.agentX-gx)**2 + (self.agentY-gy)**2)
				# print("Dis to goal point {:.3f}, Suc dis: {:.3f}".format(dis_to_goal, success_distance))
			else:
				dis_to_goal = goal_obj_bonding_box.distance(Point(self.agentX, self.agentY))
				# print("Dis to goal bonding box {:.3f}, Suc dis: {:.3f}".format(dis_to_goal, success_distance))

			if dis_to_goal < self.epsilon:
				break

			end_time = time.time()
			roadmap_creatiion_time = end_time-start_time
			#print("roadmap creation time", roadmap_creatiion_time)

			start_time = time.time()

			fov = FieldOfView([sx, sy, 0], 42.5 / 180.0 * math.pi, self.scene_obstacles_dict.values())
			fov.agentX = self.agentX
			fov.agentY = self.agentY
			fov.agentH = self.agentH
			poly = fov.getFoVPolygon(15)
			end_time = time.time()

			time_taken_part_1 = end_time-start_time

			#print ("time taken till creating FOV after roadmap",time_taken_part_1)


			if SHOW_ANIMATION:
				plt.cla()
				plt.xlim((-7, 7))
				plt.ylim((-7, 7))
				plt.gca().set_xlim((-7, 7))
				plt.gca().set_ylim((-7, 7))

				# plt.plot(self.agentX, self.agentY, "or")
				circle = plt.Circle((self.agentX, self.agentY), radius=self.radius, color='r')
				plt.gca().add_artist(circle)
				plt.plot(gx, gy, "x")
				poly.plot("red")

				for obstacle in self.scene_obstacles_dict.values():
					obstacle.plot("green")

				plt.axis("equal")
				plt.pause(0.001)

			start_time = time.time()
			stepSize, heading = self.get_one_step_move([gx, gy], roadmap)
			end_time = time.time()

			get_one_step_move_time = end_time-start_time
			#print ("get_one_step_move_time taken", get_one_step_move_time)

			if stepSize == None and heading == None:
				print("Planning Fail")
				return  False

			# needs to be replaced with turning the agent to the appropriate heading in the simulator, then stepping.
			# the resulting agent position / heading should be used to set plan.agent* values.

			start_time = time.time()

			rotation_degree = heading / (2 * math.pi) * 360 - agent.game_state.event.rotation

			if np.abs(rotation_degree) > 360:
				rotation_degree = np.sign(rotation_degree) * (np.abs(rotation_degree) - 360)
			if rotation_degree > 180:
				rotation_degree -= 360
			if rotation_degree < -180:
				rotation_degree += 360

			n = int(abs(rotation_degree) // 10)
			if rotation_degree > 0:
				for _ in range(n):
					agent.game_state.step({'action': 'RotateLeft'})
			else:
				for _ in range(n):
					agent.game_state.step({'action': 'RotateRight'})

			start_time = time.time()

			rotation = agent.game_state.event.rotation
			self.agentX = agent.game_state.event.position['x']
			self.agentY = agent.game_state.event.position['z']
			self.agentH = rotation / 360 * (2 * math.pi)
			self.current_nav_steps += 1
			cover_floor.update_seen(self.agentX, self.agentY, agent.game_state, rotation, 42.5,
									self.scene_obstacles_dict.values())


			end_time = time.time()
			action_processing_time_taken = end_time-start_time
			#print ("action processing taken", action_processing_time_taken)

			#if abs(rotation_degree) > 1e-2:
			#	if 360 - abs(rotation_degree) > 1e-2:
			#		continue

			#agent.game_state.step(action="MoveAhead", amount=0.5)
			action={'action':"MoveAhead"}
			agent.step(action)
			rotation = agent.game_state.event.rotation
			self.agentX = agent.game_state.event.position['x']
			self.agentY = agent.game_state.event.position['z']
			self.agentH = rotation / 360 * (2 * math.pi)

			cover_floor.update_seen(self.agentX, self.agentY, agent.game_state, rotation, 42.5,
									self.scene_obstacles_dict.values())


			self.current_nav_steps += 1

			if agent.game_state.number_actions >= 595 :
				return

			if agent.game_state.goals_found == True:
				return

			if self.current_nav_steps >= LIMIT_STEPS:
				print("Reach LIMIT STEPS")
				return False

		return True



def genRandomRectangle():
    width = random.randrange(5,50)
    height = random.randrange(5,50)
    botLeftX = random.randrange(1,100)
    botRightX = random.randrange(1,100)
    theta = random.random()*2*math.pi

    x = [random.randrange(1,50)]
    y = [random.randrange(1,50)]

    x.append(x[-1]+width)
    y.append(y[-1])

    x.append(x[-1])
    y.append(y[-1]+height)

    x.append(x[-1]-width)
    y.append(y[-1])

    for i in range(4):
        tx = x[i]*math.cos(theta) - y[i]*math.sin(theta)
        ty = x[i]*math.sin(theta) + y[i]*math.cos(theta)
        x[i] = tx
        y[i] = ty

    return ObstaclePolygon(x,y)

def main():
	print(__file__ + " start!!")


	for i in range(20):
		# start and goal position
		sx, sy = random.randrange(-100,-80), random.randrange(-100,-80)  # [m]
		gx, gy = random.randrange(80,100), random.randrange(80,100)  # [m]

		robot_radius = 5.0  # [m]

		cnt = 15
		obstacles=[]
		for i in range(cnt):
			obstacles.append(genRandomRectangle())
		visible = [False]*cnt

		if SHOW_ANIMATION:  # pragma: no cover
			plt.xlim((-100, 100))
			plt.ylim((-100, 100))
			plt.plot(sx, sy, "or")
			plt.plot(gx, gy, "ob")
			for ob in obstacles:
				ob.plot()
			plt.axis("equal")
			
			#plt.pause(0.1)

		#create a planner and initalize it with the agent's pose
		plan = BoundingBoxNavigator( [sx,sy,0], [])


		fov = FieldOfView( [sx,sy,0], 60/180.0*math.pi, obstacles)
			
		for stepSize, heading in plan.closedLoopPlannerFast([gx,gy]):
			
			#needs to be replaced with turning the agent to the appropriate heading in the simulator, then stepping.
			#the resulting agent position / heading should be used to set plan.agent* values.
			plan.agentH = heading
			plan.agentX = plan.agentX + stepSize*math.sin(plan.agentH)
			plan.agentY = plan.agentY + stepSize*math.cos(plan.agentH)

			#any new obstacles that were observed during the step should be added to the planner
			for i in range(len(obstacles)):
				if not visible[i] and obstacles[i].minDistanceToVertex(plan.agentX, plan.agentY) < 30:
					plan.addObstacle(obstacles[i])
					visible[i] = True

			fov.agentX = plan.agentX
			fov.agentY = plan.agentY
			fov.agentH = plan.agentH
			poly = fov.getFoVPolygon(100)
			

			if SHOW_ANIMATION:
				plt.cla()
				plt.xlim((-100, 100))
				plt.ylim((-100, 100))
				plt.gca().set_xlim((-100, 100))
				plt.gca().set_ylim((-100, 100))

				plt.plot(plan.agentX, plan.agentY, "or")
				plt.plot(gx, gy, "ob")
				poly.plot("red")
			
				for i in range(len(obstacles)):
					if visible[i]:
						obstacles[i].plot("green")
					else:
						obstacles[i].plot("black")
				
				plt.axis("equal")
				plt.pause(0.1)

    
    #if SHOW_ANIMATION:  # pragma: no cover
    #    plt.plot(rx, ry, "-r")
    #    plt.pause(0.1)
    #    plt.show()


if __name__ == '__main__':
    main()
