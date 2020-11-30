from exploration.roadmap.discrete_action_planner import DiscreteActionPlanner
import math
from shapely.geometry import Point
import numpy as np

SHOW_ANIMATION = True
LIMIT_STEPS = 350



class Navigator():

	def __init__(self, agent, robot_radius, max_step, success_distance):

		self.agent = agent
		self.scene_obstacles_in_roadmap = {}
		self.goal = None
		self.path_x, self.path_z = None, None
		self.goal_bonding_box = None
		self.success_distance = success_distance

		self.radius = robot_radius
		self.max_step = max_step


	def get_one_step_move(self, roadmap):
		gx, gz, _ = self.goal
		try :
			self.path_x, self.path_z = roadmap.planning(self.agent.agent_x, self.agent.agent_z, gx, gz)
		except ValueError:
			return None, None

		# execute a small step along that plan by turning to face the first waypoint
		if len(self.path_x) == 1 and len(self.path_z) == 1:
			i = 0
		else:
			i = 1

		dX = self.path_x[i]-self.agent.agent_x
		dZ = self.path_z[i]-self.agent.agent_z
		angleFromAxis = math.atan2(dX, dZ)
			
		#taking at most a step of size 0.1
		distToFirstWaypoint = math.sqrt((self.agent.agent_x-self.path_x[i])**2 + (self.agent.agent_z-self.path_z[i])**2)
		stepSize = min(self.max_step, distToFirstWaypoint)

		return stepSize, angleFromAxis


	def can_add_obstacle(self, obstacle, goal, eps=1e-4):
		contain = obstacle.contains(Point(goal))
		distance = obstacle.distance(Point(goal[0], goal[1]))
		return not contain and distance > eps

	def process_state(self, roadmap):
		gx, gz, _ = self.goal
		dis_to_goal = math.sqrt((self.agent.agent_x - gx) ** 2 + (self.agent.agent_z - gz) ** 2)
		print("Dis: {}".format(dis_to_goal))

		for obstacle_key, obstacle in self.agent.scene_obstacles.items():

			if obstacle_key not in self.scene_obstacles_in_roadmap:
				self.scene_obstacles_in_roadmap[obstacle_key] = 0

			if self.scene_obstacles_in_roadmap[obstacle_key] == 0:
				if self.can_add_obstacle(obstacle, (gx, gz)):
					self.scene_obstacles_in_roadmap[obstacle_key] = 1
					roadmap.addObstacle(obstacle)
				else:
					assert self.goal_bonding_box is None or self.goal_bonding_box is obstacle
					self.goal_bonding_box = obstacle

		return dis_to_goal < self.success_distance


	def go_to_goal(self, goal_pose):

		self.goal = goal_pose
		print("Go from {} to {}".format((self.agent.agent_x, self.agent.agent_z), self.goal))
		roadmap = DiscreteActionPlanner(self.radius, [])

		current_nav_steps = 0
		while True:
			if self.process_state(roadmap):
				_, _, target_r = self.goal
				if target_r:
					self.controlled_rotate(target_r)
				break

			stepSize, heading = self.get_one_step_move(roadmap)

			if stepSize == None and heading == None:
				print("Planning Fail")
				break

			# needs to be replaced with turning the agent to the appropriate heading in the simulator, then stepping.
			# the resulting agent position / heading should be used to set plan.agent* values.

			used_steps = self.controlled_rotate(heading)
			current_nav_steps += used_steps

			self.agent.step(action='MoveAhead')
			current_nav_steps += 1

			if current_nav_steps >= LIMIT_STEPS:
				print("Navigation Reach Limit Steps")
				break

		self.goal = None

	def controlled_rotate(self, target_heading):

		rotation_degree = (target_heading - self.agent.agent_rotation_radian) * 360 / (2 * math.pi)
		used_steps = 0
		if np.abs(rotation_degree) > 360:
			rotation_degree = np.sign(rotation_degree) * (np.abs(rotation_degree) - 360)
		if rotation_degree > 180:
			rotation_degree -= 360
		if rotation_degree < -180:
			rotation_degree += 360

		n = int(abs(rotation_degree) // 10)

		if rotation_degree > 0:
			for _ in range(n):
				self.agent.step(action='RotateRight')
				used_steps += 1
		else:
			for _ in range(n):
				self.agent.step(action='RotateLeft')
				used_steps += 1

		return used_steps


