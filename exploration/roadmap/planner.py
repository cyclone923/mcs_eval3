from visibility_road_map import VisibilityRoadMap,ObstaclePolygon,IncrementalVisibilityRoadMap
import random
import math
import matplotlib.pyplot as plt
from fov import FieldOfView
import shapely.geometry.polygon as sp

show_animation = True
random.seed(2)

class Planner:

	# pose is a triplet x,y,theta (heading)
	def __init__(self, pose, obst=[], robot_radius=5, eps=0.05,maxStep=10):
		self.obstacles = obst
		self.radius = robot_radius
		self.agentX = pose[0]
		self.agentY = pose[1]
		self.agentH = pose[2]
		self.epsilon = eps
		self.maxStep = maxStep
		self.roadmap = IncrementalVisibilityRoadMap(self.radius, do_plot=False)


	def addObstacle(self, obst):
		self.obstacles.append(obst)
		self.roadmap.addObstacle(obst)

	
	def closedLoopPlannerFast(self, goal):

		obstacleCnt = -1
		i = 1

		while math.sqrt((self.agentX-goal[0])**2 + (self.agentY-goal[1])**2) > self.epsilon:
			

			# compute full plan given our current knowledge
			if len(self.obstacles) != obstacleCnt:
				obstacleCnt = len(self.obstacles)
				pathX, pathY = self.roadmap.planning(self.agentX, self.agentY, goal[0], goal[1])
				i = 1

			# execute a small step along that plan by
			# turning to face the first waypoint
			dX = pathX[i]-self.agentX
			dY = pathY[i]-self.agentY
			angleFromAxis = math.atan2(dX, dY)
			
			#taking at most a step of size 0.1
			distToWaypoint = math.sqrt((self.agentX-pathX[i])**2 + (self.agentY-pathY[i])**2)
			stepSize = min(self.maxStep, distToWaypoint)
			
			if distToWaypoint < self.epsilon:
				i+=1
				#continue

			yield (stepSize, angleFromAxis)

	def closedLoopPlanner(self, goal):

		roadmap = VisibilityRoadMap(self.radius, do_plot=False)
		obstacleCnt = -1
		i = 1

		while math.sqrt((self.agentX-goal[0])**2 + (self.agentY-goal[1])**2) > self.epsilon:
			
			# compute full plan given our current knowledge
			if len(self.obstacles) != obstacleCnt:
				obstacleCnt = len(self.obstacles)
				pathX, pathY = roadmap.planning(self.agentX, self.agentY, goal[0], goal[1], self.obstacles)
				i = 1

			# execute a small step along that plan by
			# turning to face the first waypoint
			dX = pathX[i]-self.agentX
			dY = pathY[i]-self.agentY
			angleFromAxis = math.atan2(dX, dY)
			
			#taking at most a step of size 0.1
			distToWaypoint = math.sqrt((self.agentX-pathX[i])**2 + (self.agentY-pathY[i])**2)
			stepSize = min(self.maxStep, distToWaypoint)
			
			if distToWaypoint < self.epsilon:
				i+=1

			yield (stepSize, angleFromAxis)
			



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


	for i in range(5):
		# start and goal position
		# sx, sy = random.randrange(-125,125), random.randrange(-125,125)  # [m]
		# gx, gy = random.randrange(-125,125), random.randrange(-125,125)  # [m]

		sx, sy = -40.0, -100.0  # [m]
		gx, gy = -40.0, 80.0  # [m]


		robot_radius = 5.0  # [m]

		cnt = 5
		obstacles=[]
		for i in range(cnt):
			obstacles.append(genRandomRectangle())
		obstacles.append(ObstaclePolygon([150,-150,-150,150],[150,150,-150,-150]))
		
		visible = [False]*(len(obstacles))

		if show_animation:  # pragma: no cover
			plt.plot(sx, sy, "or")
			plt.plot(gx, gy, "ob")
			for ob in obstacles:
				ob.plot()
			plt.axis("equal")
			
			plt.pause(0.1)

		#create a planner and initalize it with the agent's pose
		plan = Planner( [sx,sy,0], [])
		world = sp.Polygon()
		try:
			fov = FieldOfView( [sx,sy,0], 40/180.0*math.pi, obstacles)
			#fovChecker = FieldOfView( [sx,sy,0], 40/180.0*math.pi, [obstacles[-1]])
			fovChecker = FieldOfView( [sx,sy,0], 355/180.0*math.pi, obstacles)
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
						fovChecker.obstacle.append(obstacles[i])
						visible[i] = True

				fov.agentX = plan.agentX
				fov.agentY = plan.agentY
				fov.agentH = plan.agentH
				poly = fov.getFoVPolygon(100)
				

				view = sp.Polygon(zip(poly.x_list, poly.y_list))
				world = world.union(view)
				
				cx, cy, h = random.randrange(-125,125), random.randrange(-125,125) , random.random()*2*math.pi# [m]
				fovChecker.agentX = cx
				fovChecker.agentY = cy
				fovChecker.agentH = h
				checkPoly = fovChecker.getFoVPolygon(10)

				newPoly = sp.Polygon(zip(checkPoly.x_list, checkPoly.y_list))
				newPoly = newPoly.difference(world)
				

				if show_animation:
					plt.cla()
					plt.plot(plan.agentX, plan.agentY, "or")
					plt.plot(gx, gy, "ob")
					poly.plot("-r")
					
					if world.geom_type == 'MultiPolygon':
						for i in range(len(world)):
							pts = world[i].exterior.coords
							plt.fill([p[0] for p in pts], [p[1] for p in pts], "-b")	
					else:
						pts = world.exterior.coords
						plt.fill([p[0] for p in pts], [p[1] for p in pts], "-b")

					if newPoly.geom_type == 'MultiPolygon':
						for i in range(len(newPoly)):
							pts = newPoly[i].exterior.coords
							plt.fill([p[0] for p in pts], [p[1] for p in pts], "-g")	
					else:
						pts = newPoly.exterior.coords
						plt.fill([p[0] for p in pts], [p[1] for p in pts], "-g")

					plt.text(-125,-125,"Random View Area: {:0.5}".format(newPoly.area))
				
					for i in range(len(obstacles)):
						if visible[i]:
							obstacles[i].plot("-g")
						else:
							obstacles[i].plot("-k")
					
					plt.axis("equal")
					plt.pause(1)
		except:
			continue
		
    #if show_animation:  # pragma: no cover
    #    plt.plot(rx, ry, "-r")
    #    plt.pause(0.1)
    #    plt.show()


if __name__ == '__main__':
    main()
