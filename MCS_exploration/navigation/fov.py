#from visibility_road_map import ObstaclePolygon
#from geometry import Geometry
from MCS_exploration.navigation.visibility_road_map import ObstaclePolygon
from MCS_exploration.navigation.geometry import Geometry
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import shapely.geometry as sp
from shapely.prepared import prep
from shapely.ops import unary_union

from shapely import speedups
if speedups.available:
	speedups.enable()

class FieldOfView:

	def __init__(self, pose, hvof, obs):
		self.agentX = pose[0]
		self.agentY = pose[1]
		self.agentH = pose[2]
		self.HVoF = hvof
		self.obstacle = obs
		self.poly = unary_union([o.boundary for o in obs])
		

	def getFoVPolygon(self, maxLen=15, eps=0.000001):
		poly_X = []
		poly_Y = []
		poly_angle = []

		lAngle = (self.agentH-self.HVoF/2)
		rAngle = (self.agentH+self.HVoF/2)
		p1 = Geometry.Point(self.agentX, self.agentY)
		p2L = Geometry.Point(p1.x + maxLen*math.sin(lAngle), p1.y + maxLen*math.cos(lAngle))
		p2R = Geometry.Point(p1.x + maxLen*math.sin(rAngle), p1.y + maxLen*math.cos(rAngle))

		#cast on HFOV lines
		if True:	
			for i in np.arange(0, 1.1, 0.1):
				v = Geometry.Point(p1.x + maxLen*math.sin(lAngle+i*self.HVoF), p1.y + maxLen*math.cos(lAngle+i*self.HVoF))
				theta = (np.arctan2( v.y-p1.y,  v.x-p1.x))
				x,y = self.castRayShapely(theta, maxLen,"-b")
				poly_X.append(x)
				poly_Y.append(y)
				poly_angle.append(theta)
		#print ("polyX b4 obstacle part", poly_X)
		#print ("polyY b4 obstacle part", poly_Y)

		if True:
			# find any points in the FoV
			for obs in self.obstacle:

				#check if any point lies in the viewing window
				for x,y in zip(obs.x_list, obs.y_list):
					v = Geometry.Point(x,y)

					if self.isLeftOfLine(p1, p2R, v) and not self.isLeftOfLine(p1, p2L, v):
						#cast at point
						#plt.plot(v.x,v.y,"or")
						#theta = np.arctan( (v.x-p1.x) / (v.y-p1.y))
						#theta = (np.arctan2( v.y-p1.y,  v.x-p1.x))
						x,y = self.castRayShapely(theta, maxLen)
						poly_X.append(x)
						poly_Y.append(y)
						poly_angle.append(theta)

						#cast with jitter
						theta = (theta - eps)
						x,y = self.castRayShapely(theta, maxLen)
						v = Geometry.Point(x,y)
						if self.isLeftOfLine(p1, p2R, v) and not self.isLeftOfLine(p1, p2L, v):
							poly_X.append(x)
							poly_Y.append(y)
							poly_angle.append(theta)

						theta = (theta + 2*eps)
						x,y = self.castRayShapely(theta, maxLen)
						v = Geometry.Point(x,y)
						if self.isLeftOfLine(p1, p2R, v) and not self.isLeftOfLine(p1, p2L, v):
							poly_X.append(x)
							poly_Y.append(y)
							poly_angle.append(theta)

		#poly_angle = [2*math.pi-x if x < 0 else x for x in poly_angle]
		# print(poly_angle)

		indx = sorted(range(len(poly_angle)), key=lambda x: (poly_angle[x]+self.agentH) % (2*np.pi))
		poly_X = [p1.x] + list(np.array(poly_X)[indx])+ [p1.x]
		poly_Y = [p1.y] + list(np.array(poly_Y)[indx]) + [p1.y]
		#print ("polyX", poly_X)
		#print ("polyY", poly_Y)
		return ObstaclePolygon(poly_X, poly_Y)


	def castRayShapely(self, angle, maxLen, clr="-g"):

		p1 = sp.Point( [ float(self.agentX), float(self.agentY) ] )
		p2 = sp.Point( [p1.x + maxLen*np.cos(angle), p1.y + maxLen*np.sin(angle) ])

		intersections = self.poly.intersection(sp.LineString([p1,p2]))
		
		if intersections.is_empty:
			return p2.x, p2.y
		else:
			if isinstance(intersections, sp.Point):
				return intersections.coords[0]

			elif isinstance(intersections,sp.LineString):
				points = list(intersections.coords)
				x,y,d = min(   [ (x,y, math.sqrt( (x-self.agentX)**2 + (y-self.agentY)**2)) for x,y in points], key=lambda a: a[2])
				return x,y

			elif isinstance(intersections, sp.MultiPoint) or isinstance(intersections, sp.GeometryCollection):
				points = [list(p.coords)[0] for p in list(intersections)]
				x,y,d = min(   [ (x,y, math.sqrt( (x-self.agentX)**2 + (y-self.agentY)**2)) for x,y in points], key=lambda a: a[2])
				return x,y

			else:
				print(intersections)
				raise ValueError(type(intersections))



	def castRay(self, angle, maxLen, clr="-g"):
		p1 = Geometry.Point(float(self.agentX), float(self.agentY))
		p2 = Geometry.Point(p1.x + maxLen*np.cos(angle), p1.y + maxLen*np.sin(angle))

		minD = math.inf
		minX = p2.x
		minY = p2.y
		for obs in self.obstacle:
			for i in range(len(obs.x_list) - 1):
				o1 = Geometry.Point(obs.x_list[i], obs.y_list[i])
				o2 = Geometry.Point(obs.x_list[i + 1], obs.y_list[i + 1])

				try:
					x,y = self.intersect(p1,p2,o1,o2)
					d = math.sqrt( (x-p1.x)**2+(y-p1.y)**2 )
					#plt.plot(x,y,"xg")
					if d <= minD:
						minD = d
						minX = x
						minY = y
				except ValueError:
					continue
		#plt.plot([p1.x, p2.x], [p1.y, p2.y], "-r")
		#plt.plot([p1.x, minX], [p1.y, minY], clr)
		#plt.pause(0.5)
		return minX,minY


	def isLeftOfLine(self,p1, p2, v):
		return (p2.x - p1.x)*(v.y - p1.y) > (p2.y - p1.y)*(v.x - p1.x)



	def intersect(self,a,b,c,d):

		t_num = (a.x-c.x)*(c.y-d.y) - (a.y-c.y)*(c.x-d.x)
		u_num = (a.x-b.x)*(a.y-c.y) - (a.y-b.y)*(a.x-c.x)
		denom = (a.x-b.x)*(c.y-d.y) - (a.y-b.y)*(c.x-d.x)

		if denom == 0:
			raise ValueError
		t = t_num / denom
		u = - u_num / denom


		if (-0.0000 <= t <= 1.0000) and (-0.0000 <= u <= 1.0000):
			x = c.x + u*(d.x-c.x)
			y = c.y + u*(d.y-c.y)

			x2 = a.x + t*(b.x-a.x)
			y2 = a.y + t*(b.y-a.y)

			return x,y
		raise ValueError



def genRandomRectangle():
    #width = 1#random.randrange(5,50)
    width = random.randrange(5,50)
    #height = 1#random.randrange(5,50)
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
        tx = float(x[i]*math.cos(theta) - y[i]*math.sin(theta))
        ty = float(x[i]*math.sin(theta) + y[i]*math.cos(theta))
        x[i] = tx
        y[i] = ty

    return ObstaclePolygon(x,y)

def main():
	print(__file__ + " start!!")
	for i in range(100000):
		print(i)
		plt.cla()
		# start and goal position
		x, y = random.randrange(-25,25), random.randrange(-25,25)  # [m]
		h = (2*random.random()-1)*math.pi 

		#x = y = 5.0
		#h = 180/180.0*math.pi

		cnt = 15
		obstacles=[]
		for i in range(cnt):
			obstacles.append(genRandomRectangle())
			#print(obstacles[-1].x_list, obstacles[-1].y_list,)
		obstacles.append(ObstaclePolygon([150,-150,-150,150],[150,150,-150,-150]))

		plt.plot(x, y, "or")
		for ob in obstacles:
			ob.plot()
		plt.axis("equal")


		fov = FieldOfView( [x,y,h], 40/180.0*math.pi, obstacles)
		poly = fov.getFoVPolygon(100)
		poly.plot("r")
		plt.pause(0.001)



if __name__ == '__main__':
    main()
