import math
import shapely.geometry as sp
from shapely.prepared import prep
from shapely.ops import unary_union, nearest_points
from heapq import heappush, heappop
from shapely import speedups
if speedups.available:
    speedups.enable()






class Node(object):
    def __init__(self, x,y,h,g, prev):
        self.x = x
        self.y = y
        self.h = h
        self.g = g
        self.f = (1.01)*h+g
        self.prev = prev

    def __hash__(self):
        #hash up to 4 decimals
        return hash( (int(round(self.x*20)), int(round(self.y*20)) ) )

    def __eq__(self, other):
        #fuzzy notion of equality based on hash
        return self.__hash__() == other.__hash__()
        
    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.f < other.f

    def __str__(self):
        return "Node({:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f})".format(self.x, self.y, self.h, self.g, self.f)

class DiscreteActionPlanner:

    def __init__(self, robot_radius, obstacles, eps=0.2, step=0.1, turn=10):
        self.robot_radius = robot_radius
        self.obstacles = (unary_union([o.boundary for o in obstacles]))
        self.eps = eps
        self.step = step
        self.turn = turn
        self.offsets = [ (math.sin(a)*self.step, math.cos(a)*self.step) for a in [self.turn*x/180.0*math.pi for x in range(0,360//self.turn)]]
        self.existing_plan = []
        

    def addObstacle(self, obstacle):
        self.obstacles = (self.obstacles.union(obstacle.boundary))
        
    def resetObstacles(self, obstacles=None):
        if obstacles:
            self.obstacles = unary_union([o.boundary for o in obstacles])
        else:
            self.obstacles = MultiPolygon()

    def planning(self, start_x, start_y, goal_x, goal_y, returnNearest=False, max_exp = 5000):
        #get optimized polygon to make comparisons quicker
        poly = prep(self.obstacles)

        #use a max heap for the open set ordered by heuristic
        openList = [  Node(start_x, start_y, self.heurstic(start_x, start_y, goal_x, goal_y), 0, None) ]

        #use hash-based set operations to detect duplication in O(1)
        openSet = set(openList)
        closedSet = set()

        goal = Node(goal_x, goal_y, 0, 0, None)

        i = 0

        nearest = Node(0,0,math.inf,0,None)

        while openList and openList[0].h > self.eps and i < max_exp :
            curr = heappop(openList)
            openSet.remove(curr)
            closedSet.add(curr)

            if curr.h < nearest.h:
                nearest = curr

            i += 1
            

            # add any successors that arent already in the open/closed set
            for s in filter( lambda s: self.validEdge( (curr.x, curr.y), (s.x, s.y), poly, self.robot_radius),  filter(lambda x: x not in openSet and x not in closedSet, self.successors(curr, goal))): #filter(lambda x: x not in openSet and x not in closedSet, self.validSuccessors(curr, goal, poly)):
                heappush(openList, s)
                openSet.add(s)
            #print(len(openSet), i, i*36)
            
        if openList and openList[0].h <= self.eps:
            path = [openList[0]]
        else:
            if returnNearest:
                path = [nearest]
            else:
                return [],[]

        while path[-1].prev:
            path.append(path[-1].prev)
        path.reverse()
        
        return [p.x for p in path[1:]], [p.y for p in path[1:]]
    
    def validEdge(self, p1, p2, poly, robot_radius):
        radiusPolygon = sp.LineString([p1, p2]).buffer(robot_radius)
        return not poly.intersects(radiusPolygon)

    def successors(self, loc, goal):
        return [ Node(loc.x+x, loc.y+y, self.heurstic(loc.x+x, loc.y+y, goal.x, goal.y), loc.g+self.step, loc) for x,y in self.offsets]
    

    def validSuccessors(self, loc, goal, poly):
        return [ Node(loc.x+x, loc.y+y, self.heurstic(loc.x+x, loc.y+y, goal.x, goal.y), loc.g+self.step, loc) for x,y in self.offsets if self.validEdge( (loc.x, loc.y), (loc.x+x, loc.y+y), poly, self.robot_radius) ]
       
    def heurstic(self,loc_x,loc_y, goal_x, goal_y):
        return math.sqrt( (loc_x - goal_x)**2 + (loc_y - goal_y)**2)

    def isStuck(self, pos):
        poly = prep(self.obstacles)
        return not any([ self.validEdge( (pos[0], pos[1]), (pos[0]+x, pos[1]+y), poly, self.robot_radius) for x,y in self.offsets ])

    def validPlan(self, path, cur):
        if len(path) == 0:
            return False
        tp = path.copy()

        tp.insert(0, cur)
        planPoly = sp.LineString(tp).buffer(self.robot_radius)

        poly = prep(self.obstacles)

        return not poly.intersects(planPoly)

    def distToNearest(self, x, y):
        cur, nearest = nearest_points(sp.Point( (x,y) ), self.obstacles)
        cur_x, cur_y = list(cur.coords)[0][0], list(cur.coords)[0][1]
        near_x, near_y = list(nearest.coords)[0][0], list(nearest.coords)[0][1]

        return math.sqrt( (near_x - cur_x)**2 + (near_y - cur_y)**2)

    def getUnstuckPath(self, x, y, steps=5):
        cur, nearest = nearest_points(sp.Point( (x,y) ), self.obstacles)
        cur_x, cur_y = list(cur.coords)[0][0], list(cur.coords)[0][1]
        near_x, near_y = list(nearest.coords)[0][0], list(nearest.coords)[0][1]

        x = cur_x - near_x
        y = cur_y - near_y
        n = math.sqrt( x**2 + y**2)
        x = x/(n+0.000000001)
        y = y/(n+0.000000001)

        x_path = [cur_x + self.step*x*i for i in range(1,steps)]
        y_path = [cur_y + self.step*y*i for i in range(1,steps)]

        return x_path, y_path

    