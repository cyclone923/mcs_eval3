"""

Visibility Road Map Planner

author: Atsushi Sakai (@Atsushi_twi)

"""

import cProfile, pstats
from io import StringIO

import time
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import random

from descartes import PolygonPatch

import shapely.geometry as sp
from shapely.prepared import prep

from MCS_exploration.navigation.dijkstra_search import DijkstraSearch
import ray
import psutil

from shapely.ops import unary_union
from heapq import heappush, heappop

show_animation = True


def validEdge(p1, p2, poly, robot_radius):
    if math.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) <= 0.01:
             return False

    radiusPolygon = sp.LineString([p1, p2]).buffer(robot_radius)
    return not poly.intersects(radiusPolygon)



class Node(object):
    def __init__(self, x,y,h,g, prev):
        self.x = x
        self.y = y
        self.h = h
        self.g = g
        self.f = h+g
        self.prev = prev

    def __hash__(self):
        return hash( (self.x, self.y) )

    def __eq__(self, other):
        return self.x==other.x and self.y==other.y
        #3return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2) < 0.01

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.h < other.h

    def __str__(self):
        return "Node({:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f})".format(self.x, self.y, self.h, self.g, self.f)

class DiscreteActionPlanner:

    def __init__(self, robot_radius, obstacles, eps=0.1, do_plot=False):
        self.robot_radius = robot_radius*1.1
        self.obstacles = (unary_union(obstacles))
        self.eps = eps
        self.step = 0.1
        self.turn = 10
        self.offsets = [ (math.sin(a)*self.step, math.cos(a)*self.step) for a in [self.turn*x/180*np.pi for x in range(1,360//self.turn)]]

        self.existing_plan = None

        # if not ray.is_initialized():
        #     num_cpus = psutil.cpu_count(logical=False)
        #     ray.init(num_cpus=num_cpus,ignore_reinit_error=True)
        

    def addObstacle(self, obstacle):
        # add obstacle and just rebuild the map
        self.obstacles = (self.obstacles.union(obstacle))
        
    def resetObstacles(self, obstacles=None):
        if obstacles:
            self.obstacles = unary_union(obstacles)
        else:
            self.obstacles = MultiPolygon()

    def planning(self, start_x, start_y, goal_x, goal_y):

        poly = prep(self.obstacles)


        if self.existing_plan and len(self.existing_plan) > 1:

            linePlan = sp.LineString( self.existing_plan ).buffer(self.robot_radius)

            
            if not poly.intersects(linePlan):
                self.existing_plan.pop(0)
                return [p[0] for p in self.existing_plan], [p[1] for p in self.existing_plan]
            else:
                self.existing_plan = None


        #pr = cProfile.Profile()
        #pr.enable()
        
        openSet = [  Node(start_x, start_y, self.heurstic(start_x, start_y, goal_x, goal_y), 0, None) ]
        closedSet = set()
        goal = Node(goal_x, goal_y, 0, 0, None)

        while openSet and openSet[0].h > self.eps :
            curr = heappop(openSet)
            closedSet.add(curr)

            # ATTENTION: THIS SHOULD REALLY BE DONE BUT IT IS VERY SLOW
            # add any successors that arent already in the open/closed set
            for s in filter(lambda x: x not in openSet and x not in closedSet, self.validSuccessors(curr, goal, poly)):
                heappush(openSet, s)
            
            # THIS IS BAD
            #for s in self.validSuccessors(curr, goal, poly):
            #    heappush(openSet, s)
             

        #pr.disable()
        #s = StringIO()
        #sortby = 'cumulative'
        #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #ps.print_stats()
        #print(s.getvalue())

        if openSet and openSet[0].h <= self.eps:
            path = [openSet[0]]
            while path[-1].prev:
                path.append(path[-1].prev)
            path.reverse()
            self.existing_plan = [ (n.x, n.y) for n in path ]

            return [p.x for p in path], [p.y for p in path]

        raise ValueError("Planner failed to find a path")
        
    def validSuccessors(self, loc, goal, poly):
        return [ Node(loc.x+x, loc.y+y, self.heurstic(loc.x+x, loc.y+y, goal.x, goal.y), loc.g+self.step, loc) for x,y in self.offsets if validEdge( (loc.x, loc.y), (loc.x+x, loc.y+y), poly, self.robot_radius) ]
       
    def heurstic(self,loc_x,loc_y, goal_x, goal_y):
        return math.sqrt( (loc_x - goal_x)**2 + (loc_y - goal_y)**2)




class ObstaclePolygon(sp.Polygon):
    def __init__(self, x, y):
        super().__init__(zip(x,y))
        self.x_list = x
        self.y_list = y

    def plot(self, clr="grey"):
        patch1 = PolygonPatch(self, fc=clr, ec="black", alpha=0.2, zorder=1)
        plt.gca().add_patch(patch1)

    def contains_goal(self, goal):
        return self.contains(sp.Point(goal))

    def get_goal_bonding_box_polygon(self):
        return self


