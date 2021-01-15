import json
import os
import sys

import pymunk
from pymunk.vec2d import Vec2d
import pymunk.matplotlib_util

import matplotlib as mpl
import matplotlib.pyplot as plt
import pymunk.matplotlib_util as pmu

if __name__ == "__main__":
    scene_number = "{0:02d}".format(int(sys.argv[1]))

    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(xlim=(-20, 20), ylim=(-20, 20))
    # ax.set_aspect("equal")

    with open(os.path.abspath("../gravity_scenes/gravity_support_ex_{}.json".format(scene_number))) as scene_file:
        scene_data = json.load(scene_file)
        # print(type(scene_data))
        # print(scene_data)
        objects = scene_data["objects"]
        
        # initialize space
        s = pymunk.Space()

        # initialize each object as a body and a shape
        for obj in objects:
            print(obj)
            # print(obj["shows"][0]["position"]['x'])
            # if obj[""]
            b = pymunk.Body()
            # b.position = (0, 0)
            b.position = (obj["shows"][0]["position"]['x'], obj["shows"][0]["position"]['y'])
            points = []

            if obj["type"] == "cube" or obj["type"] == "cylinder": #draw a rectangle to scale
                x_left = -obj["shows"][0]["scale"]['x']
                x_right = obj["shows"][0]["scale"]['x']
                
                y_bottom = -obj["shows"][0]["scale"]['y']
                y_top = obj["shows"][0]["scale"]['y']
                
                points = [(x_left, y_bottom), (x_left, y_top),
                          (x_right, y_top), (x_right, y_bottom)]
                
            
            elif "frustum" in obj["type"]: # draw 2D frustrum
                x_left = -obj["shows"][0]["scale"]['x']
                x_right = obj["shows"][0]["scale"]['x']
                
                y_bottom = -obj["shows"][0]["scale"]['y']
                y_top = obj["shows"][0]["scale"]['y']

                points = [(x_left, y_bottom), (x_left / 2, y_top),
                          (x_right / 2, y_top), (x_right, y_bottom)]
            
            print(points)
            p = pymunk.Poly(b, points)
            p.mass = obj["mass"]
            s.add(b, p)
            print(obj["id"])
            # s.step(0.01)

        o = pymunk.matplotlib_util.DrawOptions(ax)
        o.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
        o.flags |= pymunk.SpaceDebugDrawOptions.DRAW_COLLISION_POINTS
        s.debug_draw(o)
        # while True: 
        steps = 100
        for x in range(steps):
            s.step(0.1 / steps)
        plt.show()
