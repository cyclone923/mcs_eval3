import pybullet as p
import time
import pybullet_data
import sys
import numpy as np
import os
import json
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from rich.console import Console

console = Console()

def render_in_pybullet(step_output, velocities=None):
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    # physicsClient = p.connect(p.DIRECT)
    
    p.setAdditionalSearchPath(os.getcwd() + "/gravity/pybullet_objects/") #optionally
    p.setGravity(0,0,-10)
    planeId = p.loadURDF("plane100.urdf")
    
    p.resetDebugVisualizerCamera(3, 0, -42.5, [0,0,0])
    
    # build objects in the object_list:
    obj_dict = {
        "default": {}
    }
    total_objects = 0
    # target / supports
    for obj_id, obj in step_output["object_list"].items():
        
        boxId = createObjectShape(obj)
        print(boxId)
        console.log(boxId)
        if boxId == -1:
            print("error creating obj: {}".format(obj.shape))
        else:
            total_objects += 1
            obj_dict["default"][obj_id] = {
                "id": boxId,
                "pos": [],
                "orn": [],
                "floor_contact": [],
                # a dict where each key is the box_id of another object, the value of that key is a list t entries long with pybullet
                # contact calculation output. contacts[boxID][t] = () means no contact, contacts[boxID][t] != () means contact 
                "object_contacts": {},
                "aab_min": [],
                "aab_max": []
            }

    steps = 0
    # let simulation run
    while steps < 1000:
        p.stepSimulation()
        time.sleep(1./400.)

        for obj_id in velocities:
            if obj_id in obj_dict['default']:
                boxId = obj_dict['default'][obj_id]['id']
                pos, orn = p.getBasePositionAndOrientation(boxId)
                obj_vel = velocities[obj_id]
                p.applyExternalForce(boxId, linkIndex=-1, forceObj=100*obj_vel, posObj=pos, flags=p.WORLD_FRAME)

        at_rest = []
        for i, obj in obj_dict["default"].items():
            # get position and orientation
            cubePos, cubeOrn = p.getBasePositionAndOrientation(obj["id"])

            # get bounding box
            aabb_min, aabb_max = p.getAABB(obj["id"])
            
            # get floor contact of object
            floor_contact = p.getContactPoints(obj["id"], planeId)

            # get contact of other objects
            for j, obj2 in obj_dict["default"].items():
                if i != j:
                    contact = p.getContactPoints(obj["id"], obj2["id"])

                    # on first pass, we need to instantiate the list of contact over time
                    if not obj2["id"] in obj["object_contacts"].keys():
                        obj["object_contacts"][obj2["id"]] = [contact]
                    else:
                        obj["object_contacts"][obj2["id"]].append(contact)

            # if object has not moved or rotated, say it is at rest
            if steps > 1:
                prev_pos = obj["pos"][-1]
                prev_orn = obj["orn"][-1]
                if np.isclose(cubePos, prev_pos, rtol=1e-04, atol=1e-05).all() and np.isclose(cubeOrn, prev_orn).all():
                    at_rest.append(True)
                else:
                    at_rest.append(False)

            # update lists for new point in time
            obj["floor_contact"].append(floor_contact)
            obj["aab_min"].append(aabb_min)
            obj["aab_max"].append(aabb_max)
            obj["pos"].append(cubePos)
            obj["orn"].append(cubeOrn)

            # save updates to obj_dict
            obj_dict["default"][i] = obj

        # all objects are at rest, go ahead and end the simulation early 
        if steps > 1 and all(at_rest):
            break

        steps += 1
    
    p.disconnect()

    return steps, obj_dict    
    

def getDims(obj):
    dims = obj["dimensions"]
    min_x = sys.maxsize
    min_y = sys.maxsize
    min_z = sys.maxsize
    max_x = -1*sys.maxsize
    max_y = -1*sys.maxsize
    max_z = -1*sys.maxsize
    
    for dim in dims:
        if dim['x'] <= min_x:
            min_x = dim['x']
        if dim['x'] >= max_x:
            max_x = dim['x']

        if dim['y'] <= min_y:
            min_y = dim['y']
        if dim['y'] >= max_y:
            max_y = dim['y']

        if dim['z'] <= min_z:
            min_z = dim['z']
        if dim['z'] >= max_z:
            max_z = dim['z']
    # print([max_x, min_x, max_z, min_z, max_y, min_y])
    return [max_x - min_x, max_z - min_z, max_y - min_y]

def getColor(color_vals):
    colors = list(color_vals.values())
    colors = np.divide(colors, 255)
    colors = list(colors)
    colors.append(1)
    return colors

def createObjectShape(obj):
    meshScale = getDims(obj)
    if obj["shape"] != "structural":
        # generate noise on position and orientation
        shift = [0, 0, 0]
        # print(list(obj["rotation"].values()))
        start_orientation = p.getQuaternionFromEuler(shift)
         
        start_position = list(obj["position"].values())
        # pos_noise = np.random.normal(0,0.5,len(start_position))
        # start_position = start_position + pos_noise
        start_position = [start_position[0], start_position[2], start_position[1]]
    else:
        shift = list(obj["rotation"].values())
        shift = [round(shift[0]), round(shift[2]), round(shift[1])]
        start_orientation = p.getQuaternionFromEuler(shift)
        
        # shift = [0, 0, 0]
        start_position = list(obj["position"].values())
        start_position = [start_position[0], start_position[2], start_position[1]]
    # set color
    rgba_color = getColor(obj["color"])

    visualShapeId = ''
    collisionShapeId = ''

    # TESTING - load duck instead of object
    # if obj["shape"] != "structural":
    #     return p.loadURDF("duck_vhacd.urdf", basePosition=start_position, baseOrientation=start_orientation)

    # create visual and colision shapes
    if obj["shape"] == "cube" or obj["shape"] == "structural":
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName="cube.obj", rgbaColor=rgba_color, specularColor=[0.4,.4,0], visualFramePosition=shift, meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="cube.obj", collisionFramePosition=shift,meshScale=meshScale)
    elif obj["shape"] == "square frustum":
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName="square_frustum.obj", rgbaColor=rgba_color, specularColor=[0.4,.4,0], visualFramePosition=shift, meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="square_frustum.obj", collisionFramePosition=shift, meshScale=meshScale)
    elif obj["shape"] == "circle frustum":
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName="circle_frustum.obj", rgbaColor=rgba_color, specularColor=[0.4,.4,0], visualFramePosition=shift, meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="circle_frustum.obj", collisionFramePosition=shift,meshScale=meshScale)
    elif obj["shape"] == "cylinder":
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName="cylinder.obj", rgbaColor=rgba_color, specularColor=[0.4,.4,0], visualFramePosition=shift, meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="cylinder.obj", collisionFramePosition=shift,meshScale=meshScale)
    elif "letter l" in obj["shape"]:
        meshScale = [meshScale[0], meshScale[1], meshScale[2] * 0.75] # hard coded transformations to compensate for unknown wonkiness... needs to be tested
        start_orientation = [start_orientation[0], start_orientation[1], 90, 0.011]
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName="L_joint.obj", rgbaColor=rgba_color, specularColor=[0.4,.4,0], visualFramePosition=shift, meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="L_joint.obj", collisionFramePosition=shift, meshScale=meshScale)
    elif "triangular prism" == obj["shape"]:
        meshScale = [meshScale[1], meshScale[2], meshScale[0]]
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName="triangular prism.obj", rgbaColor=rgba_color, specularColor=[0.4,.4,0], visualFramePosition=shift, meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="triangular prism.obj", collisionFramePosition=shift, meshScale=meshScale)
    else:
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName="cube.obj", rgbaColor=rgba_color, specularColor=[0.4,.4,0], visualFramePosition=shift, meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="cube.obj", collisionFramePosition=shift, meshScale=meshScale)
    # return body
    return p.createMultiBody(baseMass=obj["mass"], baseOrientation=start_orientation, baseInertialFramePosition=[0, 0, 0], baseCollisionShapeIndex=collisionShapeId, baseVisualShapeIndex=visualShapeId, basePosition=start_position)
    
