import pybullet as p
import time
import pybullet_data
import sys
import numpy as np
import os

def render_in_pybullet(step_output, level):
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    # physicsClient = p.connect(p.DIRECT)
    # p.setAdditionalSearchPath(os.getcwd()) #optionally
    print(os.getcwd() + "/gravity/pybullet_objects/")
    p.setAdditionalSearchPath(os.getcwd() + "/gravity/pybullet_objects/") #optionally
    p.setGravity(0,0,-10)
    planeId = p.loadURDF("plane.urdf")
    
    p.resetDebugVisualizerCamera(step_output["camera_height"] * 2, 0, -42.5, [0,0,0])
    # print(step_output)
    # quit()
    # get structural objects from output
    struct_obj_dict = {}
    if level == "oracle":
        for obj, val in step_output["structural_object_list"].items():
            boxId = createObjectShape(val)
            if boxId == -1:
                print("trouble building struct object", obj)
            else:
                struct_obj_dict[obj] = {
                    "boxID": boxId,
                    "pos": [],
                    "orn": []
                }
    
    # get objects from output
    obj_dict = {}
    for obj, val in step_output["object_list"].items():
        boxId = createObjectShape(val)
        if boxId == -1:
            print("trouble building struct object", obj)
        else:
            obj_dict[obj] = {
                "boxID": boxId,
                "pos": [],
                "orn": []
            }

    # startPos = [0,0,1]
    # startOrientation = p.getQuaternionFromEuler([0,0,0])
    # boxId = p.loadURDF("simple_box.urdf",startPos, startOrientation)
    #set the center of mass frame (loadURDF sets base link frame) startPos/Orn
    # p.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
    for i in range (750):
        p.stepSimulation()
        time.sleep(1./360.)

        # keep track of obj position
        for obj in obj_dict:
            cubePos, cubeOrn = p.getBasePositionAndOrientation(obj_dict[obj]["boxID"])
            obj_dict[obj]["pos"].append(cubePos)
            obj_dict[obj]["orn"].append(cubeOrn)
            
        # if level == "oracle":
        #     for obj in struct_obj_dict:
        #         cubePos, cubeOrn = p.getBasePositionAndOrientation(struct_obj_dict[obj]["boxID"])
        #         struct_obj_dict[obj]["pos"].append(cubePos)
        #         obj_dict[obj]["orn"].append(cubeOrn)

    p.disconnect()

    if level == "oracle":
        return obj_dict
    else:
        return obj_dict

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

    return [max_x - min_x, max_z - min_z, max_y - min_y]

def getColor(color_vals):
    colors = list(color_vals.values())
    colors = np.divide(colors, 255)
    colors = list(colors)
    colors.append(1)
    return colors

def createObjectShape(obj):
    meshScale = getDims(obj)
    
    shift = list(obj["rotation"].values())
    shift = [shift[0], shift[2], shift[1]]
    start_orientation = p.getQuaternionFromEuler(shift)
    start_position = list(obj["position"].values())
    start_position = [start_position[0], start_position[2], start_position[1]]
    # set color
    rgba_color = getColor(obj["color"])

    visualShapeId = ''
    collisionShapeId = ''

    # TODO
    # if a structural object was created - not needed for now??
    if obj["shape"] == "structural":
        # if "wall" in obj.uuid or "floor" in obj.uuid:
        #     return -1 # say we can't build this. It's not really relevant at this point
            
        #     # TODO
        #     # render the floors and walls without causing everything to fly up
        # else:
        #     # render the cylinder as a cube for now
        #     visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName="cube.obj", rgbaColor=rgba_color, specularColor=[0.4,.4,0], visualFramePosition=shift, meshScale=meshScale)
        #     collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="cube.obj", collisionFramePosition=shift,meshScale=meshScale)
        return -1

    # create visual and colision shapes
    print("obj shape", obj["shape"])
    if obj["shape"] == "cube":
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName="cube.obj", rgbaColor=rgba_color, specularColor=[0.4,.4,0], visualFramePosition=shift, meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="cube.obj", collisionFramePosition=shift,meshScale=meshScale)
    elif obj["shape"] == "square frustum":
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName="square_frustum.obj", rgbaColor=rgba_color, specularColor=[0.4,.4,0], visualFramePosition=shift, meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="square_frustum.obj", collisionFramePosition=shift, meshScale=meshScale)
    elif obj["shape"] == "circle frustum":
        print(start_orientation)
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName="circle_frustum.obj", rgbaColor=rgba_color, specularColor=[0.4,.4,0], visualFramePosition=shift, meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="circle_frustum.obj", collisionFramePosition=shift,meshScale=meshScale)
    elif obj["shape"] == "cylinder":
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName="cylinder.obj", rgbaColor=rgba_color, specularColor=[0.4,.4,0], visualFramePosition=shift, meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="cylinder.obj", collisionFramePosition=shift,meshScale=meshScale)
    else:
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName="cube.obj", rgbaColor=rgba_color, specularColor=[0.4,.4,0], visualFramePosition=shift, meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="cube.obj", collisionFramePosition=shift,meshScale=meshScale)
        
    # return body
    # print(visualShapeId, collisionShapeId)
    print(start_orientation)
    return p.createMultiBody(baseMass=obj["mass"], baseOrientation=start_orientation, baseInertialFramePosition=[0, 0, 0], baseCollisionShapeIndex=collisionShapeId, baseVisualShapeIndex=visualShapeId, basePosition=start_position, useMaximalCoordinates=True)
    
