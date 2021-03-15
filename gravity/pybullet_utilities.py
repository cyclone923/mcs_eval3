import pybullet as p
import time
import pybullet_data
import sys
import numpy as np
import os
import json

def render_in_pybullet(step_output, target, supporting, level):
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    # physicsClient = p.connect(p.DIRECT)

    p.setAdditionalSearchPath(os.getcwd() + "/gravity/pybullet_objects/") #optionally
    p.setGravity(0,0,-10)
    planeId = p.loadURDF("plane100.urdf")
    print(planeId)
    
    p.resetDebugVisualizerCamera(3, 0, -42.5, [0,0,0])
    
    # get objects from output
    obj_dict = {}
    boxId = createObjectShape(step_output["structural_object_list"][supporting])
    print(boxId)
    if boxId == -1:
        print("trouble building supporting object")
    else:
        obj_dict[supporting] = {
        "boxID": boxId,
        "orn": [],
        "pos": []
        }

    # print(json.dumps(step_output["object_list"][target], indent=4))
    # print(json.dumps(step_output["structural_object_list"][supporting], indent=4))
    # quit()
    boxId = createObjectShape(step_output["object_list"][target])
    print(boxId)
    
    if boxId == -1:
        print("trouble building target object")
    else:
        obj_dict[target] = {
            "boxID": boxId,
            "pos": [],
            "orn": [],
            "support_contact": [],
            "floor_contact": [],
            "aabbMin": [],
            "aabbMax": []
        }

    # p.setRealTimeSimulation()
    for i in range(750):
        p.stepSimulation()
        time.sleep(1./360.)
    
        # confirm there aren't any overlaps on target object
        aabb_min, aabb_max = p.getAABB(obj_dict[target]["boxID"])
        overlaps = p.getOverlappingObjects(aabb_min, aabb_max)
        # print("overlaps", overlaps)
        
        # get contact points between target and supporting
        object_contact = p.getContactPoints(obj_dict[target]["boxID"], obj_dict[supporting]["boxID"])
        floor_contact = p.getContactPoints(obj_dict[target]["boxID"], planeId)
        # if contact != ():
        #     print("support and target are making contact")
        #     print(contact)

        # keep track of obj position
        for obj in obj_dict:
            if obj == target:
                obj_dict[obj]["support_contact"].append(object_contact)
                obj_dict[obj]["floor_contact"].append(floor_contact)
                obj_dict[obj]["aabbMin"].append(aabb_min)
                obj_dict[obj]["aabbMax"].append(aabb_max)
            cubePos, cubeOrn = p.getBasePositionAndOrientation(obj_dict[obj]["boxID"])
            obj_dict[obj]["pos"].append(cubePos)
            obj_dict[obj]["orn"].append(cubeOrn)

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
    print(meshScale)
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
    print(start_position)
    # set color
    rgba_color = getColor(obj["color"])

    visualShapeId = ''
    collisionShapeId = ''

    # TESTING - load duck instead of object
    # if obj["shape"] != "structural":
    #     return p.loadURDF("duck_vhacd.urdf", basePosition=start_position, baseOrientation=start_orientation)

    # create visual and colision shapes
    print("obj shape", obj["shape"])
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
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName="l_joint.obj", rgbaColor=rgba_color, specularColor=[0.4,.4,0], visualFramePosition=shift, meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="l_joint.obj", collisionFramePosition=shift, meshScale=meshScale)
    elif "triangular prism" == obj["shape"]:
        meshScale = [meshScale[1], meshScale[2], meshScale[0]]
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName="triangular prism.obj", rgbaColor=rgba_color, specularColor=[0.4,.4,0], visualFramePosition=shift, meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="triangular prism.obj", collisionFramePosition=shift, meshScale=meshScale)
    else:
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName="cube.obj", rgbaColor=rgba_color, specularColor=[0.4,.4,0], visualFramePosition=shift, meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="cube.obj", collisionFramePosition=shift, meshScale=meshScale)
    # return body
    return p.createMultiBody(baseMass=obj["mass"], baseOrientation=start_orientation, baseInertialFramePosition=[0, 0, 0], baseCollisionShapeIndex=collisionShapeId, baseVisualShapeIndex=visualShapeId, basePosition=start_position)
    
