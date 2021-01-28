import pybullet as p
import time
import pybullet_data
import sys
import numpy as np

def render_in_pybullet(step_output):
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)
    planeId = p.loadURDF("plane.urdf")
    
    object_count = len(step_output.object_list)

    p.resetDebugVisualizerCamera(step_output.camera_height * 2, 0, -42.5, [0,0,0])
    print(len(step_output.object_list))
    for obj in step_output.object_list:
        boxId = createObjectShape(obj)
        if boxId == -1:
            print("err")

    # startPos = [0,0,1]
    # startOrientation = p.getQuaternionFromEuler([0,0,0])
    # boxId = p.loadURDF("simple_box.urdf",startPos, startOrientation)
    #set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
    for i in range (10000):
        p.stepSimulation()
        time.sleep(1./240.)
    cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    print(cubePos,cubeOrn)
    p.disconnect()

def getDims(obj):
    dims = obj.dimensions
    print(dims)
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
    
    shift = list(obj.rotation.values())
    start_orientation = p.getQuaternionFromEuler(shift)
    start_position = list(obj.position.values())
    start_position = [start_position[0], start_position[2], start_position[1]]
    # set color
    rgba_color = getColor(obj.color)
    print(rgba_color)

    visualShapeId = ''
    collisionShapeId = ''
    # create visual and colision shapes
    if obj.shape == "cube":
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName="cube.obj", rgbaColor=rgba_color, specularColor=[0.4,.4,0], visualFramePosition=shift, meshScale=meshScale)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="cube.obj", collisionFramePosition=shift,meshScale=meshScale)

    # if obj.shape == "cylinder"

    # return body
    return p.createMultiBody(baseMass=obj.mass, baseOrientation=start_orientation, baseInertialFramePosition=[0, 0, 0], baseCollisionShapeIndex=collisionShapeId, baseVisualShapeIndex=visualShapeId, basePosition=start_position, useMaximalCoordinates=True)
    
