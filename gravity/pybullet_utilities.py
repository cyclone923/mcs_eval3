import pybullet as p
import time
import pybullet_data
import sys

def render_in_pybullet(step_output):
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)
    planeId = p.loadURDF("plane.urdf")
    
    object_count = len(step_output.object_list)
    print(step_output.camera_aspect_ratio)
    print(step_output.camera_clipping_planes)
    print(step_output.camera_field_of_view)
    print(step_output.camera_height)

    p.resetDebugVisualizerCamera(step_output.camera_height * 2, 0, -42.5, [0,0,0])
    print(len(step_output.object_list))
    for obj in step_output.object_list:
        start_pos = list(obj.position.values())
        rotation = list(obj.rotation.values())
        start_orientation = p.getQuaternionFromEuler(rotation)
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
    min_x = sys.maxsize
    min_y = sys.maxsize
    min_z = sys.maxsize
    max_x = 0
    max_y = 0
    max_z = 0
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

def createObjectShape(obj):
    meshScale = getDims(obj)
    print(meshScale)
    
    shift = [0,0,0]
    # set color
    rgba_color = list(obj.color.values()).append(1)
    
    # create visual and colision shapes
    visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName="cube.obj", rgbaColor=rgba_color, specularColor=[0.4,.4,0], visualFramePosition=shift, meshScale=meshScale)
    collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="cube.obj", collisionFramePosition=shift,meshScale=meshScale)

    # return body
    return p.createMultiBody(baseMass=obj.mass,baseInertialFramePosition=[0,0,0],baseCollisionShapeIndex=collisionShapeId, baseVisualShapeIndex = visualShapeId, basePosition = list(obj.position.values()), useMaximalCoordinates=True)
    
