import constants
import networkx as nx
import math
import time 
from shapely.geometry import Point, Polygon  
from navigation.fov import FieldOfView
import  numpy as np
import matplotlib.pyplot as plt
import quaternion

testing = 0

#max_abs = 5/constants.AGENT_STEP_SIZE
if testing != 1 :
    #AGENT_STEP_SIZE = 0
    max_abs = 5/constants.AGENT_STEP_SIZE
    move_step_size = 6
else :
    move_step_size = 1
    max_abs =1/constants.AGENT_STEP_SIZE
x_range = max_abs
z_range = max_abs

xMin = -x_range
xMax = x_range
zMin = -z_range
zMax = z_range

number_direction = 4
#q = [] 


class graph_2d():
    def __init__(self):#,xMin,xMax,yMin,yMax):
        self.graph = nx.DiGraph() 
        '''
        xMin = int(global xMin)
        yMin = int(global yMin)
        xMax = int(global xMax)
        yMax = int(global yMax)
        
        xMin_local = int(xMin)
        zMin_local = int(zMin)
        xMax_local = int(xMax)
        zMax_local = int(zMax)
        '''
        self.xMin = int(xMin)
        self.yMin = int(zMin)
        self.xMax = int(xMax)
        self.yMax = int(zMax)
        not_seen_points = []
        #for x in range (xMin_local, xMax_local):
        #    for y in range (xMin_local, yMax_local):
       
        for xx in range (self.xMin, self.xMax):
            for yy in range (self.yMin, self.yMax):
                self.graph.add_node((xx,yy), visited=False, seen=False, contains_object = False)
                curr_weight = 1
                if yy != self.yMax:
                    self.graph.add_edge((xx,yy),(xx, yy + 1), weight=curr_weight)
                elif xx != self.xMax:
                    self.graph.add_edge((xx,yy),(xx + 1, yy), weight=curr_weight)
                elif yy != self.yMin:
                    self.graph.add_edge((xx,yy),(xx, yy - 1), weight=curr_weight)
                elif xx != self.xMin:
                    self.graph.add_edge((xx,yy),(xx - 1, yy), weight=curr_weight)
                #not_seen_points.append((x,y))    

    def reset(self):
        for xx in range (self.xMin, self.xMax):
            for yy in range (self.yMin, self.yMax):
                self.graph.nodes[(xx,yy)]['visited']=False
                self.graph.nodes[(xx,yy)]['seem']=False
                self.graph.nodes[(xx,yy)]['contains_object']=False

    '''
    function to get all the unexplored points in the grid
    '''
    def get_unseen(self):#,xMin,xMax,yMin,yMax):

        xMin_local = int(xMin)
        yMin_local = int(zMin)
        xMax_local = int(xMax)
        yMax_local = int(zMax)
        '''
        xMin = int(global xMin)
        yMin = int(global yMin)
        xMax = int(global xMax)
        yMax = int(global yMax)
        '''
        not_seen_points = []
        for x in range (xMin_local, xMax_local):
            for y in range (yMin_local, yMax_local):
                node = self.graph.nodes[(x,y)]
                #if not (node['visited']) and not(node['seen']) and not(node['contains_object']):
                if not (node['visited'] or node['seen'] or node['contains_object']):
                    not_seen_points.append((x,y))

        return not_seen_points

    '''
    Function to get all the visible points from a certain point in the 2D grid
    '''
    def points_visible_from_position(self,x,y,camera_field_of_view,radius,obstacles):#,visibility_graph):
        number_visible_points = 0
        #number_directions = 4
        number_directions = 8
        for direction in range (0,number_directions):#,number_directions*2):
        #for direction in range (0,number_directions):
            #print (direction)
            #number_visible_points += len(get_visible_points(x,y,direction/2,camera_field_of_view, radius))
            number_visible_points += len(self.get_visible_points(x,y,direction*45,camera_field_of_view, radius,obstacles))#,visibility_graph))

        return number_visible_points

    '''
    Function to update any new explored points in the grid
    '''
    def update_seen(self, x, z, rotation,radius, camera_field_of_view,obstacles):
        #camera_field_of_view = event.camera_field_of_view
        #radius = event.camera_clipping_planes[1]
        visible_points = self.get_visible_points(x,z,rotation,camera_field_of_view,radius,obstacles)#,g )

        for elem in visible_points :
            self.graph.nodes[elem]['seen']= True
        #pass


    def get_visible_points(self,x,y,rotation,camera_field_of_view,radius,scene_obstacles_dict):#,visibility_graph):
        step_size = constants.AGENT_STEP_SIZE
        graph_x = x/constants.AGENT_STEP_SIZE
        graph_z = y/constants.AGENT_STEP_SIZE

        rotation = (rotation+90 ) % 360

        rotation_radians = rotation / 360 * (2 * math.pi)

        fov = FieldOfView([x,y, 0], camera_field_of_view / 180.0 * math.pi, scene_obstacles_dict.values())
        fov.agentH = rotation_radians
        fov_poly = fov.getFoVPolygon(12)

        '''
        plt.cla()
        plt.xlim((-7, 7))
        plt.ylim((-7, 7))
        plt.gca().set_xlim((-7, 7))
        plt.gca().set_ylim((-7, 7))

        plt.plot(x,y, "or")
        #plt.plot(gx, gy, "ob")
        fov_poly.plot("-r")

        for obstacle in scene_obstacles_dict.values():
            obstacle.plot("-g")

        plt.axis("equal")
        plt.pause(0.1)
        '''
        #print ("poly X", fov_poly.x_list)
        #print ("poly Y", fov_poly.y_list)
        pt_1_angle = math.degrees(math.atan((fov_poly.x_list[1]/step_size-graph_x)/(fov_poly.y_list[1]/step_size-graph_z)))
        pt_2_angle = math.degrees(math.atan((fov_poly.x_list[-2]/step_size-graph_x)/(fov_poly.y_list[-2]/step_size-graph_z)))

        lower_angle = min(pt_1_angle,pt_2_angle)
        higher_angle = max(pt_1_angle,pt_2_angle)
        loop_x_min, loop_x_max = int(min(fov_poly.x_list)/constants.AGENT_STEP_SIZE), int(max(fov_poly.x_list)/constants.AGENT_STEP_SIZE)
        loop_z_min, loop_z_max = int(min(fov_poly.y_list)/constants.AGENT_STEP_SIZE), int(max(fov_poly.y_list)/constants.AGENT_STEP_SIZE)

        number_ray_casted = 0
        visible_points = []
        possible_points_x = []
        possible_points_y = []
        dict_values =scene_obstacles_dict.values()
        start_time = time.time()
        for i in range(loop_x_min, loop_x_max+1 ):
            for j in range(loop_z_min, loop_z_max+1):
                #goal_point = Point(i,j)
                #if goal_point.within(poly):
                if j==graph_z or j >= zMax or i >= xMax or i<xMin or j < zMin:
                    continue
                current_pt_angle =  math.degrees(math.atan((i-graph_x)/(j-graph_z)))
                #pt_2_angle =  math.degrees(math.atan((p2_x-x)/(p2_z-z)))
                #if visibility_graph[elem]['seen'] or
                node = self.graph.nodes[(i,j)]#g.nodes[(x, y)]
                # if not (node['visited']) and not(node['seen']) and not(node['contains_object']):
                if not (node['visited'] or node['seen'] or node['contains_object']):
                    if current_pt_angle >= lower_angle and current_pt_angle <= higher_angle :
                        number_ray_casted += 1
                        if castRay(graph_x,graph_z,i,j,dict_values):
                           visible_points.append((i, j))
                        #if ray_tracing_mult():
                        #possible_points_x.append(i)
                        #possible_points_y.append(j)
                        #if ()
                #else :
                #    print ("already seen")
        #if (len(possible_points_x) >= 0):
        #    fov_ray_poly = [[x,y] for x,y in zip(fov_poly.x_list,fov_poly.y_list)]
        #    visible_points = ray_tracing_mult(possible_points_x,possible_points_y,fov_ray_poly)


        end_time = time.time()
        time_taken = end_time- start_time
        #print ("time taken for processing = " , end_time- start_time)
        return visible_points

        #poly.plot()


def ray_tracing_numpy(x,y,poly):
    n = len(poly)
    inside = np.zeros(len(x),np.bool_)
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        idx = np.nonzero((y > min(p1y,p2y)) & (y <= max(p1y,p2y)) & (x <= max(p1x,p2x)))[0]
        if p1y != p2y:
            xints = (y[idx]-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
        if p1x == p2x:
            inside[idx] = ~inside[idx]
        else:
            idxx = idx[x[idx] <= xints]
            inside[idxx] = ~inside[idxx]

        p1x,p1y = p2x,p2y
    return inside

def ray_tracing_mult(x,y,poly):
    return [ray_tracing_numpy(xi, yi, poly) for xi,yi in zip(x,y)]

def castRay(graph_x, graph_y, check_x, check_y,obstacle,clr="-g"):
    start_time = time.time()
    #p1 = Geometry.Point(float(self.agentX), float(self.agentY))
    #p2 = Geometry.Point(p1.x + maxLen * np.cos(angle), p1.y + maxLen * np.sin(angle))
    p1 = Point(graph_x,graph_y)
    p2 = Point(check_x, check_y)

    #minD = math.inf
    minD = math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
    minX = p2.x
    minY = p2.y
    for obs in obstacle:
        for i in range(len(obs.x_list) - 1):
            o1 = Point(obs.x_list[i]/constants.AGENT_STEP_SIZE, obs.y_list[i]/constants.AGENT_STEP_SIZE)
            o2 = Point(obs.x_list[i + 1]/constants.AGENT_STEP_SIZE, obs.y_list[i + 1]/constants.AGENT_STEP_SIZE)

            try:
                x, y = intersect(p1, p2, o1, o2)
                d = math.sqrt((x - p1.x) ** 2 + (y - p1.y) ** 2)
                # plt.plot(x,y,"xg")
                if d < minD:
                    end_time = time.time()
                    #time_taken = end_time - start_time
                    #print ("time taken for single ray casting" , end_time- start_time)
                    return False
            except ValueError:
                continue
    end_time = time.time()
    #time_taken = end_time - start_time
    #print ("time taken for single ray casting" , end_time- start_time)
    return True


def intersect(a, b, c, d):
    t_num = (a.x - c.x) * (c.y - d.y) - (a.y - c.y) * (c.x - d.x)
    u_num = (a.x - b.x) * (a.y - c.y) - (a.y - b.y) * (a.x - c.x)
    denom = (a.x - b.x) * (c.y - d.y) - (a.y - b.y) * (c.x - d.x)

    if denom == 0:
        raise ValueError
    t = t_num / denom
    u = - u_num / denom

    if (-0.0000 <= t <= 1.0000) and (-0.0000 <= u <= 1.0000):
        x = c.x + u * (d.x - c.x)
        y = c.y + u * (d.y - c.y)

        x2 = a.x + t * (b.x - a.x)
        y2 = a.y + t * (b.y - a.y)

        return x, y
    raise ValueError

def check_validity(x,z,q,x_z_range):
    xMin,xMax,zMin,zMax = x_z_range
    if x < xMin :
        return False
    elif x >= xMax :
        return False
    elif z < zMin :
        return False
    elif z >= zMax:
        return False
    if ((x,z)) in q:
        return False
    return True

def flood_fill(x,y, check_validity,x_z_range):
#def flood_fill(x,y, check_validity):
    #//here check_validity is a function that given coordinates of the point tells you whether
    #//the point should be colored or not
    #Queue q
    curr_q = []
    q = []
    q.append((x,y))
    curr_q.append((x,y))
    i = 1
    x_z_range = [i/constants.AGENT_STEP_SIZE for i in x_z_range]
    while (len(curr_q) != 0):
        #(x1,y1) = curr_q.pop()
        (x1,y1) = curr_q[0]
        curr_q = curr_q[1:]
        #print (x1,y1)
        #color(x1,y1)

        if (check_validity(x1+move_step_size,y1,q,x_z_range)):
        #if (check_validity(x1+move_step_size,y1,q)):
            q.append((x1+move_step_size,y1))
            curr_q.append((x1+move_step_size,y1))
            i += 1
        if (check_validity(x1,y1+move_step_size,q,x_z_range)):
        #if (check_validity(x1,y1+move_step_size,q)):
            q.append((x1,y1+move_step_size))
            curr_q.append((x1,y1+move_step_size))
            i += 1
        if (check_validity(x1-move_step_size,y1,q,x_z_range)):
        #if (check_validity(x1-move_step_size,y1,q)):
            q.append((x1-move_step_size,y1))
            curr_q.append((x1-move_step_size,y1))
            i += 1
        if (check_validity(x1,y1-move_step_size,q,x_z_range)):
        #if (check_validity(x1,y1-move_step_size,q)):
            q.append((x1,y1-move_step_size))
            curr_q.append((x1,y1-move_step_size))
            i += 1
        #print ("i = ", i, x1,y1, len(q))
        #if i > 35 :
        #    break
    return q


def explore_initial_point(x,y, agent,obstacles):
    #directions = 8
    #event = agent.game_state.event
    #camera_field_of_view = agent.game_state.event.camera_field_of_view
    explore_point(x,y,agent,obstacles)
    #if agent.game_state.goals_found == True:
    #    return
    return
    directions = 30 // 10
    action = {'action': 'LookDown'}
    for _ in range(directions):
        agent.game_state.step(action)
    explore_point(x,y,agent,obstacles)
    action = {'action': 'LookUp'}
    for _ in range(directions):
        agent.game_state.step(action)


def explore_point(x, y, agent, obstacles):
    directions = 360 // 10
    camera_field_of_view = agent.game_state.event.camera_field_of_view
    #action = {'action':'RotateLook', 'horizon':1}
    #agent.game_state.step(action)
    #action = {'action':'RotateLeft'}
    action = {'action':'RotateRight'}
    for _ in range (directions):
        agent.game_state.step(action)
        rotation = agent.game_state.rotation
        start_time = time.time()
        update_seen(x , y ,agent.game_state,rotation,camera_field_of_view, agent.nav.scene_obstacles_dict.values() )
        #print ("time taken to update seen",time.time()-start_time)
        #if agent.game_state.goals_found == True :
        #    return
    #action = {'action':'RotateLook', 'horizon':-1}
    #agent.game_state.step(action)


def update_seen(x,y,game_state,rotation,camera_field_of_view,obstacles):
    #rotation = (rotation - 45)%360
    #print (rotation)
    rotation_rad = rotation / 180 * math.pi
    #start_time = time.time()
    fov = FieldOfView([x, y, rotation_rad], camera_field_of_view / 180.0 * math.pi, obstacles)
    poly = fov.getFoVPolygon(17)
    #print ("time taken to creat new fov" ,time.time()-start_time)

    try:
        view = Polygon(zip(poly.x_list, poly.y_list)).buffer(0) #the buffer helps eliminate intersections
        game_state.world_poly = game_state.world_poly.union(view)
    except:
        return
    world = game_state.world_poly

    #obstacles_polygons = [ObstaclePolygon(obstacle) for obstacle in obstacles ]

    show_animation = False
    if show_animation:
        plt.cla()
        plt.plot(x, y, "or")
        # plt.plot(gx, gy, "ob")
        # poly.plot("-r")

        if world.geom_type == 'MultiPolygon':
            for i in range(len(world)):
                pts = world[i].exterior.coords
                plt.fill([p[0] for p in pts], [p[1] for p in pts], "-b")
        else:
            pts = world.exterior.coords
            plt.fill([p[0] for p in pts], [p[1] for p in pts], "-b")

        #for i in range(len(obstacles)):
        #    obstacles[i].plot("-k")
        #for obstacle in obstacles:
        #    obstacle.plot("-g")

        plt.axis("equal")
        plt.pause(1)


def get_point_all_new_coverage(x,y,game_state,rotation,obstacles):
    return get_point_new_coverage(x,y,game_state,rotation,360,obstacles)

def get_point_new_coverage(x,y,game_state, rotation,camera_field_of_view,obstacles):
    #rotation = (rotation - 45)%360
    rotation_rad = rotation / 180 * math.pi
    fov_checker = FieldOfView([x,y,rotation_rad],camera_field_of_view/180.0*math.pi, obstacles)

    checkPoly = fov_checker.getFoVPolygon(17)

    intersection_free_points = [[],[]]

    for x,y in zip(checkPoly.x_list,checkPoly.y_list) :
        if len(intersection_free_points[0]) != 0:
            if not ( abs(x - intersection_free_points[0][-1]) < 0.001 and abs(y-intersection_free_points[1][-1]) < 0.001):
                intersection_free_points[0].append(x)
                intersection_free_points[1].append(y)
        else :
            intersection_free_points[0].append(x)
            intersection_free_points[1].append(y)

    #newPoly = Polygon(zip(checkPoly.x_list, checkPoly.y_list))
    try :
        newPoly = Polygon(zip(intersection_free_points[0], intersection_free_points[1]))
        newPoly = newPoly.difference(game_state.world_poly.buffer(0))
    except:
        return 0

    world = game_state.world_poly

    show_animation = False

    if show_animation:
        plt.cla()
        plt.plot(x, y, "or")
        # plt.plot(gx, gy, "ob")
        # poly.plot("-r")

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

        #for i in range(len(obstacles)):
        #    obstacles[i].plot("-k")
        for obstacle in obstacles:
            obstacle.plot("-g")

        plt.axis("equal")
        plt.pause(1)

    return newPoly.area

def get_point_between_points(p1, p2, radius):

    distance_ratio = -0.1
    distance_new_point = 0
    while (distance_new_point < radius):
        x = p1[0] + distance_ratio * (p2[0] - p1[0])
        y = p1[1] + distance_ratio * (p2[1] - p1[1])
        new_pt = [x,y]
        distance_new_point= math.sqrt((new_pt[0]-p1[0])**2+(new_pt[1]-p1[1])**2)
        distance_ratio -= 0.1

    return new_pt

def get_polar_direction(goal, game_state):
    delta_x_3d = goal[0] - game_state.position['x']
    delta_y_3d = goal[1] - game_state.position['y']
    delta_z_3d = goal[2] - game_state.position['z']
    delta_x_unit_3d, delta_y_unit_3d, delta_z_unit_3d = normalize_3d_rotation(delta_x_3d, delta_y_3d, delta_z_3d)
    # the agent is originally looking at z axis
    # rotate_state = degrees of rotating right

    theta = 2 * np.pi * game_state.rotation / 360 
    souce_rotation = quat_from_angle_axis(theta)
    direction_vec_unit = [delta_x_unit_3d, delta_y_unit_3d, delta_z_unit_3d]
    direction_vec_unit_agent = quaternion_rotate_vector(souce_rotation.inverse(), direction_vec_unit)
    dir_3d = np.arctan2(-direction_vec_unit_agent[0], direction_vec_unit_agent[2])

    return dir_3d

def get_head_tilt(goal, game_state):
    distance_to_goal = distance_to_goal_calc(goal, game_state) + 1e-6
    delta_y = game_state.position['y'] - goal[1]
    return np.arctan(delta_y/distance_to_goal) * 360 / (2 * np.pi)

def distance_to_goal_calc(goal, game_state):
    distance = ( 
                    (game_state.position['x'] - goal[0]) ** 2 + \
                    (game_state.position['z'] - goal[2]) ** 2
    ) ** 0.5 

    return distance

def normalize_3d_rotation(delta_x, delta_y, delta_z):
    if delta_x == 0 and delta_y == 0 and delta_z ==0:
        return 0, 0, 0
    norm = (delta_x ** 2 + delta_y ** 2 + delta_z ** 2) ** 0.5 
    return delta_x / norm, delta_y / norm, delta_z / norm

def quaternion_rotate_vector(quat, v): 
    vq = np.quaternion(0, 0, 0, 0)
    vq.imag = v 
    return (quat * vq * quat.inverse()).imag

def quat_from_angle_axis(theta, axis=np.array([0,1,0])):
    axis = axis.astype(np.float)
    axis /= np.linalg.norm(axis)
    return quaternion.from_rotation_vector(theta * axis)


'''
def get_visible_points_v2(x,y,direction,camera_field_of_view,radius):

#camera_field_of_view = 90
#direction = 0.5
direction = (direction+3) %4
    #print ("dircetion = ", direction)
    rotation = direction * 90
    z = y
    radius = radius * 1.2
    #rotation_rad = math.radians(rotation)

    angle_pt_1 = math.radians(rotation - (camera_field_of_view/2))
    angle_pt_2 = math.radians(rotation + (camera_field_of_view/2))

    #p_x = x + radius* math.cos(rotation_rad)  
    #p_z = z + radius * sine(rotation_rad)

    p1_x = x + radius * math.cos(angle_pt_1)
    p1_z = z + radius * math.sin(angle_pt_1)

    p2_x = x + radius * math.cos(angle_pt_2)
    p2_z = z + radius * math.sin(angle_pt_2)

    #print (x,z)
    #print (p1_x,p1_z)
    #print (p2_x,p2_z)

    return  get_points_in_triangle(x,z,p1_x,p1_z,p2_x,p2_z)
'''

'''
def get_points_in_triangle(x,z,p1_x,p1_z,p2_x,p2_z):

    #(𝑥2−𝑥1)(𝑦3−𝑦1)−(𝑦2−𝑦1)(𝑥3−𝑥1)|≠0
    if (p1_x - x )*(p2_z-z) - (p1_z-z)*(p2_x-x) == 0     :
        return ["co linear points"]

    loop_x_min = math.floor(max(min(x,p1_x,p2_x),xMin))
    loop_z_min = math.floor(max(min(z,p1_z,p2_z),zMin))

    loop_x_max = math.ceil(min(max(x,p1_x,p2_x),xMax))
    loop_z_max = math.ceil(min(max(z,p1_z,p2_z),zMax))

    #print ("loop range x = ",loop_x_min,loop_x_max)
    #print ("loop range z = ",loop_z_min,loop_z_max)
    
    true_count = 0
    total_count = 0
    #in_triangle = False 
    visible_points = []

    for i in range(loop_x_min, loop_x_max+1 ): 
        for j in range(loop_z_min, loop_z_max+1):
            if i == xMax or j == zMax:    
                continue
            in_triangle =   pointInTriangle(x,z,p1_x,p1_z,p2_x,p2_z,i,j)
            total_count += 1
            if in_triangle == True :
                visible_points.append((i,j))
                #true_count += 1
    return visible_points
'''

'''

def get_points_in_polygon(x,z,p1_x,p1_z,p2_x,p2_z, p3_x,p4_z):

    #(𝑥2−𝑥1)(𝑦3−𝑦1)−(𝑦2−𝑦1)(𝑥3−𝑥1)|≠0
    if (p1_x - x )*(p2_z-z) - (p1_z-z)*(p2_x-x) == 0     :
        return ["co linear points"]

    loop_x_min = math.floor(max(min(x,p1_x,p2_x,p3_x),xMin))
    loop_z_min = math.floor(max(min(z,p1_z,p2_z,p3_z),zMin))

    loop_x_max = math.ceil(min(max(x,p1_x,p2_x,p3_x),xMax))
    loop_z_max = math.ceil(min(max(z,p1_z,p2_z,p3_z),zMax))

    print ("loop range x = ",loop_x_min,loop_x_max)
    print ("loop range z = ",loop_z_min,loop_z_max)
    
    true_count = 0
    total_count = 0
    #in_triangle = False 
    visible_points = []

    for i in range(loop_x_min, loop_x_max+1 ): 
        for j in range(loop_z_min, loop_z_max+1):
            if i == xMax or j == zMax:    
                #print ("hitting max value ")
                continue
            #in_triangle =   pointInTriangle(x,z,p1_x,p1_z,p2_x,p2_z,i,j)
            in_polygon = pointInPolygon(4, [x,p1_x,p2_x,p3_x], [z,p1_z,p2_z,p3_z],i,j)
            print ("points being checked" , i,j, in_polygon)
            
            total_count += 1
            if in_polygon == True :
                visible_points.append((i,j))
                true_count += 1

    print ("points inside = ", true_count)
    print ("total checked = ", total_count)
    
    return visible_points
'''
'''
def pointInPolygon(nvert, polygonX, polygonY, targetX, targetY):

    #i, j, c = 0;

    #for (i = 0, j = nvert-1; i < nvert; j = i++) {
    j = nvert - 1
    c = False
    i = 0
    for i in range (0, nvert): 
        #if ( ((verty[i]>testy) != (verty[j]>testy)) and (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]) ):
        #    c = not c 
        #j = i
        #i = i + 1

        if( polygonX[i] < polygonX[(i + 1) % nvert]):
            tempX = polygonX[i];
            tempY = polygonX[(i + 1) % nvert]
        else :
            tempX = polygonX[(i + 1) % nvert]
            tempY = polygonX[i]


        #First check if the ray is possible to cross the line
        if (targetX > tempX and targetX <= tempY and (targetY < polygonY[i] or targetY <= polygonY[(i + 1) % nvert])) :
            eps = 0.000001
            #Calculate the equation of the line
            dx = polygonX[(i + 1) % nvert] - polygonX[i];
            dy = polygonY[(i + 1) % nvert] - polygonY[i];

            if (abs(dx) < eps):
                k = 999999999999999999999
            else:
                k = dy / dx

            m = polygonY[i] - k * polygonX[i]
            #Find if the ray crosses the line
            y2 = k * targetX + m;
            if (targetY <= y2) :
                #crossings++;
                c = not c 

    return c
'''

'''
def  pointInTriangle(x1, y1, x2, y2, x3, y3, x, y):

    denominator = ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denominator
    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denominator
    c = 1 - a - b;

    return 0 <= a and a <= 1 and 0 <= b and b <= 1 and -0.000000000001 <= c and c <= 1;
'''

'''
Get the set of points visible from a given position and 
dircetion of the agent
'''
'''
def get_visible_points_v1(x,y,direction,camera_field_of_view,radius):
    
    #𝑝2.𝑥=𝑐𝑝.𝑥+𝑟∗𝑐𝑜𝑠(𝛼+𝜃)
    #𝑝2.𝑦=𝑐𝑝.𝑦+𝑟∗𝑠𝑖𝑛(𝛼+𝜃)
    
    visible_points = []
    #radius = event.camera_clipping_planes[1]/constants.AGENT_STEP_SIZE
    #radius = event.camera_clipping_planes[1]#/constants.AGENT_STEP_SIZE
    #radius = event.camera_clipping_planes[1]#/constants.AGENT_STEP_SIZE
    #print ("direction ", direction)
    #print ("radius = ", radius)
    z = y

    #event.camera_field_of_view = 45
    #print (event.camera_field_of_view/2)
    #print (event.rotation)
    #print (event.rotation - (event.camera_field_of_view/2))
    #print (event.rotation + (event.camera_field_of_view/2))
    
    #angle_pt_1 = math.radians(event.rotation - (event.camera_field_of_view/2))
    #angle_pt_2 = math.radians(event.rotation + (event.camera_field_of_view/2))
    
    rotation = direction * 90
    #radius = radius * 1.2
    angle_pt_1 = math.radians(rotation - (camera_field_of_view/2))
    angle_pt_2 = math.radians(rotation + (camera_field_of_view/2))

    p1_x = x + radius * math.cos(angle_pt_1)
    p1_z = z + radius * math.sin(angle_pt_1)

    p2_x = x + radius * math.cos(angle_pt_2)
    p2_z = z + radius * math.sin(angle_pt_2)

    #print (p2_x-x,p2_z-z )
    #print ((p2_x-x)/(p2_z-z))

    #print ("angle between points p2 and x = ", math.degrees(math.atan((p2_x-x)/(p2_z-z))))
    #print ("angle between points p1 and x = ", math.degrees(math.atan((p1_x-x)/(p1_z-z))))
    #print ("angle between points = ", math.degrees(math.atan(((p2_x-p1_x)/(p2_x-p1_x)))))
    #print ("centre of the curve", x,z)

    pt_1_angle =  math.degrees(math.atan((p1_x-x)/(p1_z-z)))
    pt_2_angle =  math.degrees(math.atan((p2_x-x)/(p2_z-z)))

    #TODO check if pt_1_angle == rotation- camera_filedof_view

    #print ("pt 1 angle = ", pt_1_angle)
    #print ("pt_2_andle = ", pt_2_angle)

    lower_angle = min(pt_1_angle,pt_2_angle)
    higher_angle = max(pt_1_angle,pt_2_angle)

    #print ("first end point of the curve", p1_x, p1_z)
    #print ("second end point of the curve", p2_x,p2_z)

    loop_x_min = max(min(x,p1_x,p2_x),xMin)
    loop_z_min = max(min(z,p1_z,p2_z),zMin)

    loop_x_max = min(max(x,p1_x,p2_x),xMax)
    loop_z_max = min(max(z,p1_z,p2_z),zMax)

    #print ("loop range x = ",loop_x_min,loop_x_max)
    #print ("loop range z = ",loop_z_min,loop_z_max)

    for i in range(math.floor(loop_x_min), math.ceil(loop_x_max) ): #, math.ceil(max(abs(p1_x),abs(p2_x)))):
        for j in range(math.floor(loop_z_min), math.ceil(loop_z_max)):# math.ceil(max(abs(p1_z),abs(p2_z)))):
            #print (i,j)
            if math.sqrt( (i-x)**2 + (j-y)**2) < radius:
                #print ("points inside ", (i,j))
                if  (j==z):
                    #print ("j is z" , j)
                    continue
                current_point_angle = math.degrees(math.atan((i-x)/(j-z)))
                #print ("j not z")
                if current_point_angle >= lower_angle and current_point_angle <= higher_angle :
                    visible_points.append((i,j))
                    #print ("No issue" , (i,j), ", angle = ", current_point_angle)
                else :
                    pass
                    #print ("angle issue",(i,j), ", angle = ", current_point_angle)

    return visible_points
'''
if __name__ == '__main__': 
    #q = flood_fill(0,0,check_validity)
    #print (len(q))
    #print (q)
    #g = graph(xMin,xMax,zMin,zMax)

    x,z = -0.234,-0.476
    x,z = -12,17
    x,z = 0,0
    #p1_x,p1_z = -0.07890,5.1276
    #p1_x,p1_z = 0,-5.1276
    #p1_x,p1_z = 0.0000001,5
    p1_x, p1_z = 0,5
    p2_x,p2_z = 5,0
    p3_x,p3_z = 5,5

    start_time = time.time()
    #for number_points_to_check in range(0,1):
    #    for directions in range(0,1):
    #print (get_points_in_triangle(x,z,p1_x,p1_z,p2_x,p2_z))
    #print (get_points_in_polygon(x,z,p1_x,p1_z,p2_x,p2_z, p3_x,p3_z))
    #print (len(get_visible_points (x,z,1 , 45, 40)))
    #print (len(get_visible_points_old (x,z,0 , 45, 40)))

    radius = 0.1
    p1 = [0.21,-1.68]
    p2 = [-0.14,-2.08]

    end_time = time.time()
    new_pt = get_point_between_points(p1,p2,0.1)

    print ("new point", new_pt )
    print ("processing time = ", end_time-start_time)

