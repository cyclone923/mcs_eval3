import numpy as np
from scipy.spatial.transform import Rotation

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import array
import alphashape
from descartes import PolygonPatch
#from occupancy_grid_a_star.gridmap import OccupancyGridMap
#from occupancy_grid_a_star.a_star import a_star
from scipy.ndimage.measurements import label
from shapely.geometry import Point, Polygon,box,MultiPolygon
from shapely.ops import unary_union
from pprint import pprint
import time
import math

"""
Image -> Point cloud conversion
"""

fig_save_level = 0

all_scatter_points = []
all_plotted_points_3d = []
colors = ['g', 'r', 'y', 'b','c', 'm', 'k','lime','r', 'g','y'] 

rotation_1 = 180
rotation_2 = 145
step_count = 0

map_width = 11
map_length = 11

grid_size = 0.1
#grid_size = 1

displacement = 5.5
rows = int(map_width//grid_size)
cols = int(map_length//grid_size)

from scipy.spatial import Delaunay
import numpy as np


def alpha_shape_stack(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    #print ("all points sent ", points)
    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

def convert_scenes(env, paths):
    for scene_idx, scene_path in enumerate(paths):
        for frame_idx, obs in enumerate(env.run_scene(scene_path)):
            
            for obj in obs.structural_object_list:
                if obj.uuid == "ceiling" or obj.uuid == "floor":
                    print ("ceiling/floor dim ", obj.dimensions)
                    continue
                x_list = []
                y_list = []
                for i in range(4, 8): 
                    x_list.append((obj.dimensions[i]['x'], obj.dimensions[i]['z']))
                    #y_list.append(obj.dimensions[i]['z'])

                print ("structural object data", x_list)#,y_list)
           
            pts, obj_mask = convert_output(obs)
            rgb_img = obs.image_list[0]
            depth_img  = obs.depth_mask_list[0]
            if fig_save_level >= 3:
                rgb_img.save(f'images/{scene_idx}_{frame_idx}_rgb.png', "PNG")
                depth_img.save(f'images/{scene_idx}_{frame_idx}_depth.png', "PNG")
            plot_scene(scene_idx, frame_idx, pts, obj_mask)
        #plot_all(frame_idx+1)

def convert_observation(env,frame_idx, agent_pos=None, rotation=None):
    
    if agent_pos == None and rotatoin == None :
        all_points, obj_masks = convert_output(env)
    else :
        all_points, obj_masks = convert_output_dead_reckoning(env,agent_pos, rotation)
    #occ_map_copy = env.occupancy_map[:]
    #bounding_boxes = plot_scene(1, frame_idx, all_points, obj_mask,env.occupancy_map)
    polygons,object_occupancy_grids = point_cloud_to_polygon(all_points,env.occupancy_map,env.grid_size,obj_masks)
    #process_obj_mask(env.step_output.object_mask_list[-1])
    #return bounding_boxes
    goal_id = find_goal_id(object_occupancy_grids, env.goal_bounding_box, env.occupancy_map.shape, env.grid_size, displacement)
    if goal_id == -1 :
        pass
        #print ("Goal not seen in this frame")
    else :
        env.goals_found = True
        env.goal_calculated_points = object_occupancy_grids[goal_id]
        env.goal_id= goal_id
        #print ("goal found at location : " , object_occupancy_grids[goal_id])
    env.object_mask = obj_masks
    return polygons


def convert_output(env):
    o = env.step_output
    '''
    for obj in o.structural_object_list:
        if obj.uuid == "ceiling" or obj.uuid == "floor":
            print ("ceiling/floor dim ", obj.dimensions)
            continue
    '''
    #objs = o.object_list
    #structs = o.structural_object_list
    img = o.image_list[-1]
    #obj_mask = convert_obj_mask(o.object_mask_list[-1], objs).flatten()
    obj_mask = process_obj_mask(o.object_mask_list[-1]).flatten()
    #depth_mask = np.array(o.depth_mask_list[-1])
    depth_mask = np.array(o.depth_map_list[-1])
    camera_desc = [o.camera_clipping_planes, o.camera_field_of_view,
                   o.position, o.rotation, o.head_tilt]
    pts = depth_to_points(depth_mask, *camera_desc)
    return pts,obj_mask

def convert_output_dead_reckoning(env,agent_pos, rotation ):
    o = env.step_output
    #objs = o.object_list
    structs = o.structural_object_list
    img = o.image_list[-1]
    #obj_mask = convert_obj_mask(o.object_mask_list[-1], objs).flatten()
    obj_mask = process_obj_mask(o.object_mask_list[-1]).flatten()
    depth_mask = np.array(o.depth_map_list[-1])
    #agent_pos_d = {}
    #agent_pos_d['x'] = agent_pos[0]
    #agent_pos_d['y'] = agent_pos[1]
    #agent_pos_d['z'] = agent_pos[2]
    camera_desc = [o.camera_clipping_planes, o.camera_field_of_view,
                   agent_pos, rotation, o.head_tilt]
    pts = depth_to_points(depth_mask, *camera_desc)
    return pts, obj_mask

def convert_obj_mask(mask, objs):
    col_to_tuple = lambda col: (col['r'], col['g'], col['b'])
    color_map = {col_to_tuple(o.color):i for i, o in enumerate(objs)}
    arr_mask = np.array(mask)
    out_mask = -np.ones(arr_mask.shape[0:2], dtype=np.int8)
    for x in range(arr_mask.shape[0]):
        for y in range(arr_mask.shape[1]):
            idx = color_map.get(tuple(arr_mask[x, y]), -1)
            out_mask[x, y] = idx
    return out_mask

def process_obj_mask(mask):
    #arr_mask = np.array(self.event.object_mask_list[-1])
    arr_mask = np.array(mask)
    #print (arr_mask.shape)
    color_dict = {}
    out_mask = -np.ones(arr_mask.shape[0:2], dtype=np.int8)
    color_id = 0
    for x in range(arr_mask.shape[0]):
        for y in range(arr_mask.shape[1]):
            if tuple(arr_mask[x,y]) not in color_dict:
                color_dict[tuple(arr_mask[x,y])] = color_id
                out_mask[x,y] = color_id
                color_id += 1
            else :
                out_mask[x,y] = color_dict[tuple(arr_mask[x,y])]

    #print (color_dict,np.amax(out_mask),color_id)
    #self.event.object_mask_list[-1].save("object_mask_trial.png")
    #print ("number of objects = ", color_id)
    return out_mask

def depth_to_points(depth, camera_clipping_planes,
                    camera_field_of_view, pos_dict, rotation, tilt):
    """ Convert a depth map and camera description into a list of 3D world
    points.
    Args:
        depth (np.ndarray): HxW depth mask
        camera_[...], pos_dict, rotation, tilt:
            Camera info from MCS step output
    Returns:
        Px3 np.ndarray of (x,y,z) positions for each of P points.
    """
    # Get local offset from the camera of each pixel
    global step_count
    local_pts = depth_to_local(depth, camera_clipping_planes, camera_field_of_view)
    # Convert to world space
    # Use rotation & tilt to calculate rotation matrix.
    #rot = Rotation.from_euler('yx', (rotation, tilt), degrees=True)
    rot = Rotation.from_euler('yx', (360 - rotation,360- tilt), degrees=True)
    step_count += 1
    pos_to_list = lambda x: [x['x'], x['y'], x['z']]
    pos = pos_to_list(pos_dict)
    # Apply rotation, offset by camera position to get global coords
    global_pts = np.matmul(local_pts, rot.as_matrix()) + pos
    # Flatten to a list of points
    flat_list_pts = global_pts.reshape(-1, global_pts.shape[-1])
    return flat_list_pts


def depth_to_local(depth, clip_planes, fov_deg):
    """ Calculate local offset of each pixel in a depth mask.
    Args:
        depth (np.ndarray): HxW depth image array with values between 0-255
        clip_planes: Tuple of (near, far) clip plane distances.
        fov_deg: Vertical FOV in degrees.
    Returns:
        HxWx3 np.ndarray of each pixel's local (x,y,z) offset from the camera.
    """
    """ Determine the 'UV' image-space coodinates for each pixel.
    These range from (-1, 1), with the top left pixel at index [0,0] having
    UV coords (-1, 1).
    """
    aspect_ratio = (depth.shape[1], depth.shape[0])
    #print ("aspect ratio" ,aspect_ratio)

    idx_grid = np.meshgrid(*[np.arange(ar) for ar in aspect_ratio])

    px_arr = np.stack(idx_grid, axis=-1) # Each pixel's index
    uv_arr = px_arr*[2/w for w in aspect_ratio]-1

    uv_arr[:, :, 1] *= -1 # Each pixel's UV coords

    """ Convert the depth mask values into per-pixel world-space depth
    measurements using the provided clip plane distances.
    """
    z_depth = depth[:]
    """ Determine vertical & horizontal FOV in radians.
    Use the UV coordinate values and tan(fov/2) to determine the 'XY' direction
    vector for each pixel.
    """
    vfov = np.radians(fov_deg)
    #hfov = np.radians(fov_deg*aspect_ratio[0]/aspect_ratio[1])
    hfov = 2*math.atan(math.tan(vfov/2) * (aspect_ratio[0]/aspect_ratio[1]))
    tans = np.array([np.tan(fov/2) for fov in (hfov, vfov)])
    px_dir_vec = uv_arr * tans
    """ Add Z coordinate and scale to the pixel's known depth.  """
    const_zs = np.ones((px_dir_vec.shape[0:2])+(1,))
    px_dir_vec = np.concatenate((px_dir_vec, const_zs), axis=-1)
    camera_offsets = px_dir_vec * np.expand_dims(z_depth, axis=-1)
    return camera_offsets


def occupancy_map_update (x,y,z,occupancy_map):
    start_time = time.time()
    x = np.array(x,dtype=float)
    y = np.array(y,dtype=float)
    z = np.array(z,dtype=float)
    x = (x + (5.5))/grid_size
    z = (z + (5.5))/grid_size
    y = (y/grid_size)
    number_points_in_cell =0  
    number_wall_points_in_cell = 0
    min_number_points_cell = 3 
    min_number_wall_points_cell = 3

    for i in range(0,occupancy_map.shape[0]):
        correct_x_range =  np.where(np.logical_and(x>=i, x<i+1))
        for j in range(0,occupancy_map.shape[1]):
            number_points_in_cell =0  
            number_wall_points_in_cell = 0
            if occupancy_map[i][j] != 0 :
                continue
            for elem in correct_x_range[0]:
                if z[elem] >=j and z[elem]< j+1 :
                    number_points_in_cell += 1
                    """
                    Clarify this in depth later
                    """
                    if y[elem] >= 2.15/grid_size :
                        number_wall_points_in_cell += 1

            if number_wall_points_in_cell >= min_number_wall_points_cell :
                occupancy_map[i][j] = 7
                continue

            if number_points_in_cell >= min_number_points_cell :
                occupancy_map[i][j] = 1 

    print ("time taken to update map : " , time.time()-start_time)
    return occupancy_map

def merge_occupancy_map(occupancy_map, new_occupancy_map):
    #return np.where(occupancy_map!=0 or new_occupancy_map != 0 , 1 ,0)
    #temp_occupancy_map = np.zeros(occupancy_map.shape)
    for i in range(0,occupancy_map.shape[0]):
        for j in range(0,occupancy_map.shape[1]):
            if occupancy_map[i][j] != 0 or new_occupancy_map[i][j] !=0 :
                occupancy_map[i][j] = 1
            
    #return temp_occupancy_map
    return occupancy_map


def find_goal_id(object_occupancy_grids,goal_bounding_box, size, scale,displacement):
    start_time = time.time()
    max_intersect_area = 0
    goal_object_id = -1
    #print ("number of objects seen", len (object_occupancy_grids))
    #print ("goal object bounding box" , goal_bounding_box.exterior.coords.xy)
    #print ("goal object area" , goal_bounding_box.area)
    for key,values in object_occupancy_grids.items():
        obj_occ_map = get_occupancy_from_points( values,size)   
        obj_polygon = polygon_simplify(occupancy_to_polygons( obj_occ_map, scale,displacement ))
        intersect_area = obj_polygon.intersection(goal_bounding_box).area
        #print ("obj polygons points" ,obj_polygon.exterior.coords.xy)
        #print ("Intersection area : ", intersect_area)
        #print ("polygon area" , obj_polygon.area)
        if intersect_area > max_intersect_area :
            goal_object_id = key
            max_intersect_area = intersect_area

    #print ("time taken = ", time.time()-start_time, " ####################\n")
    return goal_object_id

def get_occupancy_from_points(points, size):
    occupancy_map = np.zeros(size)
    for item in points:
        occupancy_map[item[0],item[1]] = 1
    return occupancy_map
    
def occupancy_to_polygons(occupancy, scale, displacement):

    # occupancy - binary numpy array with each pixel representing a scale*scale area
    # scale - width and height of each pixel in occupancy
    # returns union polgon
    d = displacement
    polygons = []
    for i in range(occupancy.shape[0]):
        occupied = np.nonzero(occupancy[i,:])[0]
        if len(occupied) == 0:
            continue
        minX = occupied[0]
        maxX = minX+1
        p = occupied[0]
        for o in occupied[1:]:
            if p+1 == o:
                maxX = o+1
            else:
                #polygons.append(box(minX*scale-d, i*scale-d, maxX*scale-d, (i+1)*scale-d))
                polygons.append(box(i*scale-d, minX*scale-d, (i+1)*scale-d, maxX*scale-d))
                minX = o
                maxX = o+1
            p = o
        #polygons.append(box(minX*scale-d, i*scale-d, maxX*scale-d, (i+1)*scale-d))
        polygons.append(box(i*scale-d, minX*scale-d, (i+1)*scale-d, maxX*scale-d))
    return unary_union(polygons)

def point_cloud_to_polygon(points,occupancy_map,grid_size,obj_masks = None):
    
    start_time = time.time()
    sample = 2
    points = points[::sample]
    obj_masks = obj_masks[::sample]
    new_occupancy_map = np.zeros(occupancy_map.shape)
    object_occupancy_grids = {}
    #for i,pt in enumerate( points ):
    for pt,object_id in zip( points,obj_masks ):
        if pt[1] > 0.05  and pt[1] <= 3: 
            occupancy_x = int((pt[0] + 5.5)/grid_size)
            occupancy_z = int((pt[2] + 5.5)/grid_size)
            new_occupancy_map[occupancy_x][occupancy_z] += 1    
            #object_id = obj_masks[i]
            if object_id in object_occupancy_grids:
                if (occupancy_x,occupancy_z) not in object_occupancy_grids[object_id] :
                    object_occupancy_grids[object_id].append((occupancy_x,occupancy_z))
            else :
                object_occupancy_grids[object_id] = [(occupancy_x,occupancy_z)]
                    

    new_occupancy_map = np.where(new_occupancy_map>=3, 1, 0)
    occupancy_map = merge_occupancy_map(occupancy_map, new_occupancy_map)
    #print ("[new] time taken to project into 2d occupancy map : " , time.time()-start_time)

    #print (len(object_occupancy_grids))
    start_time = time.time()
    all_polygons = occupancy_to_polygons(occupancy_map, grid_size, displacement  )
    #print ("[new] time taken to update polygons : " , time.time()-start_time)
    #time.sleep(1)
    simplified_polygon = polygon_simplify(all_polygons,0.08)
    show_animation = True
    #plt.close()
    if show_animation :
        plt.cla()
        #patch1 = PolygonPatch(all_polygons,fc='grey', ec="black", alpha=0.2, zorder=1)
        patch1 = PolygonPatch(simplified_polygon,fc='grey', ec="black", alpha=0.2, zorder=1)
        
        plt.gca().add_patch(patch1)
        plt.axis("equal")
        plt.pause(0.01)
    return simplified_polygon,object_occupancy_grids


        
def polygon_simplify(all_polygons,scale=0.0):
    
    simplified_polygon = []
    ####################Testing simplify ######################
    
    if all_polygons.geom_type == 'Polygon':
        simplified_polygon = all_polygons.simplify(scale)
        '''
        new_trial_5 = all_polygons.simplify(0.0)        
        print ("Simplified number 5 : ", len(list(new_trial_5.exterior.coords)))
        '''
         
    elif all_polygons.geom_type == 'MultiPolygon':
        for i,polygon in enumerate(all_polygons) :
            simplified_polygon.append(all_polygons[i].simplify(scale))
            #print ("[Multi] before change exterior pts : ", len(list(all_polygons[i].exterior.coords)))
        simplified_polygon = MultiPolygon(simplified_polygon)
        for i,polygon in enumerate(simplified_polygon) :
            pass
            #print ("[Multi] before-after change exterior pts : ", len(list(all_polygons[i].exterior.coords)),",",len(list(simplified_polygon[i].exterior.coords)))

    return simplified_polygon
"""
Plotting
"""

def plot_pts(ax, pts):
    ax.scatter(*zip(*pts), zdir='y', s=0.1, alpha=0.5)

def plot_pts_color(ax, pts,color_id):
    ax.scatter(*zip(*pts), zdir='y', s=0.1, alpha=0.5,c=colors[color_id])

def plot_scene(scene_idx, frame_idx, pts, obj_mask,occupancy_map = None):
    global all_scatter_points,grid_size
    '''
    fig1 = plt.figure(1,figsize=(10, 10), dpi=100)
    #fig = plt.figure()
    ax = fig1.add_subplot(1, 1, 1, projection='3d', label= "3")
    '''
    #print (pts.shape)
    #print (type(pts))
    #print (type(pts[0]))
    #exit()
    all_plotted_points = []
    #all_plotted_points_3d = []
    for obj_id in np.unique(obj_mask):
        print(obj_id)
        obj_pts = pts[obj_mask==obj_id]
        obj_pts = obj_pts[::5]
        all_plotted_points.append(obj_pts)
        #all_plotted_points_3d.extend(obj_pts)
        all_plotted_points_3d.append(obj_pts)
        #print (len(obj_pts), len(obj_pts[0]))
        #plot_pts(ax, obj_pts)
    '''
    bound = 20
    ax.set_xlim([-bound/2, bound/2])
    ax.set_ylim([-bound/2, bound/2])
    ax.set_zlim([-bound/2, bound/2])
    if fig_save_level >= 1:
        fig1.savefig(f'images/{scene_idx}_{frame_idx}_points.png')
    plt.xlabel('x direction')
    plt.ylabel('z direction')
    plt.close(fig1)
    '''
    modified_x_points = []
    modified_y_points = []
    modified_z_points = []
    
    plot_points_one_frame = []
    
    '''
    print (all_plotted_points.shape)
    print (all_plotted_points[0].shape)
    print (all_plotted_points[1].shape)
    exit()
    '''
    start_time = time.time()
    new_occupancy_map = np.zeros(occupancy_map.shape)
    new_occupancy_map_2 = np.zeros(occupancy_map.shape)
    alpha_shapes = []
    plot_all_points_one_frame = []
    all_plotted_points = array(all_plotted_points,dtype=object)


    for h in range(all_plotted_points.shape[0]):
        #new_occupancy_map[all_plotted_points[h]] 
        for i,pt in enumerate( all_plotted_points[h] ):
            if pt[1] > 0.05  and pt[1] <= 3: 
                #modified_x_points.append(pt[i][0])
                #modified_y_points.append(pt[i][1])
                #modified_z_points.append(pt[i][2])
                modified_x_points.append(pt[0])
                modified_y_points.append(pt[1])
                modified_z_points.append(pt[2])
                occupancy_x = int((pt[0] + 5.5)/grid_size)
                occupancy_z = int((pt[2] + 5.5)/grid_size)
                new_occupancy_map[occupancy_x][occupancy_z] += 1
                plot_points_one_frame.append((pt[0],pt[2]))
        #alpha_shapes.append(alphashape.alphashape(plot_points_one_frame))
        #all_outer_edges.append(alpha_shape_stack(np.array(plot_points_one_frame,dtype=object), 2.5))
        all_scatter_points = union(all_scatter_points,plot_points_one_frame)
        plot_all_points_one_frame.append(np.array(plot_points_one_frame,dtype=object))
        plot_points_one_frame = []
    
    #for 
    '''
    TODO - single line map building
    ''
    for h in range(all_plotted_points.shape[0]):
        #new_occupancy_map_2 = 
        points=all_plotted_points[h]
        print (points.shape)
        new_occupancy_map_2[int(points[:,0] / grid_size) , int(points[:,1] / grid_size)] = 1

    print ("new occ" , new_occupancy_map_2)
    '''


    new_occupancy_map = np.where(new_occupancy_map>=3, 1, 0)
    occupancy_map = merge_occupancy_map(occupancy_map, new_occupancy_map)
    #print ("new points map \n", new_occupancy_map)
    #print ("merged points map \n", merged_occupancy_map)


    print ("time taken to project into 2d occupancy map : " , time.time()-start_time)
    #k_points = [0]*len(modified_x_points)

    #print (min(modified_x_points), max(modified_x_points))
    #print (min(modified_z_points), max(modified_z_points))

    #print (len(modified_x_points), all_plotted_points[0][0].shape)
    #ax.scatter3D(modified_x_points, modified_y_points, k_points, c=k_points, cmap='hsv');
    #plt.scatter(modified_x_points, modified_y_points, cmap='hsv');
    #ax.scatter(modified_x_points, modified_z_points, cmap='hsv');
    

    #for alpha_shape in alpha_shapes :
    #    ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))
    '''
    fig2 = plt.figure(1,figsize=(10, 10), dpi=100)
    ax = fig2.add_subplot()
    ax.set_xlim([-10,10])
    ax.set_ylim([-10,10])
    plt.xlabel('x direction')
    plt.ylabel('z direction')
    if fig_save_level >= 2:
        #fig2.savefig(f'MCS_exploration/images/{scene_idx}_{frame_idx}_points_2d.png')
        fig2.savefig(f'images/{scene_idx}_{frame_idx}_points_2d.png')
    #print ("all scatter points numbers",len(all_scatter_points))
    #plt.show()
    plt.close(fig2)
    '''
    #print ("Reached occupancy map ")
    cell_size = 1
    #occupancy_map = occupancy_map_update(modified_x_points,modified_y_points,modified_z_points,occupancy_map)
    #gmap = OccupancyGridMap(occupancy_map, cell_size)
    
    #if np.all(merged_occupancy_map) == np.all(occupancy_map):
    #    print ("merging succesfully complete")
    start_time = time.time()
    unary_union_polygons = occupancy_to_polygons(occupancy_map, grid_size  )
    print ("time taken to update polygons : " , time.time()-start_time)
    #print (type(unary_union_polygons))
    #for item in list(unary_union_polygons) :
    #    print ("type of iterms " ,type(item))
    #print (pprint(list(unary_union_polygons)))
    #if len(list(unary_union_polygons)) == 0:  
    #    print ("empty first")
    show_animation = True
    plt.close()
    if show_animation and unary_union_polygons != None:
        plt.cla()
        #for obstacle in shapely_object_obstacles:
        patch1 = PolygonPatch(unary_union_polygons,fc='grey', ec="black", alpha=0.2, zorder=1)
        plt.gca().add_patch(patch1)
        #obstacle.plot("-g")

        plt.axis("equal")
        plt.pause(0.01)
        #plt.close()
    #print ("at return of union polys")
    return unary_union_polygons
    

    shapely_obstacles = {}
    structure = np.ones((3, 3), dtype=np.int)

    for i in range(occupancy_map.shape[0]):
        #shapely_obstacles.append([])
        for j in range(occupancy_map.shape[1]):
           if occupancy_map[i][j] != 0:
                #k = (i*grid_size)-5.5
                #l = (j*grid_size)-5.5
                k = i - 5.5 /grid_size
                l = j - 5.5 / grid_size
                #shapely_obstacles[i].append(Polygon([(k,l),(k+i,l),(k+1,l+1),(k,l+1)] )) 
                shapely_obstacles[(i,j)] = (Polygon([(k,l),(k+1,l),(k+1,l+1),(k,l+1)] )) 

    
    labeled,n_components = label(occupancy_map, structure)
    shapely_object_obstacles = [None] * n_components 
    #print ("shapely obstacles ", shapely_obstacles)

    #for i in range(1,n_components):
    #print ("labeled" , labeled)
    #print ("n comps" , n_components)
    
    for i in range(occupancy_map.shape[0]):
        for j in range(occupancy_map.shape[1]):
            if labeled[i][j] != 0:
                #print ("labeled not 0 - shold be here emultiple times")
                if shapely_object_obstacles[labeled[i][j]-1] == None :
                    #print ("labeled initial- shold be here once")
                    shapely_object_obstacles[labeled[i][j]-1] = shapely_obstacles[(i,j)]
                else :
                    #print ("labeled union  - multiple times")
                    #print ("exterior coords before union ", shapely_object_obstacles[labeled[i][j]-1].exterior.coords.xy)
                    #print ("shapely point being added - " , shapely_obstacles[(i,j)].exterior.coords.xy)
                    shapely_object_obstacles[labeled[i][j]-1] = shapely_object_obstacles[labeled[i][j]-1].union(shapely_obstacles[(i,j)])
                    #print ("exterior coords after union ", shapely_object_obstacles[labeled[i][j]-1].exterior.coords)

    #print (shapely_obstacles)
    print ("object obstacles " ,shapely_object_obstacles)

    show_animation = True
    plt.close()
    if show_animation:
        plt.cla()
        #plt.plot(x, y, "or")
        # plt.plot(gx, gy, "ob")
        # poly.plot("-r")

        #for obstacle in shapely_object_obstacles:
        #for obstacle in shapely_obstacles.values():
        #    patch1 = PolygonPatch(obstacle,fc='grey', ec="black", alpha=0.2, zorder=1)
        #    plt.gca().add_patch(patch1)
        #for obstacle in shapely_obstacles.values():
        for obstacle in shapely_object_obstacles:
            patch1 = PolygonPatch(obstacle,fc='grey', ec="black", alpha=0.2, zorder=1)
            plt.gca().add_patch(patch1)
            #obstacle.plot("-g")

        plt.axis("equal")
        plt.pause(1)
        #plt.close()
    return shapely_object_obstacles
    #exit()

    #start_node = (5, 5)
    #goal_node = (13,8)
    #path, path_px = a_star(start_node, goal_node, gmap, movement='4N')
    #print ("path found : ",path)
    ##plot_path(path)
    '''
    structure[0][0] = 0
    structure[0][2] = 0
    structure[2][0] = 0
    structure[2][2] = 0
    '''
    horizontal_wall_occupancy_map = np.zeros(occupancy_map.shape) 
    #print ("horizontal wall occupancy shape",horizontal_wall_occupancy_map.shape)
    vertical_wall_occupancy_map = np.zeros(occupancy_map.shape) 
    #print ("vertical wall occupancy shape",vertical_wall_occupancy_map.shape)
    object_occupancy_map = np.zeros(occupancy_map.shape) 
    #print ("wall occupancy shape", object_occupancy_map.shape)

    for i in range(occupancy_map.shape[0]):
        horizontal_wall_locations = np.where(occupancy_map[i] == 7)
        
        if len(horizontal_wall_locations[0]) > float(occupancy_map.shape[1]/5):
            #print ("horizontal points being added", horizontal_wall_locations)
            
            for elem in horizontal_wall_locations[0] :
                horizontal_wall_occupancy_map[i][elem] = 1

        #object_locations = np.where(occupancy_map[i] == 1 )
        #for elem in object_locations[0] :
        #    print ("object points being added", object_locations)
        #    object_occupancy_map[i][elem] = 1

        vertical_wall_locations = np.where(occupancy_map[:,i] == 7)
        if len(vertical_wall_locations[0]) > float(occupancy_map.shape[0]/5):
            #print ("vertical points being added", vertical_wall_locations)
            for elem in vertical_wall_locations[0] :
                vertical_wall_occupancy_map[elem][i] = 1
    #horizontal_wall_occupancy_map = np.where(occupancy_map==7,1,0)    
    #vertical_wall_occupancy_map = np.where(occupancy_map==7,1,0)    
    object_occupancy_map = np.where(occupancy_map==1, occupancy_map,0)
    '''
    print ("\n full occupancy map \n" , occupancy_map)
    print ("horizontal wall map \n ", horizontal_wall_occupancy_map)
    print ("\n object map \n ", object_occupancy_map)
    print ("\n vertical wall map \n ",vertical_wall_occupancy_map)
    '''

    #exit()    

    horizontal_wall_labeled, h_w_n_components = label(horizontal_wall_occupancy_map, structure)
    object_labeled, obj_n_components = label(object_occupancy_map, structure)
    vertical_wall_labeled, v_w_n_components = label(vertical_wall_occupancy_map, structure)
    
    #print ("\n horizontal labeled wall \n " , horizontal_wall_labeled)
    #print ("\n object labeled  \n " , object_labeled)
    #print ("\n Vertical labeled wall \n " , vertical_wall_labeled)
    #labeled, ncomponents = label(occupancy_map, structure)
    #print ("number components ", ncomponents)
    #print (labeled)

    outer_points_all = []


    all_labeled = []
    all_labeled.append(horizontal_wall_labeled)
    all_labeled.append(object_labeled)
    all_labeled.append(vertical_wall_labeled)

    all_occupancy_map = []
    all_occupancy_map.append(horizontal_wall_occupancy_map)
    all_occupancy_map.append(vertical_wall_occupancy_map)
    all_occupancy_map.append(object_occupancy_map)

    ncomponents = []
    ncomponents.append(h_w_n_components)
    ncomponents.append(obj_n_components)
    ncomponents.append(v_w_n_components)
    
    
    for c,current_occupancy_map in enumerate(all_occupancy_map) :
        outer_points_all.append({})
        for i in range(0,current_occupancy_map.shape[0]):
            #for j in range(0,occupancy_map.shape[1]):
            for k in range(1,ncomponents[c]+1):
                points_this_row = np.where(all_labeled[c][i] == k )
                #print (occupancy_map[i])
                if len(points_this_row[0]) == 0 :
                    continue
                row_point_1 = (i,min(points_this_row[0]))
                row_point_2 = (i,max(points_this_row[0])+1)
                if k not in outer_points_all[c] :
                    outer_points_all[c][k] = []
                    
                for j in range (min(points_this_row[0]), max(points_this_row[0])+2):
                    #outer_points_all[c][k].append((i,min(points_this_row[0])))
                    #outer_points_all[c][k].append((i,max(points_this_row[0])+1))
                    if (i,j) not in outer_points_all[c][k]: 
                        outer_points_all[c][k].append((i,j))
                        outer_points_all[c][k].append((i+1,j))
    
    '''
    For objects which take up only 1 pixel - will be used very very rarely in our scenario
    '''
    for o,outer_points in enumerate(outer_points_all) :
        for key,value in outer_points.items() :
            if len(value) < 3 :
                #outer_points_all[c][key].append((value[0][0]+1,value[0][1]))
                #outer_points_all[c][key].append((value[1][0]+1,value[1][1]))
                outer_points[key].append((value[0][0]+1,value[0][1]))
                outer_points[key].append((value[1][0]+1,value[1][1]))
            #outer_points_all[c][key] = np.array(outer_points_all[c][key],dtype=object)
            outer_points[key] = np.array(outer_points[key],dtype=object)

    #print (outer_points)
    #gmap.plot()

    all_outer_edges = []
    print (ncomponents)
    all_bounding_box_points = []

    for c,n_components in enumerate(ncomponents) :
        outer_edges = []
        for i in range(1,n_components+1):
            bounding_box_object = []
            #all_outer_edges.append(alpha_shape_stack(np.array(outer_points[k],dtype=object), 2.5))
            outer_edges.append(alpha_shape_stack(outer_points_all[c][i], 2.5))
            #for 
            for x,y in enumerate(outer_edges[-1]) :
                bounding_box_object.append(((outer_points_all[c][i][x, 0]*grid_size) - 5.5, (outer_points_all[c][i][x, 1]*grid_size) - 5.5))
                
            all_bounding_box_points.append(bounding_box_object)
        all_outer_edges.append(outer_edges)


    #print ("all outer points len", len(outer_points_all))
    #print ("outer edges len", len(all_outer_edges) )
    
    #print (outer_points_all)
    #print ("all outer edges ", all_outer_edges )
    #print ("" , len())
    plt.close()
    all_bounding_box_points_2 = []

    for c,outer_edge in enumerate(all_outer_edges):
        bounding_box_object = []
        for l,edges in enumerate(outer_edge) :
            bounding_box_points_x = []
            bounding_box_points_z = []
            for i, j in edges:
                #print (plot_all_points_one_frame[l].shape)
                #plt.plot(plot_all_points_one_frame[l][:, 0], plot_all_points_one_frame[l][:, 1], '.')
                #plt.plot(plot_all_points_one_frame[l][[i, j], 0], plot_all_points_one_frame[l][[i, j], 1])
                plt.plot(outer_points_all[c][l+1][[i, j], 0], outer_points_all[c][l+1][[i, j], 1])
                #print ("all edgess being plotted", outer_points_all[c][l+1][[i, j], 0], outer_points_all[c][l+1][[i, j], 1])
                #print ("all edgess being plotted points", outer_points_all[c][l+1][i, [0,1]], outer_points_all[c][l+1][j, [0,1]])
                #bounding_box_points_x.append((outer_points_all[c][l+1][i, 0]*grid_size) - 5.5)
                #bounding_box_points_z.append((outer_points_all[c][l+1][i, 1]*grid_size) - 5.5)
                #print ("being added to list" , outer_points_all[c][l+1][i, 0], outer_points_all[c][l+1][1, 1])
                #bounding_box_object.append(((outer_points_all[c][l+1][i, 0]*grid_size) - 5.5, (outer_points_all[c][l+1][i, 1]*grid_size) - 5.5))
            #bounding_box_object.append()
            #bounding_box_object.append((bounding_box_points_x,bounding_box_points_z ))
            #all_bounding_box_points.append((bounding_box_points_x,bounding_box_points_z ))
            #all_bounding_box_points.append(bounding_box_object)
        #all_bounding_box_points.append(bounding_box_object )

    #print (all_bounding_box_points)
    #print (all_bounding_box_points_2)
    #flag = 0
    #for item in all_bounding_box_points :
    #    if item in all_bounding_box_points_2 : 
    #        continue
    #    else :
    #        flag =1 

    #print ("Flag ", flag)
    plt.savefig("boudning_boxes.png")
    #plt.show()
    #exit()
    return all_bounding_box_points

def plot_path(path):
    start_x, start_y = path[0]
    goal_x, goal_y = path[-1]

    # plot path
    path_arr = np.array(path)
    plt.plot(path_arr[:, 0], path_arr[:, 1], 'y')

    # plot start point
    plt.plot(start_x, start_y, 'ro')

    # plot goal point
    plt.plot(goal_x, goal_y, 'go')

    plt.show()

def plot_all(loop_number):
    global occupancy_map

    fig1 = plt.figure(3,figsize=(10, 10), dpi=100)
    ax = fig1.add_subplot(1, 1, 1, projection='3d')
    #all_plotted_points_3d
    for i in range(len(all_plotted_points_3d)):
        plot_pts_color(ax, all_plotted_points_3d[i],i)
    bound = 20

    ax.set_xlim([-bound/2, bound/2])
    ax.set_ylim([-bound/2, bound/2])
    ax.set_zlim([-bound/2, bound/2])
    if fig_save_level >= 1:
        #fig1.savefig(f'MCS_exploration/images/{loop_number}_combined_points_3d.png')
        fig1.savefig(f'images/{loop_number}_combined_points_3d.png')
    #plt.show()
    plt.xlabel('x direction')
    plt.ylabel('z direction')
    #plt.zlabel('y direction')
    #plt.show()
    plt.close(fig1)
    #'''
    fig = plt.figure(4)
    ax = fig.add_subplot()
    #x_list = [elem[0] for elem in all_scatter_points]
    #z_list = [elem[1] for elem in all_scatter_points]
    x_list= []
    z_list = []
    for elem in all_scatter_points :
        x_list.append(elem[0])
        z_list.append(elem[1])
        
    ax.scatter(x_list,z_list,cmap='hsv')
    ax.set_xlim([-10,10])
    ax.set_ylim([-10,10])
    plt.xlabel('x direction')
    plt.ylabel('z direction')
    if fig_save_level >= 1:
        #fig.savefig(f'MCS_exploration/images/{loop_number}_points_2d_combined.png')
        fig.savefig(f'images/{loop_number}_points_2d_combined.png')
    #plt.show()
    plt.close(fig)


def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))
