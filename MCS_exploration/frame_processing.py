import numpy as np
from scipy.spatial.transform import Rotation
#from MCS_exploration.obstacle import Obstacle

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import array
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

def convert_observation(env,frame_idx, agent_pos, rotation):
    start_time = time.time()
    all_points, obj_masks = convert_output_dead_reckoning(env,agent_pos, rotation)
    env.occupancy_map, polygons,object_occupancy_grids = point_cloud_to_polygon(all_points,env.occupancy_map,env.grid_size,env.displacement,env.obj_mask)
    if env.trophy_location != None :
        #print ("trophy location not none in observation", env.trophy_location)
        update_goal_bounding_box(all_points, env)
    return polygons, object_occupancy_grids

def convert_output_dead_reckoning(env,agent_pos, rotation ):
    o = env.step_output
    structs = o.structural_object_list
    img = o.image_list[-1]
    
    obj_mask = None
    depth_mask = np.array(o.depth_map_list[-1])
    camera_desc = [o.camera_clipping_planes, o.camera_field_of_view,
                   agent_pos, rotation, o.head_tilt]
    start_time = time.time()
    pts = depth_to_points(depth_mask, *camera_desc)
    #print ("time taken for getting point cloud from depth img", time.time()-start_time)
    return pts, obj_mask


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
    local_pts = depth_to_local(depth, camera_clipping_planes, camera_field_of_view)
    # Convert to world space
    # Use rotation & tilt to calculate rotation matrix.
    #rot = Rotation.from_euler('yx', (rotation, tilt), degrees=True)
    rot = Rotation.from_euler('yx', (360 - rotation,360- tilt), degrees=True)
    #rot = Rotation.from_euler('yx', (rotation,360- tilt), degrees=True)
    pos_to_list = lambda x: [x['x'], x['y'], x['z']]
    pos = pos_to_list(pos_dict)
    # Apply rotation, offset by camera position to get global coords
    #global_pts = np.matmul(local_pts, rot.as_matrix()) + pos
    global_pts = np.matmul(local_pts, rot.as_dcm()) + pos
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

def merge_occupancy_map(occupancy_map, new_occupancy_map):
    return  np.where(occupancy_map == 0 , np.where(new_occupancy_map == 0,0,1),1)


def find_goal_id(object_occupancy_grids,goal_bounding_box, size, scale,displacement):
    start_time = time.time()
    max_intersect_area = 0.0001
    goal_object_id = -1
    for key,values in object_occupancy_grids.items():
        obj_occ_map = get_occupancy_from_points( values,size)   
        obj_polygon = polygon_simplify(occupancy_to_polygons( obj_occ_map, scale,displacement ))
        intersect_area = obj_polygon.intersection(goal_bounding_box).area
        if intersect_area > max_intersect_area :
            goal_object_id = key
            max_intersect_area = intersect_area

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

def point_cloud_to_polygon(points,occupancy_map,grid_size, displacement, obj_masks= None):
    '''
    Fucntion to create a set of obstacle polygons a point cloud

    Args :
        points        : 3D point cloud of the current observed from
        occupnacy_map : current occupancy map of the world as a 2D numpy array
        grid_size     : size of each grid in the occupancy map (in meters)
        displacement  : Dispalcement of the points from real global points to correspond to
                        points in the occupancy map
        obj_masks     : Obj masks of the current frame (either from the simulator in level 2 or 
                        Vision model in level 1)

    Returns :
        Occupancy map : updated occupancy map of the world based on the current fram 3d point cloud
        Simplified Polygon : Polygon consisting of all the obstacles in the environment, 
                             constructed based on the updated occupancy map (shapely polygon object)
        Object_occupancy_grids_row_view : Dictionary containng obstacle data where each obstacle 
                                            and it's correspongng position in the occupancy map  
    '''
    start_time = time.time()

    #print ("max min x", np.amax(points[:,0]), np.amin(points[:,0]))
    #print ("max min z", np.amax(points[:,2]), np.amin(points[:,2]))

    object_occupancy_grids_row_view = {}
    np_occ_map = np.zeros(occupancy_map.shape)
    arr_mask = np.array(obj_masks)
    obj_masks = arr_mask.reshape(-1, arr_mask.shape[-1])
    #obj_masks = obj_masks.flatten()
    sample = 1
    points = points[::sample]
    obj_masks_new = obj_masks[::sample]
    obj_masks = obj_masks_new[:]
    new_points = np.where((points[:,1] > 0.05) & (points[:,1] <= 3))#,1,0).reshape(points.shape[0],1)
    points = points[new_points[0]]
    obj_masks = obj_masks[new_points[0]]
    ar_row_view = obj_masks.view('|S%d' % (obj_masks.itemsize * obj_masks.shape[1]))
    unique_row_view = np.unique(ar_row_view)
    #unique_row_view = np.unique(obj_masks)
    #exit()

    for elem in unique_row_view :
        obj_coords = np.where(ar_row_view==elem)
        #obj_coords = np.where(obj_masks==elem)
        obj_points = points[obj_coords[0]]
        obj_height = np.max(obj_points[:,1])
        obj_points = np.delete(obj_points, 1, 1)
        obj_points = np.int_((obj_points+displacement)/grid_size)
        obj_points_str = obj_points.view('|S%d' % (obj_points.itemsize * obj_points.shape[1]))
        _,unique_indices,unique_counts = np.unique(obj_points_str,return_index=True,return_counts=True)
        object_occupancy_grids_row_view[elem] = (obj_points[unique_indices],obj_height)
        np_occ_map[obj_points[unique_indices][:,0],obj_points[unique_indices][:,1]] = unique_counts[:]
        np_occ_map = np.where(np_occ_map>=3, 1, 0)
        occupancy_map = merge_occupancy_map(occupancy_map, np_occ_map)

    all_polygons = occupancy_to_polygons(occupancy_map, grid_size, displacement  )
    simplified_polygon = polygon_simplify(all_polygons,0.08)
    show_animation =  False
    #plt.close()
    if show_animation :
        plt.cla()
        #patch1 = PolygonPatch(all_polygons,fc='grey', ec="black", alpha=0.2, zorder=1)
        patch1 = PolygonPatch(simplified_polygon,fc='grey', ec="black", alpha=0.2, zorder=1)
        plt.gca().add_patch(patch1)
        plt.axis("equal")
        plt.pause(0.01)
    #print ("time taken for point cloud to polygon part", time.time()-start_time)
    return occupancy_map, simplified_polygon,object_occupancy_grids_row_view

def polygon_simplify(all_polygons,scale=0.0):
    
    simplified_polygon = []
    if all_polygons.geom_type == 'Polygon':
        simplified_polygon = all_polygons.simplify(scale)
         
    elif all_polygons.geom_type == 'MultiPolygon':
        for i,polygon in enumerate(all_polygons) :
            simplified_polygon.append(all_polygons[i].simplify(scale))
        simplified_polygon = MultiPolygon(simplified_polygon)

    return simplified_polygon


def update_goal_bounding_box(points, env):
    new_points = np.where((points[:,1] > 0.05) & (points[:,1] <= 3))#,1,0).reshape(points.shape[0],1)
    points = points[new_points[0]]
    env.trophy_mask = env.trophy_mask[new_points[0]]
    #print ("in update goal bounding box")
    trophy_pixel_coords = np.where(env.trophy_mask >= 0.8)
    #print (np.amax(env.trophy_mask), np.amin(env.trophy_mask))
    #print ("number of pixels with pts greater than 0.7", len(trophy_pixel_coords[0]))
    trophy_points = points[trophy_pixel_coords[0]]
    #print (env.trophy_mask)
    trophy_points = np.delete(trophy_points,1,1)
    trophy_points = np.int_((trophy_points+env.displacement)/env.grid_size)
    #print ("size of trophy points ", trophy_points.shape)
    trophy_pts_str = trophy_points.view('|S%d' % (trophy_points.itemsize * trophy_points.shape[1]))
    #obj_points_str = obj_points.view('|S%d' % (obj_points.itemsize * obj_points.shape[1]))
    _,unique_indices,unique_counts = np.unique(trophy_pts_str,return_index=True,return_counts=True)
    env.trophy_occupancy_map_points = trophy_points[unique_indices]
