import numpy as np
from scipy.spatial.transform import Rotation

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Image -> Point cloud conversion
"""

def convert_scenes(env, paths):
    for scene_idx, scene_path in enumerate(paths):
        for frame_idx, obs in enumerate(env.run_scene(scene_path)):
            pts, obj_mask = convert_output(obs)
            plot_scene(scene_idx, frame_idx, pts, obj_mask)


def convert_output(o):
    objs = o.object_list
    structs = o.structural_object_list
    img = o.image_list[-1]
    obj_mask = convert_obj_mask(o.object_mask_list[-1], objs).flatten()
    depth_map = np.array(o.depth_map_list[-1])
    camera_desc = [o.camera_field_of_view, o.position, o.rotation, o.head_tilt]
    pts = depth_to_points(depth_map, *camera_desc)
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


def depth_to_points(depth, camera_field_of_view, pos_dict, rotation, tilt):
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
    local_pts = depth_to_local(depth, camera_field_of_view)
    # Convert to world space
    # Use rotation & tilt to calculate rotation matrix.
    rot = Rotation.from_euler('yx', (rotation, tilt), degrees=True)
    pos_to_list = lambda x: [x['x'], x['y'], x['z']]
    pos = pos_to_list(pos_dict)
    # Apply rotation, offset by camera position to get global coords
    global_pts = np.matmul(local_pts, rot.as_matrix()) + pos
    # Flatten to a list of points
    flat_list_pts = global_pts.reshape(-1, global_pts.shape[-1])
    return flat_list_pts


def depth_to_local(depth, fov_deg):
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
    idx_grid = np.meshgrid(*[np.arange(ar) for ar in aspect_ratio])
    px_arr = np.stack(idx_grid, axis=-1) # Each pixel's index
    uv_arr = px_arr*[2/w for w in aspect_ratio]-1
    uv_arr[:, :, 1] *= -1 # Each pixel's UV coords
    """ Determine vertical & horizontal FOV in radians.
    Use the UV coordinate values and tan(fov/2) to determine the 'XY' direction
    vector for each pixel.
    """
    vfov = np.radians(fov_deg)
    hfov = 2*math.atan(math.tan(vfov/2) * aspect_ratio[0]/aspect_ratio[1])
    tans = np.array([np.tan(fov/2) for fov in (hfov, vfov)])
    px_dir_vec = uv_arr * tans
    """ Add Z coordinate and scale to the pixel's known depth.  """
    const_zs = np.ones((px_dir_vec.shape[0:2])+(1,))
    px_dir_vec = np.concatenate((px_dir_vec, const_zs), axis=-1)
    camera_offsets = px_dir_vec * np.expand_dims(depth, axis=-1)
    return camera_offsets


"""
Plotting
"""

def plot_pts(ax, pts):
    ax.scatter(*zip(*pts), zdir='y', s=0.1, alpha=0.5)

def plot_scene(scene_idx, frame_idx, pts, obj_mask):
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for obj_id in np.unique(obj_mask):
        print(obj_id)
        obj_pts = pts[obj_mask==obj_id]
        plot_pts(ax, obj_pts)
    bound = 8
    ax.set_xlim([-bound/2, bound/2])
    ax.set_ylim([0, bound])
    ax.set_zlim([0, bound])
    fig.savefig(f'{scene_idx}_{frame_idx}_points.png')
    plt.close(fig)
