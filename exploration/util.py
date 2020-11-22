import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
from scipy.spatial.transform import Rotation

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
    global_pts = np.matmul(local_pts, rot.as_dcm()) + pos
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


# filtered out floor and ceil, maybe sample
def pre_process(pts, floor_threshold=0.01, ceil_threshold=3):
    pts = pts.reshape(-1, 3)
    select = pts[:,1] > floor_threshold
    pts = pts[select]
    select = pts[:,1] < ceil_threshold
    pts = pts[select]
    plt.scatter(pts[:, 0], pts[:, 2], s=0.02)
    plt.pause(0.01)
    return pts[:,[0,2]]


def dump_scene(agent_x, agent_z, scene_obstacles):
    debug_dict = {'obstacles': [], 'agent_xz': (agent_x, agent_z)}
    for obstacle in scene_obstacles:
        obstacle.plot("green")
        debug_dict['obstacles'].append({'x_list': obstacle.x_list, 'y_list': obstacle.y_list})

    pickle.dump(debug_dict, open("polygons.pkl", 'wb'))
    with open("polygons.pkl", 'rb') as f:
        reload = pickle.load(f)


def degree_to_radian(x):
    return x / 180 * math.pi