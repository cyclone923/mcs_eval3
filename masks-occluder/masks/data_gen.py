from .mcs_env import McsEnv
from .types import ThorFrame, CameraInfo

import pickle
import random
import itertools
import gzip
import numpy as np
from pathlib import Path
from argparse import ArgumentParser


def convert_scenes(env, paths):
    for scene_path in paths:
        out_path = scene_path.with_suffix('.pkl.gz')
        if out_path.exists():
            print(f'{out_path} exists, skipping')
            continue
        print(f'Converting {scene_path} -> {out_path}')
        scene_output = [convert_frame(o, i) for i, o in enumerate(env.run_scene(scene_path))]
        with gzip.open(out_path, 'wb') as fd:
            pickle.dump(scene_output, fd)


def convert_frame(o, i):
    objs = o.object_list
    structs = o.structural_object_list
    img = o.image_list[-1]
    obj_mask = convert_obj_mask(o.object_mask_list[-1], objs)
    struct_mask = convert_obj_mask(o.object_mask_list[-1], structs)
    depth_mask = np.array(o.depth_map_list[-1])
    camera_desc = CameraInfo(o.camera_field_of_view, o.position, o.rotation, o.head_tilt)
    # Project depth map to a 3D point cloud - removed for performance
    # depth_pts = depth_to_points(depth_mask, *camera_desc)
    return ThorFrame(objs, structs, img, depth_mask, obj_mask, struct_mask, camera_desc)


def convert_obj_mask(mask, objs):
    convert_color = lambda col: (col['r'], col['g'], col['b'])
    color_map = {convert_color(o.color):i for i, o in enumerate(objs)}
    arr_mask = np.array(mask)
    out_mask = -np.ones(arr_mask.shape[0:2], dtype=np.int8)
    for x in range(arr_mask.shape[0]):
        for y in range(arr_mask.shape[1]):
            idx = color_map.get(tuple(arr_mask[x, y]), -1)
            out_mask[x, y] = idx
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
    local_pts = depth_to_local(depth, camera_clipping_planes, camera_field_of_view)
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
    idx_grid = np.meshgrid(*[np.arange(ar) for ar in aspect_ratio])
    px_arr = np.stack(idx_grid, axis=-1) # Each pixel's index
    uv_arr = px_arr*[2/w for w in aspect_ratio]-1
    uv_arr[:, :, 1] *= -1 # Each pixel's UV coords
    """ Determine vertical & horizontal FOV in radians.
    Use the UV coordinate values and tan(fov/2) to determine the 'XY' direction
    vector for each pixel.
    """
    vfov = np.radians(fov_deg)
    hfov = np.radians(fov_deg*aspect_ratio[0]/aspect_ratio[1])
    tans = np.array([np.tan(fov/2) for fov in (hfov, vfov)])
    px_dir_vec = uv_arr * tans
    """ Add Z coordinate and scale to the pixel's known depth.  """
    const_zs = np.ones((px_dir_vec.shape[0:2])+(1,))
    px_dir_vec = np.concatenate((px_dir_vec, const_zs), axis=-1)
    camera_offsets = px_dir_vec * np.expand_dims(depth, axis=-1)
    return camera_offsets


def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--sim', type=Path, default=Path('data/thor'))
    parser.add_argument('--scenes', type=Path, default=Path('data/thor/scenes'))
    parser.add_argument('--filter', type=str, default=None)
    return parser


def main(sim_path, data_path, filter):
    env = McsEnv(sim_path, data_path, filter)
    scenes = list(env.all_scenes)
    print(f'Found {len(scenes)} scenes')
    random.shuffle(scenes)
    # Work around stupid sim bug
    Path('SCENE_HISTORY/evaluation3Training').mkdir(parents=True)
    convert_scenes(env, scenes)


if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args.sim, args.scenes, args.filter)

