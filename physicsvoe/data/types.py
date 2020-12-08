from collections import namedtuple

ThorFrame = namedtuple('ThorFrame', ('obj_data', 'struct_obj_data', 'depth_mask', 'obj_mask', 'camera'))
CameraInfo = namedtuple('CameraInfo', ('aspect_ratio', 'fov', 'position', 'rotation', 'tilt'))

DEFAULT_CAMERA = {'vfov': 42.5, 'pos': [0, 1.5, -4.5]}

def make_camera(o):
    xyz_to_list = lambda x: [x['x'], x['y'], x['z']]
    pos = xyz_to_list(o.position) if o.position is not None else DEFAULT_CAMERA['pos']
    rot = o.rotation if o.rotation is not None else 0
    camera_desc = CameraInfo(o.camera_aspect_ratio, o.camera_field_of_view,
                   pos, rot, o.head_tilt)
    return camera_desc
