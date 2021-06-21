from collections import namedtuple

ThorFrame = namedtuple('ThorFrame', ('obj_data', 'struct_obj_data', 'image', 'depth_mask', 'obj_mask', 'struct_mask', 'camera_info'))
CameraInfo = namedtuple('CameraInfo', ('fov', 'position', 'rotation', 'tilt'))

