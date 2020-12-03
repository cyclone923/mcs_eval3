from collections import namedtuple

ThorFrame = namedtuple('ThorFrame', ('obj_data', 'struct_obj_data', 'depth_mask', 'obj_mask', 'camera'))
CameraInfo = namedtuple('CameraInfo', ('aspect_ratio', 'fov', 'position', 'rotation', 'tilt'))
