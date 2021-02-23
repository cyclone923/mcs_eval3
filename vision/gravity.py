import cv2
import numpy as np
import machine_common_sense as mcs
from dataclasses import dataclass

OBJ_KINDS = {
    "circle_frustum",
    "cone",
    "cube",
    "cylinder",
    "pyramid",
    "sphere",
    "square_frustum",
    "triangle",
    "tube_narrow",
    "tube_wide",
    "cube_hollow_narrow",
    "cube_hollow_wide",
    "hash",
    "letter_l_narrow",
    "letter_l_wide",
    "letter_x",
}

OBJ_ROLES = {
    "pole",
    "target",
    "support",
    "floor"
}

def clean_object_mask(obj_mask, depth_mask):
    '''
    Uses depth_mask to return front view of objects in obj_mask
    '''
    pass


def extract_object_trajectories(temporal_rgbd_data):
    '''
    First segments objects, then classifies object kind, 
    tracks and returns trajectories
    '''
    pass

@dataclass
class Object:

    blob: np.ndarray
    role: str = "floor"

    def __post_init__(self):

        self.extract_physical_props()
        self.find_obj_kind()
        
        assert self.kind in OBJ_KINDS, f"Object creation failed! Unrecognised kind: {self.kind}"
        assert self.role in OBJ_ROLES, f"Object creation failed! Unrecognised role: {self.role}"

    def extract_physical_props(self):
        '''
        Uses blob to determine `color` & `dims`
        '''
        pass

    def find_obj_kind(self) -> str:
        '''
        Takes a binary blob and returns one among OBJ_KINDS: sets `kind`
        '''
        pass

@dataclass
class L2DataPacket:

    step_number: int
    step_meta: mcs.StepMetadata

    def __post_init__(self):

        self.get_images_from_meta()
        self.objects = self.segment_objects()
        self.determine_obj_roles()

    def get_images_from_meta(self):

        self.rgb_im = self.step_meta.image_list[0]
        self.rgb_im = np.array(self.rgb_im)[:,:,::-1] # BGR -> RGB

        self.obj_mask = self.step_meta.object_mask_list[0]
        self.obj_mask = np.array(self.obj_mask)[:,:,::-1]

        self.depth_map = self.step_meta.depth_map_list[0]
        self.depth_map *= 255. / self.depth_map.max()
        
        # For debugging
        cv2.imwrite("rgb.jpg", self.rgb_im)
        cv2.imwrite("mask.jpg", self.obj_mask)
        cv2.imwrite("depth.jpg", self.depth_map)

    def _estimate_obj_mask_front_view(self) -> np.ndarray:
        '''
        Uses depth_map to return front view of objects in obj_mask
        '''
        pass

    def segment_objects(self) -> Object:
        '''
        Takes front view of objects and returns object kind, dimensions & color for <Object>.
        Technically "z" dimension can't be inferred from front view.
        '''
        pass

    def determine_obj_roles(self) -> None:
        '''
        Heuristics to collectively assign roles to all objects 
        (one each among OBJ_ROLES)
        '''
        pass



