import pdb
from textwrap import fill
import cv2
import numpy as np
import machine_common_sense as mcs
from dataclasses import dataclass
from typing import List


OBJ_KINDS = {
    "default",
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
    "default"
    "pole",
    "target",
    "support",
    "floor",
    "back-wall"
}

TUBE_WIDTH2HEIGHT_RATIO = 2.5

@dataclass
class Coord:

    x: int
    y: int
    z: int

    def transform(self, dw, dh, dd):
        return Coord(
            self.x + dw,
            self.y + dh,
            self.z + dd
        )

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z
        }


@dataclass
class Object:

    rgb_im: np.ndarray  # Isomeric view
    obj_mask: np.ndarray
    depth_map: np.ndarray
    front_view: np.ndarray
    role: str = "default"
    kind: str = "default"

    def __post_init__(self):
        self.extract_physical_props()
        self.find_obj_kind()

    # @staticmethod
    def _get_bounding_rect(self, img, view="front") -> tuple:
        
        # TODO: Shouldn't have to remove lines at this stage. VERY RISKY. 
        # Inspect the root cause
        # Remove border pixels
        # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        # contours, _ = L2DataPacket._apply_good_contours(img)
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sometimes, side view has more than one CC. Choose the biggest
        if view == "side":
            max_cnt_idx = np.argmax(cv2.contourArea(cnt) for cnt in contours)
            
            return cv2.boundingRect(contours[max_cnt_idx])

        # That said, most other cases should have a single CC
        try:
            assert len(contours) == 1, "Object FV has more than one connected components."
        except AssertionError as e:
            total_area = sum(cv2.contourArea(cnt) for cnt in contours)
            area_ratio = total_area / (600 * 400) # TODO: don't hardcode aspect ratio

            # Check if this is back-wall
            if len(contours) == 2 and area_ratio > 0.5:
                combined_contour = np.concatenate(contours, axis=0)
                return cv2.boundingRect(combined_contour)
            # Check if this is a floor
            elif len(contours) == 2 and area_ratio > 0.25:
                max_area_cnt_idx = np.argmax([cv2.contourArea(cnt) for cnt in contours])
                return cv2.boundingRect(contours[max_area_cnt_idx])
            elif len(contours) == 0: # Most likely, object didn't fully showup
                raise(NotImplemented("Deal with objects that are just entering the scene."))
            else:
                cv2.imwrite("ae.jpg", img)
                pdb.set_trace()
                raise(e)

        return cv2.boundingRect(contours[0])

    @staticmethod
    def _count_corner_points(img) -> int:
        dst = cv2.cornerHarris(img, 2, 3, 0.04)
        dst = cv2.dilate(((dst > 0) * 255).astype("uint8"), np.ones((5,5), np.uint8))
        num_cc, _ = cv2.connectedComponents(((dst > 0) * 255).astype("uint8"))

        return num_cc - 1  # Includes background component

    @staticmethod
    def _color_prop(obj_mask, rgb_im):
        '''
        Finds average RGB channels pixel values. Doesn't make sense for 
        objects with a texture or wide color gradient
        '''
        mask_3d = np.squeeze(np.stack([obj_mask] * 3, axis=2))
        masked_rgb = np.ma.masked_where(mask_3d, rgb_im)
        # Variation for wall & floor objects is a lot. 
        # So color attr doesn't make sense of these roles
        return masked_rgb.mean((0, 1)).astype(np.int).data.tolist()

    def _dims_prop(self, obj_mask, front_view):
        '''
        Uses isometric & front view to derive 3D coordinates of an object
        Returns 8 point bounding cube around the object.
        Limitations: We can't see backside of some objects and 
                     may endup approximating them into known kinds.
                     We don't have conversion factor for camera's far clipping plane (15 units)
        Assumptions: Widest dimension will be on the front side. 
        '''
        obj_fv = self.obj_mask & self.front_view

        if obj_fv.sum() == 0:  # No intersection = "floor" role
            obj_mask = (np.squeeze(obj_mask) * 255).astype("uint8")
            x, z, w, d = self._get_bounding_rect(obj_mask)
            h = 1  # Height of floor is assumed to be 2
            # Offsetting y with h
            y = 400  # TODO: pixel to m transfer function based on floor placement
            x, z = 0, 0  # For floor, top corner is origin
            dims = [
                Coord(x, y + h, z),
                Coord(x, y - h, z),
                Coord(x + w, y + h, z),
                Coord(x + w, y - h, z),
                Coord(x + w, y + h, z + d),
                Coord(x + w, y - h, z + d),
                Coord(x, y + h, z + d),
                Coord(x, y - h, z + d),
            ]

        else:
            obj_fv = (np.squeeze(obj_fv) * 255).astype("uint8")
            x, y, w, h = self._get_bounding_rect(obj_fv)

            obj_sv = ((obj_mask & np.bitwise_not(front_view)) * 255).astype("uint8")
            _, _, d, _ = self._get_bounding_rect(obj_sv, view="side")
            
            # Offsetting z with k + d, TODO: don't hardcode max-depth
            k = 15.0 - (np.squeeze(obj_mask) * self.depth_map).max()
            z = k
            dims = [
                Coord(x, y, z),
                Coord(x, y, z + d),
                Coord(x + w, y, z),
                Coord(x + w, y, z + d),
                Coord(x + w, y + h, z),
                Coord(x + w, y + h, z + d),
                Coord(x, y + h, z),
                Coord(x, y + h, z + d),
            ]

        # TODO: visualize it
        self.dims = [
            dim.to_dict() for dim in dims
        ]

        self.w_h_d = (w, h, d)

    def extract_physical_props(self):
        '''
        Uses blob to determine `color` & `dims`
        '''
        self.color = self._color_prop(self.obj_mask, self.rgb_im)

        # Sets dims, w_h_d
        self._dims_prop(self.obj_mask, self.front_view)

        x = sum(pt["x"] for pt in self.dims) / 8
        y = sum(pt["y"] for pt in self.dims) / 8
        z = sum(pt["z"] for pt in self.dims) / 8
        self.centroid = (x, y, z)

    def find_obj_kind(self) -> None:
        '''
        Takes a binary blob and returns one among OBJ_KINDS: sets `kind`
        '''

        obj_fv = (np.squeeze(self.front_view * self.obj_mask) * 255).astype("uint8")
        if obj_fv.sum() == 0:
            self.kind = "tube_wide"
            return

        x, y, w, h = self._get_bounding_rect(obj_fv)
        n = self._count_corner_points(obj_fv)

        self.n_corners = n
        self.fill_ratio = (obj_fv > 0).sum() / (w * h)
        self.is_symmetic = w == h

        # TODO: include tolerance
        if self.fill_ratio == 1.0 and n == 4 and self.is_symmetic:
            self.kind = "cube"  # Can miss classify pyramids
        elif self.fill_ratio > 0.8 and n == 4:
            self.kind = "circle_frustum"  # Validate ratio, try to use area + prob
        elif self.fill_ratio > 0.8 and n == 3:
            self.kind = "cone"
        elif self.fill_ratio == 1.0 and n == 4 and not self.is_symmetic:  # Could be cylinder
            if h > w:
                self.kind = "tube_narrow"
        elif self.fill_ratio > 0.7 and n == 3:
            self.kind = "pyramid" # Or triangle
        elif self.fill_ratio > 0.75 and n > 10:
            self.kind = "sphere"
        elif self.fill_ratio < 0.5 and n == 6:
            self.kind = "letter_l_narrow"
        elif self.fill_ratio < 0.5 and n > 6:
            self.kind = "letter_x"
        else:
            self.kind = "default"
            # raise(Exception("Couldn't guess object kind"))
        

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
        
        # For debugging
        cv2.imwrite("rgb.jpg", self.rgb_im)
        cv2.imwrite("mask.jpg", self.obj_mask)
        cv2.imwrite("depth.jpg", self.depth_map * 255. / self.depth_map.max())

    @staticmethod
    def _apply_good_contours(img, min_area=20, min_width=5, min_height=5) -> tuple:
        '''
        Finds and applies valid contours to input image
        '''
        contours, _ = cv2.findContours(
            img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        filtered_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            _, _, w, h = cv2.boundingRect(cnt)

            if all((area > min_area, w > min_width, h > min_height)):
                filtered_contours.append(cnt)

        # Denoise image
        updated_img = cv2.drawContours(img.copy(), filtered_contours, -1, 255, -1)
        # print(f"[x] Removed {len(contours) - len(filtered_contours)} invalid contours.")

        return filtered_contours, updated_img

    def _estimate_obj_mask_front_view(self) -> np.ndarray:
        '''
        Uses depth_map to return front view of objects in obj_mask
        Excludes border pixels of object from front view. So a correction term may be necessary.
        '''
        dx, dy = np.gradient(self.depth_map)
        dx, dy = np.abs(dx), np.abs(dy)
        df = ((dx > 0) | (dy > 0))

        # Remove back wall
        wall_mask = self.depth_map == self.depth_map.max()
        obj_front_view = np.bitwise_not(df) ^ wall_mask

        # Manually add pole kind of objects or don't
        (dy > 0) & (dy < dy.max()) & (dx == 0)
        if self.pole_mask is not None:
            obj_front_view *= self.pole_mask

        # Remove lines and border pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3))
        obj_front_view = cv2.morphologyEx(
            (obj_front_view * 255).astype("uint8"), cv2.MORPH_OPEN, kernel
        )
        _, obj_front_view = self._apply_good_contours(obj_front_view)

        cv2.imwrite("front_view.jpg", obj_front_view)

        return np.expand_dims(obj_front_view > 0, axis=2)

    def _get_obj_masks(self) -> List[np.ndarray]:
        '''
        Returns 1D masks of each object in the frame
        '''
        # transform 3 channels to 1 channel
        obj_mask = np.prod(self.obj_mask, axis=2)
        uniq_obj_colors = np.unique(obj_mask) 

        obj_masks = []
        for color in uniq_obj_colors:
            mask = obj_mask == color
            if mask.sum() > 500:
                obj_masks.append(np.expand_dims(mask, axis=2))
                w, h = mask.sum(axis=1).max(), mask.sum(axis=0).max()
                if w * TUBE_WIDTH2HEIGHT_RATIO < h: 
                    self.pole_mask = mask  # Desperate side-effect 
                    print(f"[x] Chosen pole mask's w,h = {w},{h}")

        if not hasattr(self, "pole_mask"):
            self.pole_mask = None

        return obj_masks


    def segment_objects(self) -> List[Object]:
        '''
        Takes front view of objects and returns object kind, dimensions & color for <Object>.
        Technically "z" dimension can't be inferred from front view.
        '''
        obj_masks = self._get_obj_masks()
        # FV depends implicitly depends on _get_obj_masks (TODO)
        front_view = self._estimate_obj_mask_front_view()
        
        objects = []
        for obj_mask in obj_masks:

            ob = Object(
                rgb_im=self.rgb_im,
                obj_mask=obj_mask,  # This particular object mask
                depth_map=self.depth_map,
                front_view=front_view  # Front view of all objects
            )
            objects.append(ob)

        return objects

    def determine_obj_roles(self) -> None:
        '''
        Heuristics to collectively assign roles to all objects 
        (one each among OBJ_ROLES)
        '''

        # Find floor
        for idx, obj in enumerate(self.objects):
            # TODO: don't hard code, read from cfg instead
            if 600 in obj.w_h_d:  # Either back-wall or floor
                obj_depth_mask = np.squeeze(obj.obj_mask) * obj.depth_map
                depth_var = np.ma.masked_where(obj_depth_mask == 0, obj_depth_mask).var()

                if depth_var < 1e-2:
                    obj.role = "back-wall"
                    self.floor = obj
                else:
                    obj.role = "floor"
                    self.floor = obj
        try:
            self.floor
        except AttributeError:
            raise(Exception("Floor wasn't detected!"))

        # Find support
        min_dist = float("inf")
        for idx, obj in enumerate(self.objects):

            if obj.role != "default":
                continue
            
            this_dist = min([abs(c["y"] - self.floor.centroid[1]) for c in obj.dims])
            # Ideally this_dist should be 0 for a contact point
            # But due to scaling issues while creating dims, this doesn't hold
            if this_dist < min_dist: 
                min_dist = this_dist
                self.support = obj

        if min_dist == float("inf"):
            raise(Exception("Support wasn't detected!"))
    
        self.support.role = "support"

        # Find Pole
        for idx, obj in enumerate(self.objects):
            if obj.role != "default":
                continue

            w, h, d = obj.w_h_d
            # TODO: don't hardcode aspect ratio 
            if 2 * w < h and obj.centroid[1] < 0.4 * 400:
                obj.role = "pole"
                self.pole = obj
                break
        else:
            self.pole = None
            print("Pole not in scene yet!")
            
        # Find target
        try:
            assert sum(obj.role == "default" for obj in self.objects) <= 1, "Object count in scene violation!"
        except:
            print([o.role for o in self.objects])
            print([o.centroid for o in self.objects])
            print([o.w_h_d for o in self.objects])
            print("==========DEBUG==========")
        for obj in self.objects:
            if obj.role == "default":
                obj.role = "target"
                self.target = obj
                break
        else:
            self.target = None
            print("Target not in scene yet!")


        
        

        
            



