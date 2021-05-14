import pdb
import cv2
import math
import open3d as o3d
import numpy as np
import machine_common_sense as mcs
from dataclasses import dataclass
from typing import List
from vision.obj_kind import KindClassifier

OBJ_KINDS = {
    # Flat surfaces
    "cube",
    "letter_l_narrow",
    "letter_l_wide",
    "triangle",  # Always assumed to be on side view

    # Slant surfaces
    "pyramid",
    "square_frustum",
    
    # Curved surfaces
    "circle_frustum",
    "cone",
    "cylinder",
}

OBJ_ROLES = {
    "pole",
    "target",
    "support",
    "floor",
    "back-wall"
}

TUBE_WIDTH2HEIGHT_RATIO = 2.5
MAX_Y = 400
MAX_X = 600
CAM_CLIP_PLANES = [0.01, 15.0]
CAM_ASPECT = [600, 400]
CAM_FOV = 42.5
CAM_HEIGHT = 1.5
FOCAL = 30.85795
OBJ_KIND_MODEL_NAME = "model.p"


@dataclass
class Coord:

    x: int
    y: int
    z: int

    @staticmethod
    def _pixel_to_physical(x, y, z):
        '''
        Credits: Z
        '''
        cam_matrix = [
            [FOCAL, 0, CAM_ASPECT[0] / 2],
            [0, FOCAL, CAM_ASPECT[1] / 2],
            [0, 0, 1],
        ]
        inv_cam_matrix = np.linalg.inv(cam_matrix)
        
        return np.dot(inv_cam_matrix, [x, y, z]).tolist()

    def transform(self, scaled=False):
        if scaled:
            x, y, z = self._pixel_to_physical(self.x, self.y, self.z)
        else:
            x, y, z = self.x, self.y, self.z

        return {"x": x, "y": y, "z": z}


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
            dim.transform() for dim in dims
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
        cv2.imwrite("mask.png", self.obj_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
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


@dataclass
class ObjectV2:
    rgb_im: np.ndarray  # Isomeric view
    obj_mask: np.ndarray
    depth_map: np.ndarray

    role: str = "default"
    kind: str = "default"

    dims: tuple = None
    w_h_d: tuple = None
    centroid: tuple = None
    centroid_px: tuple = None

    def __post_init__(self):
        self.front_view = self._estimate_obj_mask_front_view()

    @staticmethod
    def _color_prop(obj_mask, rgb_im):
        '''
        Finds average RGB channels pixel values. Doesn't make sense for 
        objects with a texture or wide color gradient
        '''
        cropped_rgb = obj_mask * rgb_im
        masked_rgb = np.ma.masked_equal(cropped_rgb, 0)
        # Variation for wall & floor objects is a lot. 
        # So color attr doesn't make sense of these roles
        cv2.imwrite("pole.png", masked_rgb)
        return masked_rgb.mean((0, 1)).astype(np.int).data.tolist()

    @staticmethod
    def _get_obj_dims(obj_mask):
        obj_mask = np.squeeze(obj_mask)
        w = obj_mask.sum(axis=1).max()
        h = obj_mask.sum(axis=0).max()

        return w, h

    @staticmethod
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

    def _dims_prop(self, obj, visualize=False):
        '''
        Uses isometric view to approximate 3D coordinates of an object
        Returns 8 point bounding cube around the object.
        Limitations: Width & height will be slightly off and depth will be guessed
        Assumptions: Depth is assumed to be min{width, height}
        '''
        scene_points = self.depth_to_local(
            depth=obj.depth_map, clip_planes=CAM_CLIP_PLANES, fov_deg=CAM_FOV
        )
        scene_points = scene_points.reshape(-1, 3)

        scene_point_cloud = o3d.geometry.PointCloud()
        scene_point_cloud.points = o3d.utility.Vector3dVector(scene_points)
        
        obj_idx = np.nonzero(obj.obj_mask.reshape(-1))[0]
        obj_point_cloud = scene_point_cloud.select_down_sample(obj_idx)
        self.obj_cloud = obj_point_cloud

        assert obj_point_cloud.dimension() == 3, "RGB-D couldn't return a 3D object!"
        
        if visualize:
            o3d.visualization.draw_geometries([obj_point_cloud])

        bbox = obj_point_cloud.get_axis_aligned_bounding_box()
        bbox_corners = np.asarray(bbox.get_box_points())

        dy = 1.75
        bbox_corners = np.asarray(
            bbox.translate([0, dy, 0], relative=True)
                .get_box_points()
            )
        self.dims = [
            {"x": pt[0], "y": pt[1], "z": pt[2]}
            for pt in bbox_corners
        ]
        self.w_h_d = np.abs(bbox.get_extent()).tolist()

    def extract_physical_props(self):
        '''
        Uses blob to determine `color` & `dims`
        '''
        self.color = self._color_prop(self.obj_mask, self.rgb_im)

        self._dims_prop(self)

        x = sum(pt["x"] for pt in self.dims) / 8
        y = sum(pt["y"] for pt in self.dims) / 8
        z = sum(pt["z"] for pt in self.dims) / 8
        self.centroid = (x, y, z)

        self.centroid_px = L2DataPacketV2._get_obj_moments(self.obj_mask)

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

    def _get_fv_dims(self, img, safe=True) -> tuple:

        contours, img = self._apply_good_contours(img)
        # There will always be no more than one object
        
        try:
            combined_contour = np.concatenate(contours, axis=0)
            _, _, w, h = cv2.boundingRect(combined_contour)
        except Exception as e:
            if not safe:
                raise(e)
            w, h = self._get_obj_dims(self.obj_mask)

        return w, h

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
        # if self.pole_mask is not None:
        #     obj_front_view *= self.pole_mask

        # Remove lines and border pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3))
        obj_front_view = cv2.morphologyEx(
            (obj_front_view * 255).astype("uint8"), cv2.MORPH_OPEN, kernel
        )
        _, obj_front_view = self._apply_good_contours(obj_front_view)

        cv2.imwrite("front_view.jpg", obj_front_view)

        return np.expand_dims(obj_front_view > 0, axis=2)

    @staticmethod
    def _count_corner_points(img) -> int:
        dst = cv2.cornerHarris(img, 2, 3, 0.04)
        dst = cv2.dilate(((dst > 0) * 255).astype("uint8"), np.ones((5,5), np.uint8))
        num_cc, _ = cv2.connectedComponents(((dst > 0) * 255).astype("uint8"))

        return num_cc - 1  # Includes background component

    @staticmethod
    def _face_curve_classifier(dmap, obj_mask):

        dx, dy = np.gradient(dmap)
        dx, dy = np.abs(dx), np.abs(dy)
        slant_mask = (dy > 0) & (dy < dy.max()) & (dx == 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3))
        slant_mask = cv2.morphologyEx(
            (slant_mask * 255).astype("uint8"), cv2.MORPH_CLOSE, kernel
        )

        obj_mask = np.squeeze(obj_mask)
        intersection = (slant_mask & obj_mask).sum() / obj_mask.sum()

        if intersection > 0.5:
            return "slant"
        elif intersection < 0.01:
            return "flat"
        else:
            return "curved"

    def find_obj_kind(self) -> None:
        '''
        Takes a binary blob and returns one among OBJ_KINDS: sets `kind`
        '''
        obj_fv = (np.squeeze(self.front_view * self.obj_mask) * 255).astype("uint8")

        w, h = self._get_fv_dims(obj_fv, safe=True)
        n = self._count_corner_points(obj_fv)

        face_type = self._face_curve_classifier(self.depth_map, self.obj_mask)

        self.n_corners = n
        self.fill_ratio = (obj_fv > 0).sum() / (w * h)
        self.is_symmetic = w == h
        self.has_slant_face = face_type == "slant"
        self.has_flat_face = face_type == "flat"
        self.has_curved_face = face_type == "curved"

        if self.role in ["pole", "floor", "support", "back-wall"]:
            self.kind = "cube"
        else:
            pdb.set_trace()
            # TODO: include tolerance
            if self.has_flat_face and self.fill_ratio < 0.5 and n == 6:
                self.kind = "letter_l_narrow"
            elif self.has_flat_face and self.fill_ratio > 0.5 and n == 6:
                self.kind = "letter_l_wide"
            elif self.has_flat_face and n == 3:
                self.kind = "triangle"
            elif self.has_flat_face and n == 4 and self.fill_ratio > 0.9:
                self.kind = "cube"
            elif self.has_slant_face and n == 4 and self.fill_ratio < 0.9:
                self.kind = "square_frustum"
            elif self.has_slant_face:
                self.kind = "pyramid"
            elif self.has_curved_face and n == 3 and not self.is_symmetic:
                self.kind = "cone"
            elif self.has_curved_face and self.fill_ratio > 0.9:
                self.kind = "cylinder"
            elif self.has_curved_face:
                self.kind = "circle_frustum"
            else:
                self.kind = "cube"

    def find_obj_kind_nn(self) -> None:

        if self.role != "target":
            self.kind = "cube"
            return
        
        x, y, w, h = cv2.boundingRect(
            cv2.findContours(
                (self.obj_mask * 255).astype("uint8"),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
                )[0].pop()
            )

        rgb_object = self.rgb_im[y : y + h, x: x + w, :]
        depth_object = self.depth_map[y : y + h, x: x + w]
        kind_pred, conf = KindClassifier(model_name=OBJ_KIND_MODEL_NAME).run(
            rgb_object, depth_object
        )

        self.kind = kind_pred
        
@dataclass
class L2DataPacketV2:

    step_number: int
    step_meta: mcs.StepMetadata

    def __post_init__(self):

        self.get_images_from_meta()
        self.objects = self.segment_objects()
        self.determine_obj_roles()
        self.calculate_physical_props()
        self.guess_object_kinds()
        self.load_roles_as_attr()

    def get_images_from_meta(self):

        self.rgb_im = self.step_meta.image_list[0]
        self.rgb_im = np.array(self.rgb_im)[:,:,::-1] # BGR -> RGB

        self.obj_mask = self.step_meta.object_mask_list[0]
        self.obj_mask = np.array(self.obj_mask)[:,:,::-1]

        self.depth_map = self.step_meta.depth_map_list[0]
        
        # For debugging
        cv2.imwrite("rgb.png", self.rgb_im, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite("mask.png", self.obj_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite("depth.png", self.depth_map, [cv2.IMWRITE_PNG_COMPRESSION, 0])

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
            obj_masks.append(np.expand_dims(mask, axis=2))

        return obj_masks

    @staticmethod
    def _get_obj_moments(obj_mask):
        obj_mask = (np.squeeze(obj_mask) * 255).astype("uint8")

        MM = cv2.moments(obj_mask)
        cX = int(MM["m10"] / MM["m00"])
        cY = int(MM["m01"] / MM["m00"])

        return cX, cY

    def segment_objects(self):
        objects = []
        obj_masks = self._get_obj_masks()

        # Determining floor & wall
        floor_found, wall_found = False, False
        for obj in obj_masks:
            this_ob = ObjectV2(
                rgb_im=self.rgb_im,
                obj_mask=obj,
                depth_map=self.depth_map
            )

            objects.append(this_ob)

        return objects

    def determine_obj_roles(self):

        assert 3 <= len(self.objects) <= 5, "Support, floor & wall should always be in scene"

        # Determining floor & wall
        floor_found, wall_found = False, False
        for this_ob in self.objects:

            w, h = ObjectV2._get_obj_dims(this_ob.obj_mask)
            cX, cY = self._get_obj_moments(this_ob.obj_mask)

            if w > 0.9 * MAX_X: # Wide enough to be wall or floor
                if cY > 0.5 * MAX_Y:  # Resides in lower half
                    this_ob.role = "floor"
                    floor_found = True
                else:
                    this_ob.role = "back-wall"
                    wall_found = True

        assert floor_found and wall_found, "Floor & wall should have been found by now"

        # Determining support
        support_found = False
        if len(self.objects) == 3:
            for this_ob in self.objects:
                if this_ob.role not in ["floor", "back-wall"]:
                    this_ob.role = "support"
                    support_found = True

        else:
            # Find closest object to floor
            biggest_cY = -float("inf")
            support_idx = None
            for idx, this_ob in enumerate(self.objects):
                if this_ob.role not in ["floor", "back-wall"]:
                    _, cY = self._get_obj_moments(this_ob.obj_mask)
                    if cY > biggest_cY:
                        support_idx = idx
                        biggest_cY = cY
            self.objects[support_idx].role = "support"
            support_found = True

        assert support_found, "Support should have been found by now"

        # Determining target
        target_found = False
        if len(self.objects) == 4:
            for this_ob in self.objects:
                if this_ob.role not in ["floor", "back-wall", "support"]:
                    this_ob.role = "target"
                    target_found = True
        else:
            # Find closest object to floor
            biggest_cY = -float("inf")
            target_idx = None
            for idx, this_ob in enumerate(self.objects):
                if this_ob.role not in ["floor", "back-wall", "support"]:
                    _, cY = self._get_obj_moments(this_ob.obj_mask)
                    if cY > biggest_cY:
                        target_idx = idx
                        biggest_cY = cY
            if target_idx is not None:
                self.objects[target_idx].role = "target"
                target_found = True

        # Determine pole
        pole_found = False
        if len(self.objects) == 5:
            for this_ob in self.objects:
                if this_ob.role == "default":
                    this_ob.role = "pole"
                    pole_found = True
                    break
        # TODO: write assertion to check consistency


    def calculate_physical_props(self):
        for this_ob in self.objects:
            this_ob.extract_physical_props()

    def guess_object_kinds(self):

        for this_ob in self.objects:
            this_ob.find_obj_kind_nn()

    def load_roles_as_attr(self):

        for this_ob in self.objects:
            
            # In their order of appearance
            if this_ob.role == "floor":
                self.floor = this_ob

            elif this_ob.role == "support":
                self.support = this_ob

            elif this_ob.role == "target":
                self.target = this_ob

            elif this_ob.role == "pole":
                self.pole = this_ob
