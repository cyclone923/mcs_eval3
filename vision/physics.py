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
    "back-wall",
    "occluder"
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
class Object:
    rgb_im: np.ndarray  # Isomeric view
    obj_mask: np.ndarray
    depth_map: np.ndarray

    role: str = "default"
    kind: str = "default"
    id: str = "default"

    dims: tuple = None
    w_h_d: tuple = None
    centroid: tuple = None
    centroid_px: tuple = None
    bbox_corners: np.ndarray = None

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
        depth_scale_correction_factor = 255 / CAM_CLIP_PLANES[1]
        depth = (depth_scale_correction_factor * depth).astype(np.uint8)

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
        camera_offsets /= depth_scale_correction_factor

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
        obj_point_cloud = scene_point_cloud.select_by_index(obj_idx)
        self.obj_cloud = obj_point_cloud

        assert obj_point_cloud.dimension() == 3, "RGB-D couldn't return a 3D object!"
        
        if visualize:
            o3d.visualization.draw_geometries([obj_point_cloud])

        bbox = obj_point_cloud.get_axis_aligned_bounding_box()
        self.bbox_corners = np.asarray(bbox.get_box_points())

        dy = 1.75
        self.bbox_corners = np.asarray(
            bbox.translate([0, dy, 0], relative=True)
                .get_box_points()
            )
        self.dims = [
            {"x": pt[0], "y": pt[1], "z": pt[2]}
            for pt in self.bbox_corners
        ]
        self.w_h_d = np.abs(bbox.get_extent()).tolist()

    @staticmethod
    def clamp(x):
        return max(0, min(x, 255))

    def extract_physical_props(self):
        '''
        Uses blob to determine `color` & `dims`
        '''
        self.color = self._color_prop(self.obj_mask, self.rgb_im)
        # r, g, b = self.color[0], self.color[1], self.color[2]

        # self.id = "{0:02},{1:02},{2:02}".format(self.clamp(r), self.clamp(g), self.clamp(b))

        self._dims_prop(self)

        x = sum(pt["x"] for pt in self.dims) / 8
        y = sum(pt["y"] for pt in self.dims) / 8
        z = sum(pt["z"] for pt in self.dims) / 8
        self.centroid = (x, y, z)

        self.centroid_px = L2DataPacket._get_obj_moments(self.obj_mask)

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

        if self.role in ["pole", "floor", "support", "back-wall", "occluder"]:
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

        if self.role == "pole":
            self.kind = "cylinder"
            return
        # TODO: reintroduce nn
        
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
class L2DataPacket:

    step_number: int
    step_meta: mcs.StepMetadata
    scene: str

    def __post_init__(self):

        self.get_images_from_meta()
        self.non_actor_objects = self.segment_objects()
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
            this_ob = Object(
                rgb_im=self.rgb_im,
                obj_mask=obj,
                depth_map=self.depth_map,
            )

            objects.append(this_ob)

        return objects

    def determine_obj_roles(self):
        # assert 3 <= len(self.objects) <= 5, "Support, floor & wall should always be in scene"
        # Determining floor & wall
        floor_found, wall_found = False, False
        for this_ob in self.non_actor_objects:

            w, h = Object._get_obj_dims(this_ob.obj_mask)
            cX, cY = self._get_obj_moments(this_ob.obj_mask)

            if w > 0.9 * MAX_X: # Wide enough to be wall or floor
                if cY > 0.5 * MAX_Y:  # Resides in lower half
                    this_ob.role = "floor"
                    floor_found = True
                else:
                    this_ob.role = "back-wall"
                    wall_found = True

        assert floor_found and wall_found, "Floor & wall should have been found by now"

        # Determine pole(s) by finding closest objects to ceiling
        smallest_cY = float("inf")
        mid_cY = self.rgb_im.shape[0] / 2
        pole_idx = None
        for idx, this_ob in enumerate(self.non_actor_objects):
            if this_ob.role not in ["floor", "back-wall"]:
                _, cY = self._get_obj_moments(this_ob.obj_mask)
                if cY <= smallest_cY and cY < mid_cY/2:
                    pole_idx = idx
                    smallest_cY = cY
            if pole_idx is not None:
                self.non_actor_objects[pole_idx].role = "pole"
                pole_idx = None

        # Determine occluder(s) by finding large objects with centroids near the horizontal axis
        occluder_idx = None
        nearest_cY = float("inf")
        for idx, this_ob in enumerate(self.non_actor_objects):
            if this_ob.role not in ["floor", "back-wall", "pole"]:
                _, cY = self._get_obj_moments(this_ob.obj_mask)
                w, h = Object._get_obj_dims(this_ob.obj_mask)
                # if centroid is sufficiently near the middle of the screen and tall enough to be an occluder
                if np.abs(mid_cY - cY) < nearest_cY and np.abs(mid_cY - cY) < 50 and h > (2/3) * MAX_Y:
                    occluder_idx = idx
                    smallest_cY = cY
            if occluder_idx is not None:
                self.non_actor_objects[occluder_idx].role = "occluder"
                occluder_idx = None

        # for idx, this_ob in enumerate(self.objects):
        #     if this_ob.role == "default":
        #         this_ob.id = 

    def calculate_physical_props(self):
        for this_ob in self.non_actor_objects:
            this_ob.extract_physical_props()

    def guess_object_kinds(self):

        for this_ob in self.non_actor_objects:
            this_ob.find_obj_kind_nn()

    def load_roles_as_attr(self):
        self.poles = []
        self.occluders = []
        self.targets = []
        
        for this_ob in self.non_actor_objects:
            
            # In their order of appearance
            if this_ob.role == "floor":
                self.floor = this_ob

            # elif this_ob.role == "support":
            #     self.support = this_ob

            # elif this_ob.role == "target":
            #     self.target = this_ob

            elif this_ob.role == "pole":
                self.poles.append(this_ob)

            elif this_ob.role == "occluder":
                self.occluders.append(this_ob)

            elif this_ob.role == "default":
                self.targets.append(this_ob)