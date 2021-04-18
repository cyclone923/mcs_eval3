from physicsvoe.data.data_gen import convert_output
from physicsvoe import framewisevoe, occlude
from physicsvoe.timer import Timer
from physicsvoe.data.types import make_camera
from dataclasses import dataclass
from gravity import pybullet_utilities
from tracker import track, appearence as appearance, filter_masks
import visionmodule.inference as vision
from vision.gravity import L2DataPacketV2

from pathlib import Path
from PIL import Image
import numpy as np
import torch
import pickle
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import sys
import cv2
from rich.console import Console

console = Console()

APP_MODEL_PATH = './tracker/model.p'
VISION_MODEL_PATH = './visionmodule/dvis_resnet50_mc_voe.pth'
DEBUG = False

@dataclass
class ObjectFace:
    corners: list

    def __post_init__(self):
        x = sum(pt["x"] for pt in self.corners) / 4
        y = sum(pt["y"] for pt in self.corners) / 4
        self.centroid = (x, y)

    def __eq__(self, o: object, tol=1e-5) -> bool:
        # Assuming Object is rigid & won't deform during the motion
        x_matches = abs(self.centroid[0] - o.centroid[0]) < tol
        y_matches = abs(self.centroid[1] - o.centroid[1]) < tol

        return x_matches and y_matches

class VoeAgent:
    def __init__(self, controller, level, out_prefix=None):
        self.controller = controller
        self.level = level
        self.prefix = out_prefix if DEBUG else None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.app_model = appearance.AppearanceMatchModel()
        self.app_model.load_state_dict(torch.load(APP_MODEL_PATH, map_location=torch.device(self.device)))
        self.app_model = self.app_model.to(self.device).eval()
        if self.level == 'level1':
            self.visionmodel = vision.MaskAndClassPredictor(dataset='mcsvideo3_voe',
                                                            config='plus_resnet50_config_depth_MC',
                                                            weights=VISION_MODEL_PATH)
        else:
            self.visionmodel = None
    
    # # # # # # #
    # RUN SCENE #
    # # # # # # #
    def run_scene(self, config, desc_name):
        if DEBUG:
            print('* * * * DEBUG MODE ON * * * *')
            folder_name = Path(self.prefix)/Path(Path(desc_name).stem)
            if folder_name.exists():
                # ABORT and don't run the scene if we already have output
                # for this scene. Presumably if we're collecting debug output
                # we don't want to generate the same data twice.
                return None
            folder_name.mkdir(parents=True)
            console.print(folder_name)
        else:
            folder_name = None
        
        self.track_info = dict()
        self.detector = framewisevoe.FramewiseVOE(min_hist_count=3, max_hist_count=8, dist_thresh=0.5)

        self.controller.start_scene(config)

        target_position = dict()
        # target_trajectory = dict()
        pole_history = list()
        support_coords = None

        obj_traj_orn = dict()
        step_output = None
        step_output_dict = None
        drop_step = -1
        pb_state = 'incomplete'

        scene_voe_detected = False
        all_viols = list()
        all_errs = list()

        for i, x in enumerate(config['goal']['action_list']):
            step_output = self.controller.step(action=x[0])
            if step_output is None:
                break

            # Use the most recent depth map and RGB frames from the simulator's output
            depth_map = step_output.depth_map_list[-1]
            rgb_image = step_output.image_list[-1]

            # Get the object mask data, depending on level
            if self.level == 'oracle':
                masks = self.oracle_masks(step_output)
            elif self.level == 'level2':
                in_mask = step_output.object_mask_list[-1]
                masks = self.level2_masks(depth_map, rgb_image, in_mask)
            elif self.level == 'level1':
                masks = self.level1_masks(depth_map, rgb_image)
            else:
                raise ValueError(f'Unknown level `{self.level}`')

            # Calculate tracking info using mask history
            # This stores a bunch of state in `self.track_info`
            self.track_info = track.track_objects(masks, self.track_info)                

            # Calculate VoEs
            # TODO: Update to use non-binary confidence levels
            voe_detected, voe_heatmap, voe_xy_list, viols, frame_errs = self.calc_voe(step_output, i, rgb_image, depth_map, masks, folder_name)
            console.print('VoE' if voe_detected else 'Not VoE', style='green' if voe_detected else 'red')
            console.print('')
            all_viols.append(viols)
            all_errs += frame_errs

            console.print('Violations:', viols)
            console.print('Errors:', frame_errs)
            scene_voe_detected = scene_voe_detected or voe_detected
            console.print('[yellow]Scene-Wide VoE Detected?[/yellow]', scene_voe_detected, style='yellow')
            choice = plausible_str(voe_detected)
            if DEBUG:
                assert choice in config['goal']['metadata']['choose'] # Sanity check
            self.controller.make_step_prediction(
                choice=choice, confidence=1.0, violations_xy_list=voe_xy_list,
                heatmap_img=voe_heatmap)
            if step_output is None:
                break

        self.controller.end_scene(choice=plausible_str(scene_voe_detected), confidence=1.0)
        if DEBUG:
            with open(folder_name/'viols.pkl', 'wb') as fd:
                pickle.dump((all_viols, all_errs), fd)
        return scene_voe_detected

    # Calculate VoEs (gravity + physics should occur here)
    def calc_voe(self, step_output, frame_num, rgb_image, depth_map, masks, scene_name=None):
        
        # We need to calculate+store some camera properties so that we can
        # project points between screen space into world space
        camera_info = make_camera(step_output)
        
        # Add appearance VoE model's output to the object tracking info.
        self.track_info['objects'] = \
            appearance.object_appearance_match(self.app_model, rgb_image,
                                               self.track_info['objects'],
                                               self.device, self.level)

        # The tracking model assigns an ID to each mask that is consistent
        # across frames, and calculates a mask for each object ID.
        # We use `squash_masks` to turn this list of object masks into a single
        # one-hot encoded matrix associating each screen pixel with an object.
        all_obj_ids = list(range(self.track_info['object_index']))
        masks_list = [self.track_info['objects'][i]['mask'] for i in all_obj_ids]
        tracked_masks = self.squash_masks(depth_map, masks_list, all_obj_ids)
        
        # Call 'occlusion' model
        # 'Occlusion' really just means that the object is sufficiently
        # obscured that we can't rely on our appearance model or position
        # estimation.
        # Therefore, if an object is determined to be 'occluded' we just ignore
        # all raised VoEs.
        area_hists = {o_id:o_info['area_history'] for o_id, o_info in self.track_info['objects'].items()}
        obj_occluded = occlude.detect_occlusions(depth_map, tracked_masks, all_obj_ids, area_hists)
        console.print('[yellow]Objects occluded?[/yellow]', obj_occluded)
        
        # Calculate object level info from masks
        # We can use the object masks + depth map + camera data to easily
        # estimate world_space positions for each object in the scene.
        obj_ids, obj_pos, obj_present = \
            framewisevoe.calc_world_pos(depth_map, tracked_masks, camera_info)
        occ_heatmap = framewisevoe.make_occ_heatmap(obj_occluded, obj_ids, tracked_masks)

        # TODO: Add dynamics model's output to the object tracking info.
        # NOTE: This should include position, entrance, and presence violations
        # NOTE: Gravity violations should either be defined as their own class
        # of violations or should fall under position violations.
        det_result = self.detector.detect(frame_num, obj_pos, obj_occluded, obj_ids, depth_map, camera_info)
        if det_result is None:
            # Early return from VoE detector = no VoEs present
            dynamics_viols = []
            all_errs = []
        else:
            dynamics_viols, all_errs = det_result   # Gather any dynamics VoEs (i.e. Presence and Position VoEs)
        console.print('[yellow]Dynamics violations:[/yellow]', dynamics_viols)
        
        # Record any appearance VoEs
        appearance_viols = []
        for o_id, obj_info in self.track_info['objects'].items():
            console.log('Object ID:', o_id)
            o_visible = obj_info['visible']     # Is the object visible?
            o_mismatch = not obj_info['appearance']['match']    # Is the object an appearance mismatch?
            o_occluded = obj_occluded[o_id]     # Is the object occluded?
            console.log('Visible?', o_visible)
            console.log('Appearance mismatch?', o_mismatch)
            console.log('Occluded?', o_occluded)
            # Is the object a robust appearance mismatch?
            o_robust_mismatch = o_mismatch and obj_info['appearance']['mismatch_count'] > 3
            console.log('Robust appearance mismatch?', o_robust_mismatch)
            # Throw an Appearance VoE if the object is visible, unoccluded,
            # and a robust appearance mismatch.
            app_viol = o_visible and o_robust_mismatch and not o_occluded
            if app_viol:
                _idx = obj_ids.index(o_id)
                appearance_viols.append(framewisevoe.AppearanceViolation(o_id, obj_pos[_idx]))
            
        vis_count = {o_id:o_info['visible_count'] for o_id, o_info in self.track_info['objects'].items()}
        pos_hists = {o_id:o_info['position_history'] for o_id, o_info in self.track_info['objects'].items()}

        # Record any observation violations (i.e. Entrance VoEs)
        obs_viols = self.detector.record_obs(frame_num, obj_ids, obj_pos, obj_present, obj_occluded, vis_count, pos_hists, camera_info)
        console.print('[yellow]Observation violations:[/yellow]', obs_viols)
        
        # Gather all VoEs
        viols = dynamics_viols + obs_viols
        if self.level != 'level1': # Ignore appearance violations in the level1 case
            viols += appearance_viols
        
        # Create VoE heatmap from the list of violations, for step output
        voe_hmap = framewisevoe.make_voe_heatmap(viols, tracked_masks)
        if DEBUG:
            framewisevoe.output_voe(viols)
            framewisevoe.show_scene(scene_name, frame_num, depth_map, tracked_masks, voe_hmap, occ_heatmap)
        
        # Output results
        voe_detected = viols is not None and len(viols) > 0
        voe_hmap_img = Image.fromarray(voe_hmap)
        voe_xy_list = [v.xy_pos(camera_info) for v in (viols or [])]
        return voe_detected, voe_hmap_img, voe_xy_list, viols, all_errs
    
    # Find the precise point at which the pole's suction is turned off
    def determine_drop_step(self, pole_history):
        if self.level == 'level2':
            def _bgr2gray(b, g, r):
                '''
                Formula designed to expand gap between magenta & cyan
                '''
                return 0.01 * g + 0.99 * r
            gray_values = [
                _bgr2gray(*md["color"])
                for md in pole_history
            ]
            dhistory = []
            for i in range(1, len(gray_values)):
                dc = gray_values[i] - gray_values[i - 1]
                dhistory.append(abs(dc))

            if dhistory:
                if max(dhistory) < 100:
                    return -1

                offset = dhistory.index(max(dhistory))
                drop_step = pole_history[offset]["step_id"]

                return drop_step + 1
            else:
                return -1
        else:
            if type(pole_history[0]) != type({}):
                pole_y_position = [
                    pt[2][0] for pt in pole_history if pt is not None
                ]

            lowest_point = sys.maxsize
            for i in range(len(pole_y_position)):
                if pole_y_position[i] <= lowest_point:
                    lowest_point = pole_y_position[i]
                else:
                    return i
            return -1
    

    # Methods from Gravity Eval 3.5
    @staticmethod
    def get_object_bounding_simplices(dims):
        y_coords = [pt["y"] for pt in dims]
        min_y, max_y = min(y_coords), max(y_coords)

        # Assuming "nice" placement with cubes
        # For generalizing, calculate convex hull and findout extreme simplices 
        bottom_face = ObjectFace(corners=[pt for pt in dims if pt["y"] == min_y ])
        top_face = ObjectFace(corners=[pt for pt in dims if pt["y"] == max_y ])

        return top_face, bottom_face

    def states_during_and_after_drop(self, drop_step, target_trajectory, support):

        # Assuming target is moving along "y"
        target_drop_coords = target_trajectory[drop_step]
        _, target_bottom_face = self.get_object_bounding_simplices(target_drop_coords)
        
        target_resting_coords = target_trajectory[-1]
        _, target_bottom_face_end_state = self.get_object_bounding_simplices(target_resting_coords)

        support_top_face, _ = self.get_object_bounding_simplices(support)

        return target_bottom_face, support_top_face, target_bottom_face_end_state


    @staticmethod
    def target_should_be_stable(target, support):

        support_x_range = (min(pt["x"] for pt in support.corners), max(pt["x"] for pt in support.corners))
        support_y_range = (min(pt["y"] for pt in support.corners), max(pt["x"] for pt in support.corners))

        target_x_inrange = support_x_range[0] <= target.centroid[0] <= support_x_range[1]
        target_y_inrange = support_y_range[0] <= target.centroid[1] <= support_y_range[1]

        return target_x_inrange and target_y_inrange

    @staticmethod
    def get_target_obj_ids(step_output):
        return list(step_output['object_list'].keys())

    @staticmethod
    def get_structural_obj_ids(step_output):
        filtered_keys = [
            key 
            for key in step_output["structural_object_list"].keys() 
            if len(key) > 35
        ]
        out = dict({
            ("pole", so) if "pole_" in so else ("support", so)
            for so in filtered_keys
        }
        )
        return out.get("support"), out.get("pole")
    
    def convert_l2_to_dict(self, metadata):
        step_output_dict = {
            "camera_field_of_view": 42.5,
            "camera_aspect_ratio": (600, 400),
            "structural_object_list": {
                "support": {
                    "dimensions": metadata.support.dims,
                    "position": {
                        "x": metadata.support.centroid[0],
                        "y": metadata.support.centroid[1],
                        "z": metadata.support.centroid[2]
                    },
                    "color": {
                        "r": metadata.support.color[0],
                        "g": metadata.support.color[1],
                        "b": metadata.support.color[2],
                    },
                    "shape": metadata.support.kind,
                    "mass": 100
                },
                "floor": {
                    "dimensions": metadata.floor.dims,
                    "position": metadata.floor.centroid,
                    "color": metadata.floor.color,
                    "shape": metadata.floor.kind
                }
            },
            "object_list": {}
        }

        if hasattr(metadata, "pole"):
            step_output_dict["structural_object_list"]["pole"] = {
                "dimensions": metadata.pole.dims,
                "position": metadata.pole.centroid,
                "color": metadata.pole.color,
                "shape": metadata.pole.kind
            }
        
        if hasattr(metadata, "target"):
            step_output_dict["object_list"]["target"] = {
                "dimensions": metadata.target.dims,
                "position": {
                    "x": metadata.target.centroid[0],
                    "y": metadata.target.centroid[1],
                    "z": metadata.target.centroid[2]
                },
                "color": {
                        "r": metadata.target.color[0],
                        "g": metadata.target.color[1],
                        "b": metadata.target.color[2],
                },
                "shape": metadata.target.kind,
                "mass": 4.0,
                "pixel_center": metadata.target.centroid_px
            }

        return step_output_dict
    
    def getIntersectionOrContact(self, obj1, obj2):
        obj1_dims = self.getMinMax(obj1)
        obj1_dims = [(obj1_dims[i][0] - 0.05, obj1_dims[i][1] + 0.05) for i in range(len(obj1_dims))]
        obj2_dims = self.getMinMax(obj2)
        obj2_dims = [(obj2_dims[i][0] - 0.05, obj2_dims[i][1] + 0.05) for i in range(len(obj2_dims))]

        # print(obj1_dims)
        # print(obj2_dims)
        x_check = (obj1_dims[0][0] <= obj2_dims[0][1] and obj1_dims[0][1] >= obj2_dims[0][0])
        z_check = (obj1_dims[1][0] <= obj2_dims[1][1] and obj1_dims[1][1] >= obj2_dims[1][0])
        y_check = (obj1_dims[2][0] <= obj2_dims[2][1] and obj1_dims[2][1] >= obj2_dims[2][0])
        # print("x", x_check)
        # print("z", z_check)
        # print("y", y_check)

        return x_check and z_check and y_check

    def getMinMax(self, obj):
        dims = obj["dimensions"]
        min_x = sys.maxsize
        min_y = sys.maxsize
        min_z = sys.maxsize
        max_x = -1*sys.maxsize
        max_y = -1*sys.maxsize
        max_z = -1*sys.maxsize
        
        for dim in dims:
            if dim['x'] <= min_x:
                min_x = dim['x']
            if dim['x'] >= max_x:
                max_x = dim['x']

            if dim['y'] <= min_y:
                min_y = dim['y']
            if dim['y'] >= max_y:
                max_y = dim['y']

            if dim['z'] <= min_z:
                min_z = dim['z']
            if dim['z'] >= max_z:
                max_z = dim['z']
        
        return [(min_x, max_x), (min_z, max_z), (min_y, max_y)]
    
    def calc_simulator_voe(self, pybullet_traj, unity_traj):
        # calculate the difference in trajectory as the sum squared distance of each point in time
        
        distance, path = fastdtw(pybullet_traj, unity_traj, dist=euclidean)
        return distance, path
    
    def oracle_masks(self, step_output):
        frame = convert_output(step_output)
        return frame.obj_mask

    def level2_masks(self, depth_img, rgb_img, mask_img):
        in_mask = np.array(mask_img)
        unique_cols = np.unique(in_mask.reshape(-1, 3), axis=0)
        split_masks = [(in_mask == col).all(axis=-1) for col in unique_cols]
        filter_result = filter_masks.filter_objects(rgb_img, depth_img, split_masks)
        masks = -1 * np.ones(in_mask.shape[:2], dtype=np.int)
        for i, o in enumerate(filter_result['objects']):
            masks[o] = i
        return masks

    def level1_masks(self, depth_img, rgb_img):
        bgr_img = np.array(rgb_img)[:, :, [2, 1, 0]]
        result = self.visionmodel.step(bgr_img, depth_img)
        filter_result = filter_masks.filter_objects_model(rgb_img, depth_img, result)
        masks = -1 * np.ones(np.array(depth_img).shape[:2], dtype=np.int)
        for i, o in enumerate(filter_result['objects']):
            masks[o] = i
        return masks
    
    @staticmethod
    def squash_masks(ref, mask_l, ids):
        flat_mask = np.ones_like(ref) * -1
        for m, id_ in zip(mask_l, ids):
            flat_mask[m] = id_
        return flat_mask

def plausible_str(violation_detected):
    return 'implausible' if violation_detected else 'plausible'
    