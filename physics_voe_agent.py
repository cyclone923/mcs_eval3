from dataclasses import dataclass
from physicsvoe.data.data_gen import convert_output
from physicsvoe import framewisevoe, occlude
from physicsvoe.timer import Timer
from physicsvoe.data.types import make_camera

from gravity import pybullet_utilities
from vision.physics import L2DataPacketV2
# from gravity.gravity_utilities import convert_l2_to_dict

from tracker import track, appearence, filter_masks
import visionmodule.inference as vision

from pathlib import Path
from PIL import Image
import numpy as np
import torch
import pickle
import sys

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

class PhysicsVoeAgent:
    def __init__(self, controller, level, out_prefix=None):
        self.controller = controller
        self.level = level
        if DEBUG:
            self.prefix = out_prefix
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.app_model = appearence.AppearanceMatchModel()
        self.app_model.load_state_dict(torch.load(APP_MODEL_PATH, map_location=torch.device('cpu')))
        self.app_model = self.app_model.to(self.device).eval()
        # if self.level == 'level1':
        #     self.visionmodel = vision.MaskAndClassPredictor(dataset='mcsvideo3_voe',
        #                                                     config='plus_resnet50_config_depth_MC',
        #                                                     weights=VISION_MODEL_PATH)
    
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
    def target_obj_ids(step_output):
        if len(list(step_output['object_list'].keys())) > 0:
            return list(step_output['object_list'].keys())
        return None

    @staticmethod
    def struc_obj_ids(step_output):
        filtered_keys = [
            key 
            for key in step_output["structural_object_list"].keys() 
            if len(key) > 35
        ]
        out = dict({
            ("pole", so) if "pole_" in so else ("support", so) if "support_" in so else ("occluder", so)
            for so in filtered_keys
        }
        )
        return out.get("support"), out.get("pole"), out.get("occluder")
    
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
                },
                "occluders": {}
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
        
        if hasattr(metadata, "occluders"):
            for o_id, occluder in metadata.occluders.items():
                step_output_dict["structural_object_list"]["occluders"][o_id] = {
                    "dimensions": occluder.dims,
                    "position": occluder.centroid,
                    "color": occluder.color,
                    "shape": occluder.kind
                }
        
        if hasattr(metadata, "default"):
            for o_id, object in metadata.default.items():
                step_output_dict["object_list"][o_id] = {
                    "dimensions": object.dims,
                    "position": {
                        "x": object.centroid[0],
                        "y": object.centroid[1],
                        "z": object.centroid[2]
                    },
                    "color": {
                            "r": object.color[0],
                            "g": object.color[1],
                            "b": object.color[2],
                    },
                    "shape": object.kind,
                    "mass": 4.0,
                    "pixel_center": object.centroid_px
                }

        return step_output_dict
    
    def run_scene(self, config, desc_name):
        if DEBUG:
            folder_name = Path(self.prefix)/Path(Path(desc_name).stem)
            if folder_name.exists():
                return None
            folder_name.mkdir(parents=True)
            print(folder_name)
        else:
            folder_name = None
        
        self.detector = \
            framewisevoe.FramewiseVOE(min_hist_count=3, max_hist_count=8,
                                      dist_thresh=0.5)
        self.controller.start_scene(config)
        pole_history = []  # To determine drop step
        self.track_info = dict()
        scene_voe_detected = False
        all_viols = []
        all_errs = []

        for i, x in enumerate(config['goal']['action_list']):
            step_output = self.controller.step(action=x[0])
            if self.level == "oracle":
                if step_output is None:
                    break
                else:
                    step_output_dict = dict(step_output)
                    try:
                        step_output = L2DataPacketV2(step_number=i, step_meta=step_output)
                    except Exception as e:
                        print("Couldn't process step i+{}, skipping ahead".format(i))
                        print(e)
                        continue

                floor_object = "floor"  # Not a dynamic object ID
                target_objects = self.target_obj_ids(step_output_dict)
                supporting_object, pole_object, occluder_objects = self.struc_obj_ids(step_output_dict)
            else:
                if step_output is None:
                    break
                
                try:
                    step_output = L2DataPacketV2(step_number=i, step_meta=step_output)
                except Exception as e:
                    print("Couldn't process step i+{}, skipping ahead".format(i))
                    print(e)
                    continue
                
                # convert L2DataPacketV2 to dictionary
                step_output_dict = self.convert_l2_to_dict(step_output)
                
                floor_object = "floor"
                target_objects = self.target_obj_ids(step_output_dict)
                supporting_object, pole_object, occluder_objects = self.struc_obj_ids(step_output_dict)
            
            try:
                # Expected to not dissapear until episode ends
                support_coords = step_output_dict["structural_object_list"][supporting_object]["dimensions"]
                floor_coords = step_output_dict["structural_object_list"][floor_object]["dimensions"]

                # Target / Pole may not appear in view yet, start recording when available
                for obj_id, obj in step_output_dict['object_list'].items():
                    if obj_id not in self.track_info.keys():
                        self.track_info[obj_id] = {
                            'trajectory': list(),
                            'position': list()
                        }
                    self.track_info[obj_id]['trajectory'].append(obj["dimensions"])
                    self.track_info[obj_id]['position'].append(obj['position'])
                    step_output_dict["object_list"][obj_id]["pixel_center"] = step_output.default[obj_id].centroid_px
                
                if self.level == 'level2':
                    pole_history.append({
                            "color": step_output_dict["structural_object_list"][pole_object]['color'],
                            "step_id": i
                        })
                elif self.level == "oracle":
                    pole_history.append(self.getMinMax(step_output_dict["structural_object_list"][pole_object]))
            
            except KeyError:  # Object / Pole is not in view yet
                pass
            except AttributeError:
                pass

            # TODO: PERFORM TARGET OBJECT APPEARANCE MATCH

            # TODO: CHECK FOR ENTRANCE VOE

            # TODO: RUN PYBULLET

class VoeAgent:
    def __init__(self, controller, level, out_prefix=None):
        self.controller = controller
        self.level = level
        if DEBUG:
            self.prefix = out_prefix
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.app_model = appearence.AppearanceMatchModel()
        self.app_model.load_state_dict(torch.load(APP_MODEL_PATH, map_location=torch.device('cpu')))
        self.app_model = self.app_model.to(self.device).eval()
        if self.level == 'level1':
            self.visionmodel = vision.MaskAndClassPredictor(dataset='mcsvideo3_voe',
                                                            config='plus_resnet50_config_depth_MC',
                                                            weights=VISION_MODEL_PATH)

    def run_scene(self, config, desc_name):
        if DEBUG:
            folder_name = Path(self.prefix)/Path(Path(desc_name).stem)
            if folder_name.exists():
                return None
            folder_name.mkdir(parents=True)
            print(folder_name)
        else:
            folder_name = None
        self.track_info = {}
        self.detector = \
            framewisevoe.FramewiseVOE(min_hist_count=3, max_hist_count=8,
                                      dist_thresh=0.5)
        self.controller.start_scene(config)
        scene_voe_detected = False
        all_viols = []
        all_errs = []
        for i, x in enumerate(config['goal']['action_list']):
            step_output = self.controller.step(action=x[0])
            voe_detected, voe_heatmap, voe_xy_list, viols, frame_errs = self.calc_voe(step_output, i, folder_name)

            all_viols.append(viols)
            all_errs += frame_errs
            scene_voe_detected = scene_voe_detected or voe_detected
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

    def calc_voe(self, step_output, frame_num, scene_name=None):
        depth_map = step_output.depth_map_list[-1]
        rgb_image = step_output.image_list[-1]
        camera_info = make_camera(step_output)
        if self.level == 'oracle':
            masks = self.oracle_masks(step_output)
        elif self.level == 'level2':
            in_mask = step_output.object_mask_list[-1]
            masks = self.level2_masks(depth_map, rgb_image, in_mask)
        elif self.level == 'level1':
            masks = self.level1_masks(depth_map, rgb_image)
        else:
            raise ValueError(f'Unknown level `{self.level}`')
        
        # get object data for pybullet
        step_output_dict = {}
        try:
            step_output = L2DataPacketV2(step_number=frame_num, step_meta=step_output, scene=scene_name)
            # step_output_dict = convert_l2_to_dict(step_output)
        except Exception as e:
            print("Couldn't process step i={}".format(frame_num))
            print(e)

        # Calculate tracking info
        self.track_info = track.track_objects(masks, self.track_info)
        self.track_info['objects'] = \
            appearence.object_appearance_match(self.app_model, rgb_image,
                                               self.track_info['objects'],
                                               self.device, self.level)
        if DEBUG:
            img = appearence.draw_bounding_boxes(rgb_image, self.track_info['objects'])
            img = appearence.draw_appearance_bars(img, self.track_info['objects'])
            img.save(scene_name/f'DEBUG_{frame_num:02d}.png')
        all_obj_ids = list(range(self.track_info['object_index']))
        masks_list = [self.track_info['objects'][i]['mask'] for i in all_obj_ids]
        tracked_masks = squash_masks(depth_map, masks_list, all_obj_ids)
        # Calculate occlusion from masks
        area_hists = {o_id:o_info['area_history'] for o_id, o_info in self.track_info['objects'].items()}
        obj_occluded = occlude.detect_occlusions(depth_map, tracked_masks, all_obj_ids, area_hists)
        # Calculate object level info from masks
        obj_ids, obj_pos, obj_present = \
            framewisevoe.calc_world_pos(depth_map, tracked_masks, camera_info)
        occ_heatmap = framewisevoe.make_occ_heatmap(obj_occluded, obj_ids, tracked_masks)
        # Calculate violations
        det_result = self.detector.detect(frame_num, obj_pos, obj_occluded, obj_ids, depth_map, camera_info)
        if det_result is None:
            dynamics_viols = []
            all_errs = []
        else:
            dynamics_viols, all_errs = det_result
        appearance_viols = []
        for o_id, obj_info in self.track_info['objects'].items():
            o_visible = obj_info['visible']
            o_mismatch = not obj_info['appearance']['match']
            o_occluded = obj_occluded[o_id]
            o_robust_mismatch = o_mismatch and obj_info['appearance']['mismatch_count'] > 3
            app_viol = o_visible and o_robust_mismatch and not o_occluded
            if app_viol:
                _idx = obj_ids.index(o_id)
                appearance_viols.append(framewisevoe.AppearanceViolation(o_id, obj_pos[_idx]))
        # Update tracker
        vis_count = {o_id:o_info['visible_count'] for o_id, o_info in self.track_info['objects'].items()}
        pos_hists = {o_id:o_info['position_history'] for o_id, o_info in self.track_info['objects'].items()}
        obs_viols = self.detector.record_obs(frame_num, obj_ids, obj_pos, obj_present, obj_occluded, vis_count, pos_hists, camera_info)
        # Output violations
        viols = dynamics_viols + obs_viols
        if self.level != 'level1': #Ignore appearance violations in the level1 case
            viols += appearance_viols
        voe_hmap = framewisevoe.make_voe_heatmap(viols, tracked_masks)
        if DEBUG:
            framewisevoe.output_voe(viols)
            framewisevoe.show_scene(scene_name, frame_num, depth_map, tracked_masks, voe_hmap, occ_heatmap)
        # Output results
        voe_detected = viols is not None and len(viols) > 0
        voe_hmap_img = Image.fromarray(voe_hmap)
        voe_xy_list = [v.xy_pos(camera_info) for v in (viols or [])]
        return voe_detected, voe_hmap_img, voe_xy_list, viols, all_errs

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

def squash_masks(ref, mask_l, ids):
    flat_mask = np.ones_like(ref) * -1
    for m, id_ in zip(mask_l, ids):
        flat_mask[m] = id_
    return flat_mask

def prob_to_mask(prob, cutoff, obj_scores):
    obj_pred_class = obj_scores.argmax(-1)
    valid_ids = []
    for mask_id, pred_class in zip(range(cutoff, prob.shape[0]), obj_pred_class):
        if pred_class == 1: #An object!
            valid_ids.append(mask_id+cutoff)
    out_mask = -1 * np.ones(prob.shape[1:], dtype=np.int)
    am = np.argmax(prob, axis=0)
    for out_id, mask_id in enumerate(valid_ids):
        out_mask[am==mask_id-cutoff] = out_id
    return out_mask

def plausible_str(violation_detected):
    return 'implausible' if violation_detected else 'plausible'
