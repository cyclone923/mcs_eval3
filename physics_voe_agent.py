from dataclasses import dataclass
from physicsvoe.data.data_gen import convert_output
from physicsvoe import framewisevoe, occlude
from physicsvoe.timer import Timer
from physicsvoe.data.types import make_camera
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from gravity import pybullet_utilities, gravity_utilities
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
            # if len(key) > 35
        ]
        out = dict({
            ("pole", so) if "pole_" in so else ("support", so) if "support_" in so else ("occluder", so)
            for so in filtered_keys
        }
        )
        # console.log(out)
        return out.get("support"), out.get("pole"), out.get("occluder")
    
    def convert_l2_to_dict(self, metadata):
        step_output_dict = {
            "camera_field_of_view": 42.5,
            "camera_aspect_ratio": (600, 400),
            "structural_object_list": {
                "floor": {
                    "dimensions": metadata.floor.dims,
                    "position": metadata.floor.centroid,
                    "color": metadata.floor.color,
                    "shape": metadata.floor.kind
                }
            },
            "object_list": {}
        }

        if hasattr(metadata, "poles"):
            step_output_dict["structural_object_list"]["poles"] = []

            for pole in metadata.poles:
                step_output_dict["structural_object_list"]["poles"].append({
                    "dimensions": pole.dims,
                    "position": pole.centroid,
                    "color": pole.color,
                    "shape": pole.kind
                })
        
        if hasattr(metadata, "occluders"):
            step_output_dict["structural_object_list"]["occluders"] = []

            for occluder in metadata.occluders:
                step_output_dict["structural_object_list"]["occluders"].append({
                    "dimensions": occluder.dims,
                    "position": occluder.centroid,
                    "color": occluder.color,
                    "shape": occluder.kind
                })
        
        if hasattr(metadata, "targets"):
            # step_output_dict["object_list"] = dict()
            
            for i, target in enumerate(metadata.targets):
                step_output_dict["object_list"][str(i)] = {
                    "dimensions": target.dims,
                    "position": {
                        "x": target.centroid[0],
                        "y": target.centroid[1],
                        "z": target.centroid[2]
                    },
                    "color": {
                            "r": target.color[0],
                            "g": target.color[1],
                            "b": target.color[2],
                    },
                    "shape": target.kind,
                    "mass": 10.0,
                    "pixel_center": target.centroid_px,
                    "obj_mask": target.obj_mask
                }

        return step_output_dict
    
    def calc_simulator_voe(self, pybullet_traj, unity_traj):
        # calculate the difference in trajectory as the sum squared distance of each point in time
        
        distance, path = fastdtw(pybullet_traj, unity_traj, dist=euclidean)
        return distance, path
    
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
        voe_xy_list = list()

        obj_traj_orn = None

        pb_state = 'incomplete'
        for i, x in enumerate(config['goal']['action_list']):
            step_output = self.controller.step(action=x[0]) # Get the step output

            # Get details of the objects in the scene
            if self.level == "oracle":
                if step_output is None:
                    break
                else:
                    step_output_dict = dict(step_output)
                    try:
                        step_output = L2DataPacketV2(step_number=i, step_meta=step_output, scene=desc_name)
                    except Exception as e:
                        print("Couldn't process step i+{}, skipping ahead".format(i))
                        print(e)
                        continue

                floor_object = "floor"  # Not a dynamic object ID
                # console.log(step_output_dict)
                target_objects = self.target_obj_ids(step_output_dict)
                # console.log(target_objects)
                supporting_object, pole_object, occluder_objects = self.struc_obj_ids(step_output_dict)
            else:
                if step_output is None:
                    break
                
                try:
                    step_output = L2DataPacketV2(step_number=i, step_meta=step_output, scene=desc_name)
                except Exception as e:
                    print("Couldn't process step i+{}, skipping ahead".format(i))
                    print(e)
                    continue
                
                # convert L2DataPacketV2 to dictionary
                # console.log(dir(step_output.step_meta))
                # console.log(dir(step_output))
                step_output_dict = self.convert_l2_to_dict(step_output)
                # console.log(step_output_dict.keys())
                
                floor_object = "floor"
                target_objects = self.target_obj_ids(step_output_dict)
                # console.log(target_objects)
                supporting_object, pole_object, occluder_objects = self.struc_obj_ids(step_output_dict)
            
            # console.log('check')
            # Track each object's trajectory and position through the scene
            try:
                # Expected to not dissapear until episode ends
                # support_coords = step_output_dict["structural_object_list"][supporting_object]["dimensions"]
                # floor_coords = step_output_dict["structural_object_list"][floor_object]["dimensions"]
                # console.log(support_coords)
                # console.log(floor_coords)

                # Target / Pole may not appear in view yet, start recording when available
                # console.log(step_output_dict['object_list'].keys())
                # console.log(self.track_info.keys())
                for obj_id, obj in step_output_dict['object_list'].items():
                    # console.log(obj_id)
                    if obj_id not in self.track_info.keys():
                        console.log('new')
                        self.track_info[obj_id] = {
                            'trajectory': list(),
                            'position': list()
                        }
                    self.track_info[obj_id]['trajectory'].append(obj["dimensions"])
                    self.track_info[obj_id]['position'].append(obj['position'])
                    step_output_dict["object_list"][obj_id]["pixel_center"] = obj['pixel_center']
                
                # if self.level == 'level2':
                #     pole_history.append({
                #             "color": step_output_dict["structural_object_list"][pole_object]['color'],
                #             "step_id": i
                #         })
                # elif self.level == "oracle":
                #     pole_history.append(self.getMinMax(step_output_dict["structural_object_list"][pole_object]))
            
            except KeyError as e:  # Object / Pole is not in view yet
                console.log(e)
                pass
            except AttributeError as e:
                console.log(e)
                pass

            # TODO: PERFORM TARGET OBJECT APPEARANCE MATCH

            # TODO: CHECK FOR ENTRANCE VOE

            # TODO: RUN PYBULLET
            if 'pole' in step_output_dict['structural_object_list'].keys():
                drop_step = self.determine_drop_step(pole_history)
                if drop_step != -1 and pb_state != 'complete' and len(step_output_dict['structural_object_list']['occluders']) == 0:
                    # TEMP; NEED TO HANDLE PB OUTPUT
                    _, obj_traj_orn = pybullet_utilities.render_in_pybullet(step_output_dict)
                    pb_state = 'complete'
            else:
                new_obj_in_scene = False
                for obj in self.track_info.values():
                    if len(obj['position']) == 3:
                        new_obj_in_scene = True
                        break
                if new_obj_in_scene:
                    # TEMP; NEED TO HANDLE PB OUTPUT
                    _, obj_traj_orn = pybullet_utilities.render_in_pybullet(step_output_dict)
                    pb_state = 'complete'
            
            choice = plausible_str(False)
            voe_heatmap = np.ones((600, 400))
            
            # if pb_run:
            # TODO: Calculate distance between actual position (Unity) and expected position (PB)
            if pb_state == 'complete' and ('pole' not in step_output_dict['structural_object_list'].keys() or (len(pole_history) > 1 and drop_step != -1)):
                obj_confidence = dict()
                # Calculate confidence
                for obj_id, obj in self.track_info.items():
                    unity_traj = [[x['x'], x['y'], x['z']] for x in obj['position']]

                    if len(unity_traj) > 2 and unity_traj[-1] != unity_traj[-2]:
                        obj_confidence[obj_id] = 1.0
                    try:
                        distance, _ = self.calc_simulator_voe(obj_traj_orn['default'][obj_id]['pos'], unity_traj)
                        obj_confidence[obj_id] = 100 * np.tanh(1 / (distance + 1e-9))
                    except KeyError as e:
                        console.log(e)
                        obj_confidence[obj_id] = 1.0
            
                    # TODO: If distance is sufficiently large, raise Position VoE
                    # confidence has to be bounded between 0 and 1
                    if obj_confidence[obj_id] >= 1:
                        obj_confidence[obj_id] = 1.0
                        # if confidence is 1, throw a no voe signal in the pixels, or the object in unity hasn't stopped moving
                        voe_xy_list.append(
                            {
                                "x": -1, # no voe 
                                "y": -1  # noe voe
                            }
                        )
                    else:
                        p_c = step_output_dict["object_list"][obj_id]["pixel_center"]
                        voe_xy_list.append(
                            {
                                "x": p_c[0],
                                "y": p_c[1] 
                            }
                        )
                        voe_heatmap = np.float32(step_output_dict['object_list'][obj_id]['obj_mask'])
                        voe_heatmap[np.all(1.0 == voe_heatmap, axis=-1)] = obj_confidence[obj_id]
                        voe_heatmap[np.all(0 == voe_heatmap, axis=-1)] = 1.0

                    # console.log(confidence)
                    if obj_confidence[obj_id] <= 0.5:
                        choice = plausible_str(True)

                    # TODO: If object not found in Unity but is expected in PB: raise Presence VoE
                
                # TODO: Make step prediction
                self.controller.make_step_prediction(
                    choice=choice, confidence=min([c for c in obj_confidence.values()]), violations_xy_list=voe_xy_list[-1],
                    heatmap_img=voe_heatmap)
            else:   # Not enough info to make a prediction on yet
                self.controller.make_step_prediction(
                    choice=choice, confidence=1.0, violations_xy_list=[{"x": -1, "y": -1}],
                    heatmap_img=voe_heatmap)


        # TODO: Make final scene-wide prediction
