from dataclasses import dataclass
from physicsvoe.data.data_gen import convert_output
from physicsvoe import occlude
from physicsvoe.framewisevoe import AppearanceViolation, EntranceViolation, PositionViolation
from physicsvoe.timer import Timer
from physicsvoe.data.types import make_camera
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from gravity import pybullet_utilities, gravity_utilities
from vision.physics import L2DataPacket
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

APPEARANCE_PLAUSIBILITY_THRESHOLD = 0.5 # The threshold at which to raise an appearance VoE if the plausibility score of the appearance match is below this threshold
POSITION_PLAUSIBILITY_THRESHOLD = 0.5   # The threshold at which to raise a position VoE if the plausibility score of the object's position is below this threshold
OBJ_TIME_TO_PASS_THROUGH = 3    # How long to let the object pass through the scene before triggering PyBullet

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
    
    # @staticmethod
    # def get_object_bounding_simplices(dims):
    #     y_coords = [pt["y"] for pt in dims]
    #     min_y, max_y = min(y_coords), max(y_coords)

    #     # Assuming "nice" placement with cubes
    #     # For generalizing, calculate convex hull and findout extreme simplices 
    #     bottom_face = ObjectFace(corners=[pt for pt in dims if pt["y"] == min_y ])
    #     top_face = ObjectFace(corners=[pt for pt in dims if pt["y"] == max_y ])

    #     return top_face, bottom_face

    # def states_during_and_after_drop(self, drop_step, target_trajectory, support):

    #     # Assuming target is moving along "y"
    #     target_drop_coords = target_trajectory[drop_step]
    #     _, target_bottom_face = self.get_object_bounding_simplices(target_drop_coords)
        
    #     target_resting_coords = target_trajectory[-1]
    #     _, target_bottom_face_end_state = self.get_object_bounding_simplices(target_resting_coords)

    #     support_top_face, _ = self.get_object_bounding_simplices(support)

    #     return target_bottom_face, support_top_face, target_bottom_face_end_state


    # @staticmethod
    # def target_should_be_stable(target, support):

    #     support_x_range = (min(pt["x"] for pt in support.corners), max(pt["x"] for pt in support.corners))
    #     support_y_range = (min(pt["y"] for pt in support.corners), max(pt["x"] for pt in support.corners))

    #     target_x_inrange = support_x_range[0] <= target.centroid[0] <= support_x_range[1]
    #     target_y_inrange = support_y_range[0] <= target.centroid[1] <= support_y_range[1]

    #     return target_x_inrange and target_y_inrange

    @staticmethod
    def target_obj_ids(step_output):
        if len(list(step_output['object_list'].keys())) > 0:
            return list(step_output['object_list'].keys())
        return None

    @staticmethod
    def get_actor_ids(step_output):
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
        return [out.get("support"), out.get("pole"), out.get("occluder")]
    
    def convert_meta_to_dict(self, metadata, level):
        step_output_dict = {
            "camera_field_of_view": 42.5,
            "camera_aspect_ratio": (600, 400),
            "structural_object_list": {
                # "floor": {
                #     "dimensions": metadata.floor.dims,
                #     "position": metadata.floor.centroid,
                #     "color": metadata.floor.color,
                #     "shape": metadata.floor.kind
                # }
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
                step_output_dict["object_list"][i] = {
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
                    "obj_mask": target.obj_mask if level != 'level1' else None
                }

        return step_output_dict
    
    def calc_simulator_voe(self, pybullet_traj, unity_traj):
        # calculate the difference in trajectory as the sum squared distance of each point in time
        
        distance, path = fastdtw(pybullet_traj, unity_traj, dist=euclidean)
        return distance, path
    
    # TODO: TRACK OBJECTS. Maybe this gets moved to another tracker module file
    def track_objects(self, step_output_dict, actor_objects, masks):
        # self.track_info values required by agent:
        # - position: a list of {'x': num, 'y': num, 'z': num} dictionaries storing information on the 3D location of the object
        # - appearance_voe: a boolean value denoting if the tracker has sensed a sufficiently robust appearance mismatch to raise an appearance VoE
        # - appearance_match_conf: a list of numbers indicating how confident the tracker is in the appearance matching between frames
        # - simulated: a boolean value indicating if the object has been simulated by PyBullet (the tracker module doesn't need to worry about this. It should just initialize this flag to False for new objects when they're detected by the tracker)

        ## TEMP: OLD LOGIC TO CONFIRM AGENT WORKS WITHOUT FAILING ##
        self.track_info = track.track_objects(masks, self.track_info)
        for obj in self.track_info['objects'].values():
            obj['position'] = [{'x': pos['x'], 'y': pos['y'], 'z': 1} for pos in obj['position_history']]
            obj['appearance_voe'] = False
            obj['appearance_match_conf'] = [1.0]
            if 'simulated' not in obj.keys():
                obj['simulated'] = False
    
    def run_scene(self, config, desc_name):
        if DEBUG:
            folder_name: Path = Path(self.prefix)/Path(Path(desc_name).stem)
            if folder_name.exists():
                return None
            folder_name.mkdir(parents=True)
            print(folder_name)
        else:
            folder_name: Path = None

        self.controller.start_scene(config)
        self.track_info: dict[int, dict] = dict()
        scene_voe_detected: bool = False
        all_vios: list = list()

        # THIS IS TEMPORARY TO ENSURE AGENT IS WORKING
        if self.level == 'level1':
            self.visionmodel = vision.MaskAndClassPredictor(dataset='mcsvideo3_voe',
                                                            config='plus_resnet50_config_depth_MC',
                                                            weights=VISION_MODEL_PATH)

        object_sims = None

        for i, pos in enumerate(config['goal']['action_list']):
            frame_vios: list = list()
            step_output = self.controller.step(action=pos[0])  # Get the step output
            if step_output is None:
                break
            
            # TEMPORARY FROM EVAL 3 TO ENSURE AGENT IS WORKING
            depth_map = step_output.depth_map_list[-1]
            rgb_image = step_output.image_list[-1]
            if self.level == 'oracle':
                masks = self.oracle_masks(step_output)
            elif self.level == 'level2':
                in_mask = step_output.object_mask_list[-1]
                masks = self.level2_masks(depth_map, rgb_image, in_mask)
            elif self.level == 'level1':
                masks = self.level1_masks(depth_map, rgb_image)
            else:
                raise ValueError(f'Unknown level `{self.level}`')

            # Identify actor objects
            # Oracle
            if self.level == 'oracle':
                step_output_dict: dict = dict(step_output)
                try:
                    step_output: L2DataPacket = L2DataPacket(step_number=i, step_meta=step_output, scene=desc_name)
                except Exception as e:
                    console.log("Couldn't process step i+{}, skipping ahead".format(i))
                    console.log(e)
                    continue
            
            # Level 2
            elif self.level == 'level2':
                try:
                    step_output: L2DataPacket = L2DataPacket(step_number=i, step_meta=step_output, scene=desc_name)
                    step_output_dict = self.convert_meta_to_dict(step_output, self.level)
                except Exception as e:
                    console.log("Couldn't process step i+{}, skipping ahead".format(i))
                    console.log(e)
                    continue
            
            # Level 1
            else:
                step_output_dict = self.convert_meta_to_dict(step_output, self.level)
                try:
                    # TODO: L1DataPacket (since we won't have everything thing Level 2 has)
                    step_output: L2DataPacket = L2DataPacket(step_number=i, step_meta=step_output, scene=desc_name)
                except Exception as e:
                    console.log("Couldn't process step i+{}, skipping ahead".format(i))
                    console.log(e)
                    continue
            actor_objects: list = self.get_actor_ids(step_output_dict)  # Extract IDs of actor objects

            obj_appearance_plausibility: dict[int, float] = dict()  # plausibility score for each object
            try:
                # TODO: TRACK OBJECTS (MOTION AND APPEARANCE)
                self.track_objects(step_output_dict, actor_objects, masks)

                # CHECK TRACK INFO FOR ANY APPEARANCE MATCHES
                for obj_id, obj in self.track_info['objects'].items():
                    # check if appearance VoE flag has been set for this object
                    if i > 0 and obj['appearance_voe']:
                        obj_xy_pos: dict = {'x': obj['position'][-1]['x'], 'y': obj['position'][-1]['y']}
                        obj_appearance_plausibility[obj_id] = obj['appearance_match_conf'][-1]
                        frame_vios.append(AppearanceViolation(obj_id, obj_xy_pos, obj_appearance_plausibility))
            except KeyError as e:   # Object is not in view yet
                console.log('key error', e)
                pass
            except AttributeError as e:
                console.log('attribute error', e)
                pass

            # RUN PYBULLET IF NEW OBJECT IN SCENE
            # if any objects are new and have NOT been simulated by PyBullet yet, run PyBullet
            if len([o for o in self.track_info['objects'].values() if len(o['position']) == OBJ_TIME_TO_PASS_THROUGH and not o['simulated']]) > 0:
                new_obj_velocity = {}
                for obj_id, obj in self.track_info['objects'].items():
                    if len(obj['position']) == OBJ_TIME_TO_PASS_THROUGH:
                        # transform unity position to pybullet position
                        initial_position = list(obj['position'][0].values())
                        initial_position = np.array([initial_position[1], initial_position[2], initial_position[0]])
                        
                        current_position = list(obj['position'][-1].values())
                        current_position = np.array([current_position[1], current_position[2], current_position[0]])

                        new_vel = (current_position - initial_position) / OBJ_TIME_TO_PASS_THROUGH ## average velocity of object
                        
                        # TODO: this is temporary because we are using pixel coordinates in track_info
                        new_vel[-1] = -1 * new_vel[-1]
                        # if velocity is strictly free-fall - based on the assumption that it is being manually lowered
                        if new_vel[2] < 0 and new_vel[0] == 0.0 and new_vel[1] == 0.0:
                            new_vel[2] = 0.0 # set vertical velocity to none,  

                        new_obj_velocity[obj_id] = new_vel



                console.log(new_obj_velocity)
                _, object_sims = pybullet_utilities.render_in_pybullet(step_output_dict, new_obj_velocity)
                for obj_id, obj in self.track_info['objects'].items():
                    if obj_id in object_sims.keys() and not obj['simulated']:
                        obj['simulated'] = True

            # CHECK FOR ANY POSITION-RELATED VoEs
            if object_sims:
                obj_pos_plausibility: dict[int, float] = dict()
                for obj_id, obj in self.track_info['objects'].items():
                    unity_trajectory = [[pos['x'], pos['y'], pos['z']] for pos in obj['position']]  # Get actual object trajectory
                    unity_xy_pos = {'x': obj['position'][-1]['x'], 'y': obj['position'][-1]['y']}   # Get actual object position

                    # TODO: CHECK FOR PRESENCE VoEs
                    try:
                        distance, _ = self.calc_simulator_voe(object_sims[obj_id]['pos'], unity_trajectory) # Calculate difference in actual and simulated trajectory
                        obj_pos_plausibility[obj_id] = 100 * np.tanh(1 / (distance + 1e-9))
                        if obj_pos_plausibility[obj_id] < POSITION_PLAUSIBILITY_THRESHOLD:
                            frame_vios.append(PositionViolation(obj_id, unity_xy_pos, obj_pos_plausibility[obj_id]))
                    except KeyError as e:
                        # TODO: OBJECT NOT IN PYBULLET SIMULATION; CHECK IF OBJECT IS OCCLUDED BY ACTOR OBJECT
                        # IF OCCLUDED BY ACTOR OBJECT, CONF = 1.0. ELSE, RAISE ENTRANCE VOE
                        obj_pos_plausibility[obj_id] = 1.0
                        if obj_pos_plausibility[obj_id] < POSITION_PLAUSIBILITY_THRESHOLD:
                            frame_vios.append(EntranceViolation(obj_id, unity_xy_pos, obj_pos_plausibility[obj_id]))

            # MAKE STEP PREDICTION
            # choice = plausible_str(True)
            camera_aspect_ratio: tuple[int, int] = step_output_dict['camera_aspect_ratio'] if 'camera_aspect_ratio' in step_output_dict.keys() else (600, 400)
            voe_heatmap = np.ones(camera_aspect_ratio)

            all_plausibility_values: list[float] = [voe.conf for voe in frame_vios]
            frame_confidence: float = 1.0 if len(all_plausibility_values) == 0 else min(all_plausibility_values)
            if len(frame_vios) == 0:
                # If no frame vios, all confidence values will be >= POSITION_PLAUSIBILITY_THRESHOLD
                self.controller.make_step_prediction(
                    choice=plausible_str(True),
                    confidence=frame_confidence,
                    violations_xy_list=[{'x': -1, 'y': -1}],
                    heatmap_img=voe_heatmap
                )
            else:
                # If there are frame vios, at least one plausibility value will be < POSITION_PLAUSIBILITY_THRESHOLD
                # TODO: Build VoE heatmap that accounts for ALL violations
                for voe in frame_vios:
                    pass
                self.controller.make_step_prediction(
                    choice=plausible_str(False),
                    confidence=frame_confidence,
                    violations_xy_list=[voe.pos for voe in frame_vios],
                    heatmap_img=voe_heatmap
                )
            
            all_vios.extend(frame_vios)
            if not scene_voe_detected and len(frame_vios) > 0:
                scene_voe_detected = True
        
        # MAKE SCENE PREDICTION
        scene_plausibility_values: list[float] = [voe.conf for voe in all_vios]
        scene_confidence = 1.0 if len(scene_plausibility_values) == 0 else min(scene_plausibility_values)
        self.controller.end_scene(
            choice=plausible_str(False if scene_voe_detected else True),
            confidence=scene_confidence
        )
        if DEBUG:
            with open(folder_name/'viols.pkl', 'wb') as fd:
                pickle.dump(all_vios, fd)
        return scene_voe_detected


    @staticmethod
    def oracle_masks(step_output):
        frame = convert_output(step_output)
        return frame.obj_mask

    @staticmethod
    def level2_masks(depth_img, rgb_img, mask_img):
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

    # def run_scene(self, config, desc_name):
    #     if DEBUG:
    #         folder_name = Path(self.prefix)/Path(Path(desc_name).stem)
    #         if folder_name.exists():
    #             return None
    #         folder_name.mkdir(parents=True)
    #         print(folder_name)
    #     else:
    #         folder_name = None
        
    #     self.detector = \
    #         framewisevoe.FramewiseVOE(min_hist_count=3, max_hist_count=8,
    #                                   dist_thresh=0.5)
    #     self.controller.start_scene(config)
    #     pole_history = []  # To determine drop step
    #     self.track_info = dict()
    #     scene_voe_detected = False
    #     all_viols = []
    #     all_errs = []
    #     voe_xy_list = list()

    #     obj_traj_orn = None

    #     pb_state = 'incomplete'
    #     for i, x in enumerate(config['goal']['action_list']):
    #         step_output = self.controller.step(action=x[0]) # Get the step output
    #         # Get details of the objects in the scene
    #         if self.level == "oracle":
    #             if step_output is None:
    #                 break
    #             else:
    #                 step_output_dict = dict(step_output)
    #                 try:
    #                     step_output = L2DataPacketV2(step_number=i, step_meta=step_output, scene=desc_name)
    #                 except Exception as e:
    #                     print("Couldn't process step i+{}, skipping ahead".format(i))
    #                     print(e)
    #                     continue

    #             floor_object = "floor"  # Not a dynamic object ID
    #             # console.log(step_output_dict)
    #             target_objects = self.target_obj_ids(step_output_dict)
    #             # console.log(target_objects)
    #             supporting_object, pole_object, occluder_objects = self.struc_obj_ids(step_output_dict)
    #         else:
    #             if step_output is None:
    #                 break
                
    #             try:
    #                 step_output = L2DataPacketV2(step_number=i, step_meta=step_output, scene=desc_name)
    #             except Exception as e:
    #                 print("Couldn't process step i+{}, skipping ahead".format(i))
    #                 print(e)
    #                 continue
                
    #             # convert L2DataPacketV2 to dictionary
    #             step_output_dict = self.convert_l2_to_dict(step_output)
                
    #             floor_object = "floor"
    #             target_objects = self.target_obj_ids(step_output_dict)
    #             # console.log(target_objects)
    #             supporting_object, pole_object, occluder_objects = self.struc_obj_ids(step_output_dict)
            
    #         # console.log('check')
    #         # Track each object's trajectory and position through the scene
    #         try:
    #             # Expected to not dissapear until episode ends
    #             # support_coords = step_output_dict["structural_object_list"][supporting_object]["dimensions"]
    #             # floor_coords = step_output_dict["structural_object_list"][floor_object]["dimensions"]

    #             # Target / Pole may not appear in view yet, start recording when available
    #             for obj_id, obj in step_output_dict['object_list'].items():
    #                 if obj_id not in self.track_info.keys():
    #                     self.track_info[obj_id] = {
    #                         'color': list(),
    #                         'trajectory': list(),
    #                         'position': list()
    #                     }
    #                 self.track_info[obj_id]['trajectory'].append(obj["dimensions"])
    #                 self.track_info[obj_id]['position'].append(obj['position'])
    #                 self.track_info[obj_id]['color'].append(obj['color'])
                
    #             # if self.level == 'level2':
    #             #     pole_history.append({
    #             #             "color": step_output_dict["structural_object_list"][pole_object]['color'],
    #             #             "step_id": i
    #             #         })
    #             # elif self.level == "oracle":
    #             #     pole_history.append(self.getMinMax(step_output_dict["structural_object_list"][pole_object]))
            
    #         except KeyError as e:  # Object / Pole is not in view yet
    #             console.log("key error", e)
    #             pass
    #         except AttributeError as e:
    #             console.log("attribute error", e)
    #             pass

    #         # TODO: PERFORM TARGET OBJECT APPEARANCE MATCH

    #         # TODO: CHECK FOR ENTRANCE VOE

    #         # TODO: RUN PYBULLET
    #         if 'pole' in step_output_dict['structural_object_list'].keys():
    #             drop_step = self.determine_drop_step(pole_history)
    #             if drop_step != -1 and pb_state != 'complete' and len(step_output_dict['structural_object_list']['occluders']) == 0:
    #                 # TEMP; NEED TO HANDLE PB OUTPUT
    #                 print("rendering in pybullet")
    #                 _, obj_traj_orn = pybullet_utilities.render_in_pybullet(step_output_dict)
    #         else:
    #             new_obj_velocity = {}
    #             for obj_id, obj in self.track_info.items():
    #                 if len(obj['position']) == 3:
    #                     initial_position = np.array(list(obj['position'][0].values()))
    #                     current_position = np.array(list(obj['position'][-1].values()))
    #                     new_obj_velocity[obj_id] = (current_position - initial_position) / 5 ## average velocity of object

    #             if len(new_obj_velocity.keys()) and len(step_output_dict['object_list']):
    #                 # TEMP; NEED TO HANDLE PB OUTPUT
    #                 print("rendering in pybullet")
    #                 _, obj_traj_orn = pybullet_utilities.render_in_pybullet(step_output_dict, velocities=new_obj_velocity)
            
    #         choice = plausible_str(False)
    #         voe_heatmap = np.ones((600, 400))
            
    #         # if pb_run:
    #         # TODO: Calculate distance between actual position (Unity) and expected position (PB)
    #         if pb_state == 'complete' and ('pole' not in step_output_dict['structural_object_list'].keys() or (len(pole_history) > 1 and drop_step != -1)):
    #             obj_confidence = dict()
    #             # Calculate confidence
    #             for obj_id, obj in self.track_info.items():
    #                 unity_traj = [[x['x'], x['y'], x['z']] for x in obj['position']]
    #                 if len(unity_traj) > 2 and unity_traj[-1] != unity_traj[-2]:
    #                     obj_confidence[obj_id] = 1.0
    #                 try:
    #                     distance, _ = self.calc_simulator_voe(obj_traj_orn['default'][obj_id]['pos'], unity_traj)
    #                     obj_confidence[obj_id] = 100 * np.tanh(1 / (distance + 1e-9))
    #                 except KeyError as e:
    #                     console.log(e)
    #                     obj_confidence[obj_id] = 1.0
            
    #                 # TODO: If distance is sufficiently large, raise Position VoE
    #                 # confidence has to be bounded between 0 and 1
    #                 if obj_confidence[obj_id] >= 1:
    #                     obj_confidence[obj_id] = 1.0
    #                     # if confidence is 1, throw a no voe signal in the pixels, or the object in unity hasn't stopped moving
    #                     voe_xy_list.append(
    #                         {
    #                             "x": -1, # no voe 
    #                             "y": -1  # noe voe
    #                         }
    #                     )
    #                 else:
    #                     if obj_id in step_output_dict['object_list']:
    #                         p_c = step_output_dict["object_list"][obj_id]["pixel_center"]
    #                         voe_xy_list.append(
    #                             {
    #                                 "x": p_c[0],
    #                                 "y": p_c[1] 
    #                             }
    #                         )
    #                         voe_heatmap = np.float32(step_output_dict['object_list'][obj_id]['obj_mask'])
    #                         voe_heatmap[np.all(1.0 == voe_heatmap, axis=-1)] = obj_confidence[obj_id]
    #                         voe_heatmap[np.all(0 == voe_heatmap, axis=-1)] = 1.0

    #                 # console.log(confidence)
    #                 if obj_confidence[obj_id] <= 0.5:
    #                     choice = plausible_str(True)

    #                 # TODO: If object not found in Unity but is expected in PB: raise Presence VoE
                
    #             # TODO: Make step prediction
    #             self.controller.make_step_prediction(
    #                 choice=choice, confidence=min([c for c in obj_confidence.values()]), violations_xy_list=voe_xy_list[-1],
    #                 heatmap_img=voe_heatmap)
    #         else:   # Not enough info to make a prediction on yet
    #             self.controller.make_step_prediction(
    #                 choice=choice, confidence=1.0, violations_xy_list=[{"x": -1, "y": -1}],
    #                 heatmap_img=voe_heatmap)


    #     # TODO: Make final scene-wide prediction

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

def plausible_str(plausible) -> str:
    return 'plausible' if plausible else 'implausible'
