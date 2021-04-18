from dataclasses import dataclass
from gravity import pybullet_utilities
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from vision.gravity import L2DataPacketV2
import sys
import cv2

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

class GravityAgent:
    def __init__(self, controller, level):
        self.controller = controller
        self.level = level

    @staticmethod
    def _determine_drop_step(pole_dimension_history):
        '''
        Find the precise point when the suction is off.
        '''
        
        # Based on an assumption that the pole color changes after the drop (suction off)
        

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
    def target_obj_id(step_output):
        if len(list(step_output['object_list'].keys())) > 0:
            return list(step_output['object_list'].keys()).pop()
        return None

    @staticmethod
    def struc_obj_ids(step_output):
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

    def sense_voe(self, drop_step, support_coords, target_trajectory, physics_flag):
        '''
        Assumptions:
        -> Objects are assumed to be rigid with uniform mass density
        -> Supporting object is assumed to be at rest
        -> Law of conservation of energy is ignored
        -> Accn. due to gravity & target object velocity are along the "y" direction
        '''

        # Surface states when the target is (possibly) placed on support
        target, support, target_end = self.states_during_and_after_drop(
            drop_step, target_trajectory, support_coords
        )
        # Determine if target should rest
        target_should_rest = self.target_should_be_stable(target, support)

        # Now verify if the target's final state is consistent with the above
        target_actually_rested = target == target_end

        return (target_should_rest ^ target_actually_rested) or physics_flag # return xor of three flags, if return = True, there was a voe

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

    def run_scene(self, config, desc_name):
        if DEBUG:
            print("DEBUG MODE!")

        '''
        # switchs some plausible scene to implausible
        for o in config["objects"]:
            if o["id"] == "target_object":
                    for step in o["togglePhysics"]:
                        step["stepBegin"] *= 100
        '''

        self.controller.start_scene(config)

        # Inputs to determine VoE
        targ_pos = []
        target_trajectory = []
        pole_history = []  # To determine drop step
        support_coords = None

        # initial variables
        obj_traj_orn = None # list of trajectories and orientations for objects in scene
        step_output = None # output of controller.step(), type varies between eval level
        step_output_dict = None # dict version of step_output
        drop_step = -1 # determined during execution of unity scene
        voe_xy_list = [] # list of pixels with potential voes 
        pb_state = "incomplete" # flag set to "complete" after we've executed pybullet the first time

        # for each action IE for each timestep
        for i, x in enumerate(config['goal']['action_list']):
            step_output = self.controller.step(action=x[0]) # get the step output
            
            if step_output is None:
                break

            if self.level == "oracle":
                step_output_dict = dict(step_output) # convert oracle data to dictionary
                
                floor_object = "floor"  # Not a dynamic object ID
                
                # use vision module to grab masks and rgbd
                try:
                    step_output = L2DataPacketV2(step_number=i, step_meta=step_output)
                except Exception as e:
                    print("Couldn't process step i+{}, skipping ahead".format(i))
                    print(e)
                    continue # skip this step because vision system failed
                
                # get dynamic target, support, and pole IDs
                target_object = self.target_obj_id(step_output_dict)
                supporting_object, pole_object = self.struc_obj_ids(step_output_dict)
            else:
                # use vision module to grab masks and rgbd
                try:
                    step_output = L2DataPacketV2(step_number=i, step_meta=step_output)
                except Exception as e:
                    print("Couldn't process step i+{}, skipping ahead".format(i))
                    print(e)
                    continue # skip this step because vision system failed
                
                # convert vision model representations into dictionary
                step_output_dict = self.convert_l2_to_dict(step_output)
                
                # static object ids
                floor_object = "floor"
                supporting_object = "support"
                pole_object = "pole"
                target_object = "target"
            
            # in each step we try to get the dimensions of the objects we are tracking
            # the pole and target objects may not be in the frame at this time, so we only collect data we can see at the moment
            # we could easily add a placeholder value for these lists to indicate they were out of frame at this point
            try:
                # Expected to not dissapear until episode ends
                support_coords = step_output_dict["structural_object_list"][supporting_object]["dimensions"]
                floor_coords = step_output_dict["structural_object_list"][floor_object]["dimensions"]
                
                # Target / Pole may not appear in view yet, start recording when available
                target_trajectory.append(step_output_dict["object_list"][target_object]["dimensions"])
                step_output_dict["object_list"][target_object]["pixel_center"] = step_output.target.centroid_px
                
                # add current target position to list of all positions over time
                targ_pos.append(step_output_dict["object_list"][target_object]["position"])
                
                # different levels have different expectations for what the pole history looks like 
                # this is because we determine the drop step differently between these two levels
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

            # if the pole has been seen before, try to determine the drop step
            if len(pole_history) != 0:
                drop_step = self.determine_drop_step(pole_history)
                
                # if we have enough data to find the drop step, and the pb simulation hasn't been run before
                if drop_step != -1 and pb_state != "complete":
                    # run pybullet simulation and get trajectories for each object present in the drop step
                    obj_traj_orn = pybullet_utilities.render_in_pybullet(step_output_dict, target_object, supporting_object, self.level)
                    pb_state = "complete"
            
            # default values for output to MCS controller at the end of each step
            choice = plausible_str(False)
            pixel_coordinates = [300, 200]
            voe_heatmap = np.ones((600, 400))

            # if we we have run the simulation, compute confidence and voe pixels 
            if pb_state == "complete":
                # calc confidence
                unity_traj = [[x["x"], x["z"], x["y"]] for x in targ_pos]
                if unity_traj[-1] != unity_traj[-2]: #if the target pos hasn't changed between steps, assuming everything is okay
                    confidence = 1.0
                elif obj_traj_orn != None:
                    # if the target object has moved calculate distance between unity sim and pybullet sim object trajectories 
                    distance, path = self.calc_simulator_voe(obj_traj_orn[target_object]['pos'], unity_traj)
                    confidence = 100 * np.tanh(1 / distance)

                # confidence has to be bounded between 0 and 1
                if confidence >= 1:
                    confidence = 1.0
                    # if confidence is 1, throw a no voe signal in the pixels, or the object in unity hasn't stopped moving
                    voe_xy_list.append(
                        {
                            "x": -1, # no voe 
                            "y": -1  # noe voe
                        }
                    )
                else:
                    # get the pixel center of the target object and output the voe_heatmap using this as our expected voe location
                    p_c = step_output_dict["object_list"][target_object]["pixel_center"]
                    voe_xy_list.append(
                        {
                            "x": p_c[0],
                            "y": p_c[1] 
                        }
                    )
                    voe_heatmap = np.float32(step_output.target.obj_mask)
                    voe_heatmap[np.all(1.0 == voe_heatmap, axis=-1)] = confidence
                    voe_heatmap[np.all(0 == voe_heatmap, axis=-1)] = 1.0

                # for each step, we assume the scene is plausible or not based on confidence calculated above
                if confidence <= 0.5:
                    choice = plausible_str(True)

                self.controller.make_step_prediction(
                    choice=choice, confidence=confidence, violations_xy_list=voe_xy_list[-1],
                    heatmap_img=voe_heatmap)
            else: # drop step hasn't happened yet
                self.controller.make_step_prediction(
                    choice=choice, confidence=1.0, violations_xy_list=[{"x": -1, "y": -1}],
                    heatmap_img=voe_heatmap)

        physics_voe_flag = None
        final_confidence = 0

        # at the end of the scene, calculate final confidence and determine if the scene contains a voe
        if obj_traj_orn != None:
            # get the inverse distance as confidence in dynamics of scene
            unity_traj = [[x["x"], x["z"], x["y"]] for x in targ_pos[drop_step:]]
            distance, path = self.calc_simulator_voe(obj_traj_orn[target_object]['pos'], unity_traj)
            
            if distance != 0:
                final_confidence = 100 * np.tanh(1 / distance)
                if final_confidence > 1.0:
                    final_confidence = 1.0
            else:
                final_confidence = 1.0
            
            # calculate if unity target object is resting on support
            target_dims = self.getMinMax(step_output_dict["object_list"][target_object])
            unity_support_position = list(step_output_dict["structural_object_list"][supporting_object]['position'].values())
            unity_target_on_support = self.getIntersectionOrContact(step_output_dict["object_list"][target_object], step_output_dict["structural_object_list"][supporting_object])
            
            # determine if on floor 
            unity_target_on_floor = self.getIntersectionOrContact(step_output_dict["object_list"][target_object], step_output_dict["structural_object_list"][floor_object])
                
            # if unity target isn't on the floor or the support, its floating, automatic voe
            unity_target_floating = False
            if not unity_target_on_floor and not unity_target_on_support:
                unity_target_floating = True
                final_confidence = 0

            #calculate if pybullet object is resting on support
            pb_support_position = obj_traj_orn[supporting_object]['pos'][-1]
            pb_target_on_support = obj_traj_orn[target_object]["support_contact"][-1] != ()
            pb_target_on_floor = obj_traj_orn[target_object]["floor_contact"][-1] != () 
            
            # this means target is on floor and touching support, not on support
            if pb_target_on_floor and pb_target_on_support:
                pb_target_on_support = not pb_target_on_support

            if unity_target_on_floor and unity_target_on_support:
                unity_target_on_support = not unity_target_on_support

            # if simulators are in agreement on the object being on or below the support
            on_floor_agreement = not (unity_target_on_floor ^ pb_target_on_floor) # 1 is good
            on_support_agreement = not (unity_target_on_support ^ pb_target_on_support) # 1 is good

            if (on_floor_agreement or on_support_agreement) and not unity_target_floating:
                print("Physics Sim Suggests no VoE for", config['name'])
                physics_voe_flag = False
            else:
                print("Physics Sim Suggests VoE for",config['name'])
                physics_voe_flag = True
            print("plausability of unity scene: ", final_confidence)

        voe_flag = physics_voe_flag

        if voe_flag:
            print("VoE observed for", config["name"])
        else:
            print("No violation for", config["name"])

        self.controller.end_scene(choice=plausible_str(voe_flag), confidence=final_confidence)
        return True

    def getIntersectionOrContact(self, obj1, obj2):
        obj1_dims = self.getMinMax(obj1)
        obj1_dims = [(obj1_dims[i][0] - 0.05, obj1_dims[i][1] + 0.05) for i in range(len(obj1_dims))]
        obj2_dims = self.getMinMax(obj2)
        obj2_dims = [(obj2_dims[i][0] - 0.05, obj2_dims[i][1] + 0.05) for i in range(len(obj2_dims))]

        x_check = (obj1_dims[0][0] <= obj2_dims[0][1] and obj1_dims[0][1] >= obj2_dims[0][0])
        z_check = (obj1_dims[1][0] <= obj2_dims[1][1] and obj1_dims[1][1] >= obj2_dims[1][0])
        y_check = (obj1_dims[2][0] <= obj2_dims[2][1] and obj1_dims[2][1] >= obj2_dims[2][0])

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

def plausible_str(violation_detected):
    return 'implausible' if violation_detected else 'plausible'
