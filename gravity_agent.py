import pdb
from typing import Tuple
import numpy as np
from dataclasses import dataclass, field

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
    def determine_drop_step(pole_state_history):
        '''
        Find the precise point when the suction is off.
        '''
        # Based on an assumption that the pole color changes after the drop (suction off)

        pole_color_history = [md["texture_color_list"][0] for md in pole_state_history]
        init_color = pole_color_history[0]

        for idx, color in enumerate(pole_color_history):
            if color != init_color:
                return idx - 1
        else:
            raise(Exception("Drop step not detected by observing color of pole"))

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

    def sense_voe(self, drop_step, support_coords, target_trajectory):
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

        return target_should_rest ^ target_actually_rested

    def run_scene(self, config, desc_name):
        if DEBUG:
            print("DEBUG MODE!")
        self.controller.start_scene(config)

        # Inputs to determine VoE
        target_trajectory = []
        pole_states = []  # To determine drop step
        support_coords = None

        for i, x in enumerate(config['goal']['action_list']):
            step_output = self.controller.step(action=x[0])

            if step_output is None:
                break
            else:
                step_output = dict(step_output)

            # Collect observations
            if support_coords is None:
                support_coords = step_output["object_list"]["supporting_object"]["dimensions"]
            
            try:
                target_trajectory.append(step_output["object_list"]["target_object"]["dimensions"])
                pole_states.append(step_output["structural_object_list"]["pole_object"])
            except KeyError:  # Object / Pole is not in view yet
                pass
            
            choice = plausible_str(True)
            voe_xy_list = []
            voe_heatmap = None
            self.controller.make_step_prediction(
                choice=choice, confidence=1.0, violations_xy_list=voe_xy_list,
                heatmap_img=voe_heatmap)

        drop_step = self.determine_drop_step(pole_states)
        voe_flag = self.sense_voe(drop_step, support_coords, target_trajectory)

        if voe_flag:
            print("VoE!")
        else:
            print("No VoE.")

        self.controller.end_scene(choice=plausible_str(voe_flag), confidence=1.0)
        return True

    def calc_voe(self, step_output, frame_num, scene_name=None):
        pass

def plausible_str(violation_detected):
    return 'implausible' if violation_detected else 'plausible'
