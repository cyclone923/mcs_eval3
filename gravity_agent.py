from dataclasses import dataclass
import pandas as pd
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull

DEBUG = False

@dataclass
class ObjectFace:
    corners: list

    def __post_init__(self):

        self.polygon = Polygon([(pt["x"], pt["z"]) for pt in self.corners])

        x = sum(pt["x"] for pt in self.corners) / 4
        y = sum(pt["y"] for pt in self.corners) / 4
        z = sum(pt["z"] for pt in self.corners) / 4
        self.centroid = (x, y, z)

    def on(self, o: object, tol=1e-2) -> bool:
        '''
        Returns True if `self` "appears" to be on `o`
        '''
        # Doesn't verify if method is invoked from top face and NOT bottom face
        
        def get_3d_polygon(corners):
            return Polygon([
                list(pt.values()) for pt in corners
            ])
        
        this_polygon = get_3d_polygon(self.corners)
        given_polygon = get_3d_polygon(o.corners)
        polygons_touch = this_polygon.distance(given_polygon) < tol
        
        return polygons_touch


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
        # Bounding along "y" direction

        def face_pts_from_hull(hull, target_y):
            bounding_simplices_vtx = set(hull.simplices[hull.equations[:, 1] == target_y].flatten().tolist())
            bounding_face_pts = [
                {"x": hull.points[pt][0], "y": hull.points[pt][1], "z": hull.points[pt][2]}
                for pt in bounding_simplices_vtx
            ]

            return bounding_face_pts

        ndims = [list(pt.values()) for pt in dims]
        hull = ConvexHull(ndims)

        smallest_y, largest_y = hull.equations[:, 1].min(), hull.equations[:, 1].max()
        bottom_face = ObjectFace(corners=face_pts_from_hull(hull, smallest_y))
        top_face = ObjectFace(corners=face_pts_from_hull(hull, largest_y))

        return top_face, bottom_face

    def states_during_and_after_drop(self, drop_step, target_trajectory, support, floor):

        # Assuming target is moving along "y"
        target_drop_coords = target_trajectory[drop_step]
        _, target_bottom_face = self.get_object_bounding_simplices(target_drop_coords)
        
        target_resting_coords = target_trajectory[-1]
        _, target_bottom_face_end_state = self.get_object_bounding_simplices(target_resting_coords)

        support_top_face, _ = self.get_object_bounding_simplices(support)

        floor_top_face, _ = self.get_object_bounding_simplices(floor)

        return target_bottom_face, target_bottom_face_end_state, support_top_face, floor_top_face


    @staticmethod
    def target_should_be_on_support(target, support):

        support_x_range = (min(pt["x"] for pt in support.corners), max(pt["x"] for pt in support.corners))
        support_z_range = (min(pt["z"] for pt in support.corners), max(pt["z"] for pt in support.corners))

        target_x_inrange = support_x_range[0] <= target.centroid[0] <= support_x_range[1]
        target_z_inrange = support_z_range[0] <= target.centroid[2] <= support_z_range[1]

        return target_x_inrange and target_z_inrange

    def sense_voe(self, drop_step, support_coords, floor_coords, target_trajectory):
        '''
        Assumptions:
        -> Objects are assumed to be rigid with uniform mass density
        -> Supporting object is assumed to be at rest
        -> Law of conservation of energy is ignored
        -> Accn. due to gravity & target object velocity are along the "y" direction
        '''
        # Surface states when the target is (possibly) placed on support
        target, target_end, support, floor = self.states_during_and_after_drop(
            drop_step, target_trajectory, support_coords, floor_coords
        )
        # Determine if target should rest on support
        target_expected_on_support = self.target_should_be_on_support(target, support)

        # Now verify if the target's final state is consistent with the above
        target_actually_on_support = target_end.on(support)

        # Sense if target is on floor
        target_on_floor = target_end.on(floor)

        # Target should either be on support or on floor
        target_on_support_when_it_should = target_expected_on_support and target_actually_on_support
        voe_flag = not (target_on_support_when_it_should ^ target_on_floor)

        return voe_flag

    def run_scene(self, config, desc_name):
        if DEBUG:
            print("DEBUG MODE!")
            
        # # Filter to run specific examples :: debug help
        # specific_scenes = ["01", "09"]
        # if all(code not in config["name"] for code in specific_scenes):
        #     return True
        # else:
        #     print(f"[x] Running {config['name']}")

        # # switchs some plausible scene to implausible
        # for o in config["objects"]:
        #     if o["id"] == "target_object":
        #             for step in o["togglePhysics"]:
        #                 step["stepBegin"] *= 100

        self.controller.start_scene(config)

        # Inputs to determine VoE
        target_trajectory = []
        pole_states = []  # To determine drop step
        support_coords, floor_coords = None, None

        for i, x in enumerate(config['goal']['action_list']):
            step_output = self.controller.step(action=x[0])

            if step_output is None:
                break
            else:
                step_output = dict(step_output)

            # Collect observations
            if support_coords is None:
                support_coords = step_output["object_list"]["supporting_object"]["dimensions"]

            if floor_coords is None:
                floor_coords = step_output["structural_object_list"]["floor"]["dimensions"]
            
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
        voe_flag = self.sense_voe(drop_step, support_coords, floor_coords, target_trajectory)

        if voe_flag:
            print(f"[x] VoE observed for {config['name']}")
        else:
            print(f"[x] No violation for {config['name']}")

        self.controller.end_scene(choice=plausible_str(voe_flag), confidence=1.0)
        return True

    def calc_voe(self, step_output, frame_num, scene_name=None):
        pass

def plausible_str(violation_detected):
    return 'implausible' if violation_detected else 'plausible'
