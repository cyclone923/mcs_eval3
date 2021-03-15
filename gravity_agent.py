from dataclasses import dataclass
import pdb
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from vision.gravity import L2DataPacketV2

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

    def __repr__(self) -> str:

        out_str = ""
        for cpt in self.corners:
            out_str += f'({cpt["x"]}, {cpt["y"]}, {cpt["z"]}) -> '

        return out_str

class GravityAgent:
    def __init__(self, controller, level):
        self.controller = controller
        self.level = level

    @staticmethod
    def determine_drop_step(pole_color_history):
        def _bgr2gray(b, g, r):
            '''
            Formula designed to expand gap between magenta & cyan
            '''
            return 0.01 * g + 0.99 * r

        gray_values = [
            _bgr2gray(*md["color"]) 
            for md in pole_color_history
        ]
        dhistory = []
        for i in range(1, len(gray_values)):
            dc = gray_values[i] - gray_values[i - 1]
            dhistory.append(abs(dc))

        offset = dhistory.index(max(dhistory))
        drop_step = pole_color_history[offset]["step_id"]

        return drop_step + 1
        
    @staticmethod
    def _determine_drop_step(pole_centroid_history):
        '''
        Find the precise point when the suction is off.
        Assumes the direction change in pole implies suction is off.
        '''
        # Based on an assumption that the pole color changes after the drop (suction off)
        pole_y_position = [
            pt[1] for pt in pole_centroid_history if pt is not None
        ]
        none_offset = len(pole_centroid_history) - len(pole_y_position)
        
        if len(pole_y_position) == 0:
            raise(Exception("Drop step not calculated as pole was never detected."))
        elif sorted(pole_y_position, reverse=True) == pole_y_position:
            for idx in range(len(pole_centroid_history)):
                if pole_centroid_history[idx] is not None:
                    # Assumed to have recorded only pole retraction
                    return none_offset + idx  
        else:
            for idx in range(1, len(pole_y_position) - 1):
                if pole_y_position[idx] > pole_y_position[idx + 1]:
                    # First direction change of pole
                    return none_offset + idx
            else:  # pole_y_position is in ascending order all along
                if pole_y_position[-2] == pole_y_position[-1]:
                    # Pole stood still after drop
                    return none_offset + pole_y_position.index(pole_y_position[-1])
                else:
                    raise(Exception("Drop step not calculated as pole never retracted."))

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
        })
        return out.get("support"), out.get("pole")

    def run_scene(self, config, desc_name):
        if DEBUG:
            print("DEBUG MODE!")
            
        # # Filter to run specific examples :: debug help
        # specific_scenes = ["04", "08", "12"]
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
        pole_centroid_history = []  # To determine drop step
        pole_color_history = []
        support_coords, floor_coords = None, None

        for i, x in enumerate(config['goal']['action_list']):
            step_output = self.controller.step(action=x[0])

            if step_output is None:
                break
            
            try:
                # Map visuals to semantic actors
                step_output = L2DataPacketV2(step_number=i, step_meta=step_output)

                # Collect observations
                if hasattr(step_output, "pole"):
                    pole_centroid_history.append(step_output.pole.centroid)
                    pole_color_history.append({
                        "color": step_output.pole.color,
                        "step_id": i
                    })
                if hasattr(step_output, "target"):
                    target_trajectory.append(step_output.target.dims)
                support_coords = step_output.support.dims
                floor_coords = step_output.floor.dims

            except AssertionError as e:
                print(e)
                print(f"Couldn't extract states of {i}th frame, using fallback...")

            choice = plausible_str(True)
            voe_xy_list = []
            voe_heatmap = None
            self.controller.make_step_prediction(
                choice=choice, confidence=1.0, violations_xy_list=voe_xy_list,
                heatmap_img=voe_heatmap)
        
        # Run inference
        try:
            drop_step = self.determine_drop_step(pole_color_history)
            voe_flag = self.sense_voe(drop_step, support_coords, floor_coords, target_trajectory)

            if voe_flag:
                print(f"[x] VoE observed for {config['name']}")
            else:
                print(f"[x] No violation for {config['name']}")

        except Exception as e:
            print(e)
            print(f"[x] Rule based agent failed on {config['name']}")

        self.controller.end_scene(choice=plausible_str(voe_flag), confidence=1.0)
        return True

    def calc_voe(self, step_output, frame_num, scene_name=None):
        pass

def plausible_str(violation_detected):
    return 'implausible' if violation_detected else 'plausible'
