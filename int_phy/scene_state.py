from int_phy.object_state import ObjectState, get_cropped_object_appearane
from int_phy.occluder_state import OccluderState
import sys
from int_phy.explain import *
from int_phy_recollect_appearance import SHAPE_TYPES
from matplotlib.lines import Line2D


class SceneState:

    def __init__(self, step_output, plot=False):
        self.static_object_set = set()
        for obj in step_output.object_list:
            self.static_object_set.add(obj.uuid)

        self.object_state_dict = {}
        self.occluder_state_dict = {}

        self.depth_frame = step_output.depth_map_list[-1]
        self.object_frame = step_output.object_mask_list[-1]
        self.frame_size = self.object_frame.size
        self.agent_position = step_output.position

        self.plot = plot
        if self.plot:
            plt.cla()
            plt.xlim((-6,6))
            plt.ylim(-1,5)
            legend_elements = [
                Line2D([0], [0], marker='X', color='w', markerfacecolor='b', label='Prediction of Next Step', markersize=5),
                Line2D([0], [0], marker='X', color='w', markerfacecolor='r', label='Next Step Ground Truth', markersize=5)
            ]
            plt.legend(handles=legend_elements, prop={'size': 6})


    def get_new_object_state_dict(self, new_step_output, new_depth_frame, new_object_frame):
        new_object_state_dict = {}
        for obj in new_step_output.object_list:
            if obj.uuid in self.static_object_set:
                continue
            try: # it is possible to have a obj in step_output but not in object_mask_frame
                obj_state = ObjectState(obj, new_depth_frame, new_object_frame, new_step_output, self.plot)
                new_object_state_dict[obj.uuid] = obj_state
            except:
                pass
                # print("Unexpected error:\n {}".format(sys.exc_info()))
        return new_object_state_dict

    def get_new_occluder_state_dict(self, structural_object_list, new_depth_frame, new_object_frame):
        new_structrual_object_state_dict = {}
        for obj in structural_object_list:
            if "occluder_wall" not in obj.uuid:
                continue
            try: # it is possible to have a obj in step_output but not in object_mask_frame
                obj_state = OccluderState(obj, new_depth_frame, new_object_frame)
                new_structrual_object_state_dict[obj.uuid] = obj_state
            except:
                pass
                # print("Unexpected error:\n {}".format(sys.exc_info()))
        return new_structrual_object_state_dict

    def update(self, new_step_output, appearance_checker, locomotion_checker):
        # print('-'*40)
        new_depth_frame = new_step_output.depth_map_list[-1]
        new_object_frame = new_step_output.object_mask_list[-1]

        new_object_state_dict = self.get_new_object_state_dict(
            new_step_output, new_depth_frame, new_object_frame
        )

        new_occluder_state_dict = self.get_new_occluder_state_dict(
            new_step_output.structural_object_list, new_depth_frame, new_object_frame
        )

        for id, state in new_object_state_dict.items():
            if id not in self.object_state_dict: # object appearance
                # print("object {} first appears".format(id))
                self.object_state_dict[id] = state

        for id, state in self.object_state_dict.items():
            if id not in new_object_state_dict:
                # print("object {} disappears".format(id))# object disappearance
                self.object_state_dict[id].loc_out_view_update(new_occluder_state_dict, locomotion_checker)
            else:
                self.object_state_dict[id].loc_in_view_update(new_object_state_dict[id], locomotion_checker)

            if not check_object_patially_occlusion(new_occluder_state_dict, state, new_object_state_dict):
                if not check_object_on_edge(state, self.frame_size):
                    cropped_object_frame = get_cropped_object_appearane(new_object_frame, state.edge_pixels, state.color)
                    decision, likelihoods = appearance_checker.check_appearance(cropped_object_frame, SHAPE_TYPES)
                    self.object_state_dict[id].appearance_update(decision, likelihoods)

        self.object_frame = new_object_frame
        self.depth_frame = new_depth_frame

    def get_scene_appearance_scrore(self):
        min_object_appearance_score = 1
        for obj_state in self.object_state_dict.values():
            obj_appearance_score = obj_state.get_appearance_score()
            min_object_appearance_score = min(obj_appearance_score, min_object_appearance_score)
        return (min_object_appearance_score - 0.5) * 2


    def get_scene_locomotion_score(self):
        min_object_locomotion_score = 1
        for obj_state in self.object_state_dict.values():
            obj_locomotion_score = obj_state.get_locomotion_score()
            min_object_locomotion_score = min(obj_locomotion_score, min_object_locomotion_score)
        return min_object_locomotion_score

    def get_final_score(self):
        return min(self.get_scene_locomotion_score(), self.get_scene_appearance_scrore())
