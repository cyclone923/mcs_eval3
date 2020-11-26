import os
import cv2
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement
from voe.agency_util import *

class AgencyVoeAgent:

    def __init__(self, controller, level, debug=False):
        self.controller = controller
        self.level = level
        self.debug = debug
        
    def run_scene(self, config):

        # quick and dirty way to toggle debug printing to screen
        if not self.debug:
            def silence_print(*args,**kwargs):
                return True

            print = silence_print

        M_wall, M_gnd = get_homographies()

        def step(cam_im, mask_im, info, first_frame=False):

            gnd_mask = gnd_trans(mask_im, M_gnd)
            gnd_rgb = gnd_trans(cam_im, M_gnd)
            
            trans_im = wall_trans(cam_im, M_wall)
            arena = info["arena"]
            wall_i_s = info["wall_i_s"]
            if type(arena) == type(None):
                arena = create_arena()
                _walls_im, arena, wall_i_s = find_walls(trans_im, arena)

            structural_mask_colors = []
            structural_mask_colors.append(get_gnd_mask_color(gnd_mask))
            parallel_wall_colors = []
            wall_filtered = []
            for x, y in wall_i_s:
                c = get_mask_color(gnd_mask, x, y)
                if is_wall(gnd_rgb, gnd_mask, c):
                    structural_mask_colors.append(c)
                    parallel_wall_colors.append(c)
                    wall_filtered.append((x, y))
            wall_i_s = wall_filtered
            structural_mask_colors = add_border_colors(mask_im, structural_mask_colors)

            # agent(s), object(s), home square(s), and maybe a wall occluded by an agent
            unknown_colors = get_unknowns(gnd_mask, structural_mask_colors)
            #print("Number of yet unidentified objects:", len(unknown_colors))

            agent_dict = {
                "rgb": None,
                "mask": None,
                "c": None
            }
            home_dict = {
                "rgb": None,
                "mask": None,
                "c": None,
                "center": None
            }
            obj_1_dict = {
                "rgb": None,
                "mask": None,
                "c": None,
            }
            obj_2_dict = {
                "rgb": None,
                "mask": None,
                "c": None,
            }

            # home side length in px after being cropped and homographied
            side_len = 29
            home_wiggle = 10
            a_ch_avgs = info["agent_ch_avgs"]
            agent_wiggle = 30
            for c in unknown_colors:
                if is_wall(gnd_rgb, gnd_mask, c):
                    structural_mask_colors.append(c)
                    #print("removing possible wall piece from unknowns")
                    continue

                obj_im, mask = apply_color_mask(np.copy(gnd_rgb), np.copy(gnd_mask), c, show=False)
                #cv2_show_im(obj_im)                

                if type(a_ch_avgs) == type(None):
                    a_color_dist = 0
                else:
                    # color histogram different between the agent and the current unknown object
                    a_color_dist = np.absolute(a_ch_avgs - get_ch_avgs(obj_im)).sum()
                #print(a_color_dist)

                # could definately be fine tuned to bring down the thresh below
                home_colors = [252, 49, 254]
                color_diff_to_home = np.absolute(home_colors - get_ch_avgs(obj_im)).sum()
                non_zero_c = cv2.countNonZero(obj_im.sum(axis=2))
                color_diff_to_home = color_diff_to_home / non_zero_c
                #print(color_diff_to_home, non_zero_c)
                #print("color_diff_to_home:", color_diff_to_home)            

                # home square is a certain color, height and width after homography
                l,r,t,b = get_mask_cardinals(obj_im)
                width = r-l
                height = b-t
                # can change this to per channel avg if this becomes too broad a range
                is_home = color_diff_to_home < 0.33
                is_home = is_home and abs(width-side_len) < home_wiggle and abs(height-side_len) < home_wiggle
                if is_home:
                    #if not first_frame:
                        #print("Found home square for step #" + str(info["step_num"]))
                    home_dict["rgb"] = obj_im
                    home_dict["mask"] = mask
                    home_dict["c"] = c
                    home_dict["center"] = (l + width//2, t + height//2)
                elif a_color_dist < agent_wiggle:
                    if type(agent_dict["mask"]) != type(None) and first_frame:
                        if type(obj_1_dict["mask"]) != type(None):
                            #print("Set object 2!!")
                            obj_2_dict["rgb"] = agent_dict["rgb"]
                            obj_2_dict["mask"] = agent_dict["mask"]
                            obj_2_dict["c"] = agent_dict["c"]
                        else:
                            #print("Set object 1!!")
                            obj_1_dict["rgb"] = agent_dict["rgb"]
                            obj_1_dict["mask"] = agent_dict["mask"]
                            obj_1_dict["c"] = agent_dict["c"]
                    #print("Found agent!")
                    agent_dict["rgb"] = obj_im
                    agent_dict["mask"] = mask
                    agent_dict["c"] = c
                elif type(obj_1_dict["mask"]) == type(None):
                    #print("Set object 1!")
                    obj_1_dict["rgb"] = obj_im
                    obj_1_dict["mask"] = mask
                    obj_1_dict["c"] = c
                else:
                    #print("Set object 2!")
                    obj_2_dict["rgb"] = obj_im
                    obj_2_dict["mask"] = mask
                    obj_2_dict["c"] = c

            if first_frame:
                tmp = find_and_rm_jerk_walls(gnd_rgb, gnd_mask, structural_mask_colors, agent_dict, obj_1_dict, obj_2_dict, home_dict)                    
                erase_wall_mask, gnd_mask, structural_mask_colors, agent_dict, obj_1_dict, obj_2_dict = tmp
            else:
                erase_wall_mask = info["erase_wall"]
                if type(erase_wall_mask) != type(None):
                    gnd_mask[erase_wall_mask != 0] = 255
                
            # update the agent's color histogram since moving around occludes bits of it and the lighting changes
            a_ch_avgs = get_ch_avgs(agent_dict["rgb"])

            objs_ch_avgs = info["objs_ch_avgs"]
            both_o_pos = info["trial_1_objs_pos"]
            if type(objs_ch_avgs) == type(None) and first_frame and type(obj_2_dict["c"]) != type(None):
                o_1_avgs = get_ch_avgs(obj_1_dict["rgb"])
                o_2_avgs = get_ch_avgs(obj_2_dict["rgb"])
                objs_ch_avgs = (o_1_avgs, o_2_avgs)
                o_1_x, o_1_y  = get_obj_pos(obj_1_dict["rgb"])
                o_2_x, o_2_y  = get_obj_pos(obj_2_dict["rgb"])
                both_o_pos = [(o_1_x, o_1_y), (o_2_x, o_2_y)]

            h_x, h_y = get_home_pos(home_dict["rgb"])
            a_x, a_y = get_agent_pos(agent_dict["rgb"])
            
            if first_frame:
                # we can calculate how wrong our agent_pos fx is on the first frame using home's center
                a_x_offset = h_x - a_x
                a_y_offset = h_y - a_y
            else:
                a_x_offset = info["agent_offset"][0]
                a_y_offset = info["agent_offset"][1]
            a_x, a_y = a_x + a_x_offset, a_y + a_y_offset
            agent_dict["x"] = a_x
            agent_dict["y"] = a_y

            # the default for multi-object case during trial #1 steps, but shouldn't even be used
            o_x, o_y = 0, 0
            if type(obj_2_dict["c"]) == type(None):
                o_x, o_y = get_obj_pos(obj_1_dict["rgb"])
            elif info["trial_num"] != 0 and first_frame and type(obj_2_dict["c"]) != type(None):
                # the tuple is sorted after trial #1 so the preferred object's colors is the first array of the tuple
                pref_obj_color = objs_ch_avgs[0]
                current_objs_avgs = [get_ch_avgs(obj_1_dict["rgb"]), get_ch_avgs(obj_2_dict["rgb"])]
                dist_1 = np.absolute(pref_obj_color - current_objs_avgs[0]).sum()
                dist_2 = np.absolute(pref_obj_color - current_objs_avgs[1]).sum()
                if dist_1 < dist_2:
                    o_x, o_y = get_obj_pos(obj_1_dict["rgb"])
                else:
                    o_x, o_y = get_obj_pos(obj_2_dict["rgb"])

            _a_x, _a_y, = px_to_arena((a_x, a_y), home_dict["rgb"])
            path_taken.append((_a_x, _a_y))

            trial_err = info["trial_err"]
            i = info["step_num"]
            best_path = info["path"]

            if first_frame:
                if type(obj_2_dict["c"]) == type(None) or info["trial_num"] != 0:
                    best_path = calc_path(arena, (_a_x, _a_y), px_to_arena((o_x, o_y), home_dict["rgb"]))

            # Don't know the path for multi-object cases until the end of trial #1
            if type(obj_2_dict["c"]) == type(None) or (type(obj_2_dict["c"]) != type(None) and info["trial_num"] != 0): 
                end_offset = 4
                if i >= len(best_path) - end_offset:
                    if (len(best_path)) == 0:
                        guess_x, guess_y = (0,0)
                        print("PATHING ISSUE :(")
                    else:    
                        tot_off = -1 - end_offset
                        if abs(tot_off) >= len(best_path):
                            tot_off = -1
                        guess_x, guess_y = best_path[tot_off]
                else:
                    guess_x, guess_y = best_path[i]
                euclid_err = ((guess_x - _a_x)**2 + (guess_y - _a_y)**2) ** 0.5
                trial_err += euclid_err

            return_dict = {
                "path": best_path,
                "arena": arena,
                "agent": agent_dict,
                "wall_i_s": wall_i_s,
                "agent_offset": (a_x_offset, a_y_offset),
                "obj_1_dict": obj_1_dict,
                "obj_2_dict": obj_2_dict,
                "trial_1_objs_pos": both_o_pos,
                "objs_ch_avgs": objs_ch_avgs,
                "agent_ch_avgs": a_ch_avgs,
                "home": home_dict,
                "erase_wall": erase_wall_mask,
                "trial_err": trial_err,
                "step_num": i+1,
                "trial_num": info["trial_num"]
            }

            return return_dict

        scene_err = scene_steps = scene_path_err = scene_voe = 0
        num_trials = 9
        trial_info = None
        action_i = 0
        self.controller.start_scene(config)

        for idx in range(num_trials):
            path_taken = []

            first_action = config['goal']['action_list'][action_i][0]
            action_i += 1
            print(config['goal']['action_list'])
            step_output = self.controller.step(action=first_action)
            cam_im = step_output.image_list[0]
            cam_im = np.array(cam_im)[:,:,::-1]
            mask_im = step_output.object_mask_list[0]
            mask_im = np.array(mask_im)[:,:,::-1]

            init_info_dict = {
                "trial_num": idx,
                "trial_err": 0,
                "step_num": 0,
                "arena": None,
                "agent_ch_avgs": None,
                "objs_ch_avgs": None,
                "trial_1_objs_pos": None,
                "path": None,
                "wall_i_s": None
            }
            if type(trial_info) != type(None):
                # we want to carry these over between trials
                init_info_dict["objs_ch_avgs"] = trial_info["objs_ch_avgs"]
                init_info_dict["agent_ch_avgs"] = trial_info["agent_ch_avgs"]
                init_info_dict["arena"] = trial_info["arena"]
                init_info_dict["wall_i_s"] = trial_info["wall_i_s"]

            # trial's first step
            trial_info = step(cam_im, mask_im, init_info_dict, True)  
            self.controller.make_step_prediction(
                choice=random_choice(), confidence=random_confidence(), violations_xy_list=random_voe_list(),
                heatmap_img=step_output.image_list[0], internal_state={}
            )

            next_action = config['goal']['action_list'][action_i][0]
            action_i += 1
            step_output = self.controller.step(action=next_action)
            cam_im = step_output.image_list[0]
            cam_im = np.array(cam_im)[:,:,::-1]
            mask_im = step_output.object_mask_list[0]
            mask_im = np.array(mask_im)[:,:,::-1]

            while type(cam_im) != type(None) and next_action == "Pass":
                trial_info["trial_num"] = idx
                trial_info = step(cam_im, mask_im, trial_info, False)
                self.controller.make_step_prediction(
                    choice=random_choice(), confidence=random_confidence(), violations_xy_list=random_voe_list(),
                    heatmap_img=step_output.image_list[0], internal_state={}
                )
                try:
                    next_action = config['goal']['action_list'][action_i][0]
                except:
                    break
                step_output = self.controller.step(action=next_action)
                if type(step_output) != type(None):
                    cam_im = step_output.image_list[0]
                    cam_im = np.array(cam_im)[:,:,::-1]
                    mask_im = step_output.object_mask_list[0]
                    mask_im = np.array(mask_im)[:,:,::-1]
                else:
                    cam_im = None
                    mask_im = None
                action_i += 1

            # end of multi-object trial #1
            if trial_info["trial_num"] == 0 and type(trial_info["obj_2_dict"]["c"]) != type(None):
                a_x, a_y = trial_info["agent"]["x"], trial_info["agent"]["y"]
                o_1_x, o_1_y = trial_info["trial_1_objs_pos"][0]
                o_2_x, o_2_y = trial_info["trial_1_objs_pos"][1]
                dist_1 = ((o_1_x - a_x)**2 + (o_1_y - a_y)**2) ** 0.5
                dist_2 = ((o_2_x - a_x)**2 + (o_2_y - a_y)**2) ** 0.5

                if dist_1 > dist_2:
                    # reverse so the preferred is the tuple's 1st array
                    trial_info["objs_ch_avgs"] = (trial_info["objs_ch_avgs"][1], trial_info["objs_ch_avgs"][0])
                    trial_info["path"] = calc_path(trial_info["arena"], tuple(path_taken[0]), px_to_arena((o_2_x, o_2_y), trial_info["home"]["rgb"]))
                else:
                    trial_info["path"] = calc_path(trial_info["arena"], tuple(path_taken[0]), px_to_arena((o_1_x, o_1_y), trial_info["home"]["rgb"]))

                end_offset = 4
                trial_err = 0
                for i in range(trial_info["step_num"]):
                    if i >= len(trial_info["path"]) - end_offset:
                        if (len(trial_info["path"])) == 0:
                            guess_x, guess_y = (0,0)
                            print("PATHING ISSUE :(")
                        else: 
                            tot_off = -1 - end_offset
                            if abs(tot_off) >= len(trial_info["path"]):
                                tot_off = -1
                            guess_x, guess_y = trial_info["path"][tot_off]
                    else:
                        guess_x, guess_y = trial_info["path"][i]
                    a_x, a_y = path_taken[i]
                    euclid_err = ((guess_x - a_x)**2 + (guess_y - a_y)**2) ** 0.5
                    trial_err += euclid_err
                trial_info["trial_err"] = trial_err

            #print("path:", trial_info["path"])
            path_len = len(trial_info["path"])
            # @TODO remove hack for a common problem where the path isn't possible.
            # e.g. sometimes we detect one of the objects inside a wall and no path is possible.
            if path_len == 0:
                print("PATHING ISSUE :(")
                path_len = 1
            actual_len = len(path_taken)
            #print("actual path lenght:", actual_len)
            grid = Grid(matrix=trial_info["arena"])
            start = grid.node(path_taken[0][0], path_taken[0][1])
            #print("end:", path_taken[actual_len-1])
            #print(path_taken)
            end = grid.node(path_taken[actual_len-1][0], path_taken[actual_len-1][1])
            print(grid.grid_str(path=path_taken, start=start, end=end))
            #print("total error:", trial_info["trial_err"])
            scene_err += trial_info["trial_err"]
            scene_steps += actual_len
            avg_trial_a_pos_err = trial_info["trial_err"]/actual_len
            print("trial #" + str(idx+1) + " avg error:", avg_trial_a_pos_err)
            empirical_offset = 2
            print("gt path len:", actual_len, "inferred path len:", path_len - empirical_offset)
            path_err = abs(actual_len - empirical_offset - path_len)
            scene_path_err += path_err
            print("trial #" + str(idx+1) + "'s path length difference:", path_err)
            trial_voe = 0
            wiggle = 3
            path_err_plus_wiggle = (actual_len - empirical_offset - path_len - wiggle)
            if path_err_plus_wiggle > 0:
                trial_voe = (actual_len - empirical_offset - path_len) / (2 * path_len)
                if trial_voe > 1:
                    trial_voe = 1
            print("trial #" + str(idx+1) + "'s VOE:", trial_voe)
            scene_voe += trial_voe
        print("avg trial VOE:", scene_voe/num_trials)
        print("avg step error over scene:", scene_err/scene_steps)
        print("avg trial path difference:", scene_path_err/num_trials)

        self.controller.end_scene(choice=random_choice(), confidence=random_confidence())