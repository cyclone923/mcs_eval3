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

    def __init__(self, controller, level, debug=True):
        self.controller = controller
        self.level = level
        self.debug = debug
        self.trial_num = 0
        self.trial_err = 0
        self.step_num = 0
        self.agent_offset = [0,0]
        self.arena = None
        self.path = None # our prediction of where the agent will go
        self.home_side_len = 29 # home side length in px after being cropped and homographied
        self.home_side_wiggle = 10
        self.home_colors = [252, 49, 254]
        self.home_color_wiggle = 0.33
        self.agent_wiggle = 30
        self.path_wiggle = 1 # used to calculate VOE based on how accurately we infer the agent's path
        self.wall_i_s = None
        self.gnd_mask = None
        self.gnd_rgb = None
        self.M_wall, self.M_gnd = get_homographies()
        self.choice = None
        self.confidence = 1
        self.pref_confidence = 1
        self.pref_dict = {
            "obj_1_color": [],
            "obj_2_color": [],
            "obj_1_hog": [],
            "obj_2_hog": [],
            "obj_1_dist": [],
            "obj_2_dist": [],
            "chosen": []
        }
        self.agent_dict = {
            "ch_avgs": None,
            "mask": None,
        }
        self.home_dict = {
            "rgb": None,
            "mask": None,
            "c": None,
            "center": None,
        }
        self.obj_1_dict = {
            "rgb": None,
            "mask": None,
            "c": None,
        }
        self.obj_2_dict = {
            "rgb": None,
            "mask": None,
            "c": None,
        }

        # quick and dirty way to toggle debug printing to screen
        if not self.debug:
            def silence_print(*args,**kwargs):
                return True

            #print = silence_print
        
    def run_scene(self, config):

        self.arena = create_arena()
        self.structural_mask_colors = []

        try:
            scene_err = scene_steps = scene_path_err = scene_voe = 0
            num_trials = 9
            action_i = 0
            self.controller.start_scene(config)

            for idx in range(num_trials):
                self.trial_err = 0
                self.path_taken = []
                self.trial_num = idx
                self.first_trial_frame = True

                while True:
                    try:
                        next_action = config['goal']['action_list'][action_i][0]
                    except:
                        next_action = "I like toytles"
                    action_i += 1 
                    step_output = self.controller.step(action=next_action)
                    if next_action != "Pass":
                        break
                    self.cam_im = step_output.image_list[0]
                    self.cam_im = np.array(self.cam_im)[:,:,::-1]
                    self.mask_im = step_output.object_mask_list[0]
                    self.mask_im = np.array(self.mask_im)[:,:,::-1]

                    self.step()
                    self.first_trial_frame = False

                # if multiple object trial
                if type(self.obj_2_dict["c"]) != type(None):
                    # set nearest object to agent as the chosen one
                    a_x, a_y = self.agent_dict["x"], self.agent_dict["y"]
                    a_pos = [a_x, a_y]
                    obj_1 = self.obj_1_dict["rgb"]
                    obj_2 = self.obj_2_dict["rgb"]
                    dist1 = dist_agent_obj(a_pos, obj_1)
                    dist2 = dist_agent_obj(a_pos, obj_2)
                    if dist1 < dist2:
                        #cv2_show_im(obj_1)
                        self.pref_dict["chosen"].append(1)
                    else:
                        #cv2_show_im(obj_2)
                        self.pref_dict["chosen"].append(2)

                if self.trial_num == 8:
                    path_len = len(self.path)
                    # @TODO remove hack for a common problem where the path isn't possible.
                    # e.g. sometimes we detect one of the objects inside a wall and no path is possible.
                    if path_len == 0:
                        print("PATHING ISSUE :(")
                        path_len = 1
                    actual_len = len(self.path_taken)
                    #print("actual path lenght:", actual_len)
                    grid = Grid(matrix=self.arena)
                    start = grid.node(self.path_taken[0][0], self.path_taken[0][1])
                    end = grid.node(self.path_taken[actual_len-1][0], self.path_taken[actual_len-1][1])
                    print(grid.grid_str(path=self.path_taken, start=start, end=end))
                    scene_err += self.trial_err
                    scene_steps += actual_len
                    avg_trial_a_pos_err = self.trial_err/actual_len
                    print("trial #" + str(idx+1) + " avg error:", avg_trial_a_pos_err)
                    empirical_offset = 2
                    print("gt path len:", actual_len, "inferred path len:", path_len - empirical_offset)
                    path_err = abs(actual_len - empirical_offset - path_len)
                    scene_path_err += path_err
                    print("trial #" + str(idx+1) + "'s path length difference:", path_err)
            print("avg step error over scene:", scene_err/scene_steps)
            print("avg trial path difference:", scene_path_err/num_trials)

            voe_threshold = 6.5
            # confidence is proportional to how well the agent followed our inferred path
            # and weighted by our confidence in our agent's object preference hypothesis.
            confidence = ((avg_trial_a_pos_err - self.path_wiggle) * 2 - voe_threshold) / voe_threshold
            confidence *= self.pref_confidence
            choice = "expected" if confidence < 0 else "unexpected"
            confidence = abs(confidence)
            if confidence > 1.0:
                confidence = 1.0

            print("Final choice:", choice)
            print("Final confidence:", confidence)

            self.controller.end_scene(choice=choice, confidence=confidence)

        # end try from the beginning of run_scene(). This is better than throwing an error all the way up and breaking the agent
        except Exception as e:
            if self.debug:
                import traceback 
                traceback.print_exc()
                print(e)
                exit()
            i = 0
            while i < 400:
                try:
                    i += 1
                    self.controller.step(action="Pass")
                    self.controller.step(action="EndHabituation")
                    self.controller.make_step_prediction(
                        choice=random_choice(), confidence=random_confidence(), violations_xy_list=random_voe_list(),
                        heatmap_img=step_output.image_list[0], internal_state={"info": "runtime error"}
                    )
                except Exception as e:
                    break
            try:
                self.controller.end_scene(choice="unexpected", confidence=1.0)
            except Exception as e:
                pass

    def step(self):
        self.gnd_mask = gnd_trans(self.mask_im, self.M_gnd)
        self.gnd_rgb = gnd_trans(self.cam_im, self.M_gnd)
        
        trans_im = wall_trans(self.cam_im, self.M_wall)
        
        if self.trial_num + self.step_num == 0:
            _walls_im, self.arena, self.wall_i_s = find_walls(trans_im, self.arena)

        self.structural_mask_colors.append(get_gnd_mask_color(self.gnd_mask))
        wall_filtered = []
        for x, y in self.wall_i_s:
            c = get_mask_color(self.gnd_mask, x, y)
            if is_wall(self.gnd_rgb, self.gnd_mask, c):
                self.structural_mask_colors.append(c)
                wall_filtered.append((x, y))
        self.wall_i_s = wall_filtered

        # hard coded clockwise border mask positions
        borders_pos = [(535, 114), (456, 212), (141, 210), (64, 112)]
        for x,y in borders_pos:
            self.structural_mask_colors.append(tuple(self.mask_im[y, x, ::-1]))

        self.id_unknowns()

        self.pref_confidence = 1
        a_pos = (self.agent_dict["x"], self.agent_dict["y"])

        # the default for multi-object case during trial #1 steps, but shouldn't even be used
        o_x, o_y = 0, 0
        if type(self.obj_2_dict["c"]) == type(None):
            o_x, o_y = get_obj_pos(self.obj_1_dict["rgb"])

        a_x, a_y, = px_to_arena(a_pos, self.home_dict["rgb"])
        self.path_taken.append((a_x, a_y))
        # if self.trial_num == 7:
        #     cv2_show_im(self.agent_dict["rgb"], self.home_dict["rgb"])
        #     cv2_show_im(self.obj_1_dict["rgb"], self.obj_2_dict["rgb"])

        # if the first frame of the multi-object case
        if self.first_trial_frame and type(self.obj_2_dict["c"]) != type(None):
            if self.trial_num == 8:
                cv2_show_im(self.agent_dict["rgb"])
                o_x, o_y = self.get_pref_obj_pos()
                self.path = calc_path(self.arena, (a_x, a_y), px_to_arena((o_x, o_y), self.home_dict["rgb"]))

                choice = "expected"
                confidence = 0.0

                end_offset = 4
                # if type(self.path) == type(None):
                #     self.path = calc_path(self.arena, (a_x, a_y), px_to_arena((o_x, o_y), self.home_dict["rgb"]))
                if self.step_num >= len(self.path) - end_offset:
                    if (len(self.path)) == 0:
                        guess_x, guess_y = (0,0)
                        print("PATHING ISSUE :(")
                    else:    
                        tot_off = -1 - end_offset
                        if abs(tot_off) >= len(self.path):
                            tot_off = -1
                        guess_x, guess_y = self.path[tot_off]
                else:
                    guess_x, guess_y = self.path[self.step_num]
                euclid_err = ((guess_x - a_x)**2 + (guess_y - a_y)**2) ** 0.5
                self.trial_err += euclid_err

                # A little more than a wall piece distance between calculated and actual agent pos is voe_threshold.
                voe_threshold = 12
                self.confidence = (self.trial_err * 2 - voe_threshold) / voe_threshold
                choice = "expected" if self.confidence < 0 else "unexpected"
                self.confidence = abs(self.confidence)
                self.confidence *= self.pref_confidence
                if self.confidence > 1.0:
                    self.confidence = 1.0

                self.controller.make_step_prediction(
                    choice=self.choice, confidence=self.confidence, violations_xy_list=[],
                    heatmap_img=None, internal_state={}
                )
            else:
                self.pref_dict["obj_1_color"].append(list(get_ch_avgs(self.obj_1_dict["rgb"])))
                self.pref_dict["obj_2_color"].append(list(get_ch_avgs(self.obj_2_dict["rgb"])))
                self.pref_dict["obj_1_hog"].append(list(get_obj_hog(self.obj_1_dict["rgb"])))
                self.pref_dict["obj_2_hog"].append(list(get_obj_hog(self.obj_2_dict["rgb"])))
                self.pref_dict["obj_1_dist"].append(dist_agent_obj(a_pos, self.obj_1_dict["rgb"]))
                self.pref_dict["obj_2_dist"].append(dist_agent_obj(a_pos, self.obj_2_dict["rgb"]))
                
                self.controller.make_step_prediction(
                    choice="expected", confidence=1.0, violations_xy_list=[],
                    heatmap_img=None, internal_state={}
                )

        self.step_num += 1

        return True

    def id_unknowns(self):
        '''
        Identifies what the remaining masks are. Should be agent(s), object(s), home square(s), 
        and maybe a wall occluded by an agent.
        '''

        unknown_colors = get_unknowns(self.gnd_mask, self.structural_mask_colors)
        #print("Number of yet unidentified objects:", len(unknown_colors))

        # finding home should be easiest. Except for edge cases the agent is the largest unknown mask
        # the object(s) of interest never change location unless occulded by the agent.
        # can loop thru these multiple times if necessary, doesn't have to be done in a single pass.
        # Hog might make tracking the agent easier, current method suffers from shadows, lighting changes
        # while moving across the map, and when the agent is occulded by walls.
        for c in unknown_colors:
            if is_wall(self.gnd_rgb, self.gnd_mask, c):
                self.structural_mask_colors.append(c)
                print("removing possible wall piece from unknowns")
                continue

            obj_im, mask = apply_color_mask(np.copy(self.gnd_rgb), np.copy(self.gnd_mask), c, show=False)

            if type(self.agent_dict["ch_avgs"]) == type(None):
                a_color_dist = 0
            else:
                # color histogram different between the agent and the current unknown object
                a_color_dist = np.absolute(self.agent_dict["ch_avgs"] - get_ch_avgs(obj_im)).sum()

            color_diff_to_home = np.absolute(self.home_colors - get_ch_avgs(obj_im)).sum()
            non_zero_c = cv2.countNonZero(obj_im.sum(axis=2))
            color_diff_to_home = color_diff_to_home / non_zero_c

            # home square is a certain color, height and width after homography
            l,r,t,b = get_mask_cardinals(obj_im)
            width = r-l
            height = b-t
            # can change this to per channel avg if this becomes too broad a range
            is_home = color_diff_to_home < self.home_color_wiggle
            is_home = is_home and abs(width-self.home_side_len) < self.home_side_wiggle and abs(height-self.home_side_len) < self.home_side_wiggle
            if is_home:
                self.home_dict["rgb"] = obj_im
                self.home_dict["mask"] = mask
                self.home_dict["c"] = c
                self.home_dict["center"] = (l + width//2, t + height//2)
            elif a_color_dist < self.agent_wiggle:
                if type(self.agent_dict["mask"]) != type(None) and self.first_trial_frame:
                    if type(self.obj_1_dict["mask"]) != type(None):
                        #print("Set object 2!!")
                        self.obj_2_dict["rgb"] = self.agent_dict["rgb"]
                        self.obj_2_dict["mask"] = self.agent_dict["mask"]
                        self.obj_2_dict["c"] = self.agent_dict["c"]
                    else:
                        #print("Set object 1!!")
                        self.obj_1_dict["rgb"] = self.agent_dict["rgb"]
                        self.obj_1_dict["mask"] = self.agent_dict["mask"]
                        self.obj_1_dict["c"] = self.agent_dict["c"]
                #print("Found agent!")
                self.agent_dict["rgb"] = obj_im
                self.agent_dict["mask"] = mask
                self.agent_dict["c"] = c
            elif type(self.obj_1_dict["mask"]) == type(None):
                #print("Set object 1!")
                self.obj_1_dict["rgb"] = obj_im
                self.obj_1_dict["mask"] = mask
                self.obj_1_dict["c"] = c
            else:
                #print("Set object 2!")
                self.obj_2_dict["rgb"] = obj_im
                self.obj_2_dict["mask"] = mask
                self.obj_2_dict["c"] = c

        # update the agent's color histogram since moving around occludes bits of it and the lighting changes
        self.agent_dict["ch_avgs"] = get_ch_avgs(self.agent_dict["rgb"])

        h_x, h_y = get_home_pos(self.home_dict["rgb"])
        a_x, a_y = get_agent_pos(self.agent_dict["rgb"])
        
        if self.first_trial_frame:
            # we can calculate how wrong our agent_pos fx is on the first frame using home's center
            self.agent_offset[0] = h_x - a_x
            self.agent_offset[1] = h_y - a_y

        a_x, a_y = a_x + self.agent_offset[0], a_y + self.agent_offset[1]
        self.agent_dict["x"] = a_x
        self.agent_dict["y"] = a_y

    def get_pref_obj_pos(self):
        history = self.pref_dict
        o_1, o_2 = self.obj_1_dict, self.obj_2_dict
        a_pos = [self.agent_dict["x"], self.agent_dict["y"]]

        chosen = history["chosen"]
        o_1_d = history["obj_1_dist"]
        o_2_d = history["obj_2_dist"]
        o_1_c = history["obj_1_color"]
        o_2_c = history["obj_2_color"]
        o_1_h = history["obj_1_hog"]
        o_2_h = history["obj_2_hog"]
        num_trials = len(o_1_d)

        # calculate short_dist classifier accuracy
        dist_acc = 0
        for i in range(num_trials):
            if o_1_d[i] < o_2_d[i]:
                closest = 1 
            else:
                closest = 2

            if closest == chosen[i]:
                dist_acc += 1
        dist_acc /= num_trials
        short_acc = dist_acc
        far_acc = 1 - dist_acc

        # calculate color classifier accuracy with k-fold x-validation 
        color_acc = 0
        for i in range(num_trials):
            # needs to be an average of the chosen color!
            avg_chosen_c = []
            for j in range(num_trials):
                # leave one out for the gt label
                if j == i: continue
                if chosen[j] == 1:
                    avg_chosen_c.append(o_1_c[j])
                else:
                    avg_chosen_c.append(o_2_c[j])
            
            # need to do this per channel :-/ ...
            avg_chosen_c = np.array(avg_chosen_c)
            #print("avg_chosen_c.shape:", avg_chosen_c.shape)
            avg_chosen_c = avg_chosen_c.sum(axis=0)
            avg_chosen_c /= num_trials
            #print("avg_chosen_c.shape:", avg_chosen_c.shape)
            assert avg_chosen_c.shape[0] == 3
            
            dist_1 = np.absolute(o_1_c[i] - avg_chosen_c).sum()
            dist_2 = np.absolute(o_2_c[i] - avg_chosen_c).sum()
            if dist_1 < dist_2:
                closest = 1
            else:
                closest = 2

            if closest == chosen[i]:
                color_acc += 1
        color_acc /= num_trials

        # calculate shape/HOG classifier accuracy with k-fold x-validation 
        hog_acc = 0
        for i in range(num_trials):
            avg_chosen_h = []
            for j in range(num_trials):
                # leave one out for the gt label
                if j == i: continue
                if chosen[j] == 1:
                    avg_chosen_h.append(o_1_h[j])
                else:
                    avg_chosen_h.append(o_2_h[j])
            
            avg_chosen_h = np.array(avg_chosen_h)
            #print("avg_chosen_h.shape:", avg_chosen_h.shape)
            avg_chosen_h = avg_chosen_h.sum(axis=0)
            avg_chosen_h /= num_trials
            #print("avg_chosen_h.shape:", avg_chosen_h.shape) # should be 1d vector
            
            dist_1 = np.absolute(o_1_h[i] - avg_chosen_h).sum()
            dist_2 = np.absolute(o_2_h[i] - avg_chosen_h).sum()
            if dist_1 < dist_2:
                closest = 1
            else:
                closest = 2

            if closest == chosen[i]:
                hog_acc += 1
        hog_acc /= num_trials

        # use best classifier to determine the currently preferred object
        pref_confidence = 1
        # used in the case of tied hypothesis classifier accuracy
        h_1_chosen = None
        h_2_chosen = None
        h_3_chosen = None
        hypotheses = [short_acc, far_acc, color_acc, hog_acc]
        print("short_acc:", short_acc)
        print("far_acc:", far_acc)
        print("color_acc:", color_acc)
        print("hog_acc:", hog_acc)
        best_acc = max(hypotheses)
        # tied preference hypotheses affects our confidence
        tied_count = hypotheses.count(best_acc)
        # will be altered later if theirs a classifier tie
        self.pref_confidence = best_acc
        best_classifier = hypotheses.index(best_acc)
        
        if tied_count == 1:
            if best_classifier == 0 or best_classifier == 1:
                print("preferred object is based on distance")
                dist_1 = dist_agent_obj(a_pos, o_1["rgb"])
                dist_2 = dist_agent_obj(a_pos, o_2["rgb"])
                if best_classifier == 0:
                    if dist_1 < dist_2:
                        o_x, o_y = get_obj_pos(o_1["rgb"])
                    else:
                        o_x, o_y = get_obj_pos(o_2["rgb"])
                if best_classifier == 1:
                    if dist_1 > dist_2:
                        o_x, o_y = get_obj_pos(o_1["rgb"])
                    else:
                        o_x, o_y = get_obj_pos(o_2["rgb"])
            elif best_classifier == 2:
                print("preferred object is a certain color")
                dist_1 = np.absolute(get_ch_avgs(o_1["rgb"]) - avg_chosen_c).sum()
                dist_2 = np.absolute(get_ch_avgs(o_2["rgb"]) - avg_chosen_c).sum()
                if dist_1 < dist_2:
                    o_x, o_y = get_obj_pos(o_1["rgb"])
                else:
                    o_x, o_y = get_obj_pos(o_2["rgb"])
            else:
                print("preferred object is a certain shape")
                dist_1 = np.absolute(get_obj_hog(o_1["rgb"]) - avg_chosen_h).sum()
                dist_2 = np.absolute(get_obj_hog(o_2["rgb"]) - avg_chosen_h).sum()
                if dist_1 < dist_2:
                    o_x, o_y = get_obj_pos(o_1["rgb"])
                else:
                    o_x, o_y = get_obj_pos(o_2["rgb"])
        elif tied_count == 4:
            # All 4 would be 0.5 accurate and we divide that by 4
            pref_confidence = 0.125
            o_x, o_y = get_obj_pos(o_2["rgb"])
        else:
            # need to calculate the best 2 and then compare predictions
            # same accuracy as tied_count == 1 if they predict the same object
            if hypotheses[0] == best_acc:
                dist_1 = dist_agent_obj(a_pos, o_1["rgb"])
                dist_2 = dist_agent_obj(a_pos, o_2["rgb"])
                if dist_1 < dist_2:
                    h_1_chosen = 1
                else:
                    h_1_chosen = 2
            if hypotheses[1] == best_acc:
                dist_1 = dist_agent_obj(a_pos, o_1["rgb"])
                dist_2 = dist_agent_obj(a_pos, o_2["rgb"])
                if dist_1 > dist_2:
                    if type(h_1_chosen) == type(None):
                        h_1_chosen = 1  
                    else:
                        h_2_chosen = 1
                else:
                    if type(h_1_chosen) == type(None):
                        h_1_chosen = 2  
                    else:
                        h_2_chosen = 2
            if hypotheses[2] == best_acc:
                # color classifier
                dist_1 = np.absolute(get_ch_avgs(o_1["rgb"]) - avg_chosen_c).sum()
                dist_2 = np.absolute(get_ch_avgs(o_2["rgb"]) - avg_chosen_c).sum()
                # print("Here's what the color classifier chose:")
                if dist_1 < dist_2:
                    if type(h_1_chosen) == type(None):
                        h_1_chosen = 1
                    elif type(h_2_chosen) == type(None):
                        h_2_chosen = 1
                    else:
                        h_3_chosen = 1
                    # cv2_show_im(o_1["rgb"])
                else:
                    if type(h_1_chosen) == type(None):
                        h_1_chosen = 2  
                    elif type(h_2_chosen) == type(None):
                        h_2_chosen = 2
                    else:
                        h_3_chosen = 2
                    # cv2_show_im(o_2["rgb"])
            if hypotheses[3] == best_acc:
                # shape/HOG classifier
                dist_1 = np.absolute(get_obj_hog(o_1["rgb"]) - avg_chosen_h).sum()
                dist_2 = np.absolute(get_obj_hog(o_2["rgb"]) - avg_chosen_h).sum()
                # print("Here's what the shape/HOG classifier chose:")
                if dist_1 < dist_2:
                    if type(h_2_chosen) == type(None):
                        h_2_chosen = 1  
                    else:
                        h_3_chosen = 1
                    # cv2_show_im(o_1["rgb"])
                else:
                    if type(h_2_chosen) == type(None):
                        h_2_chosen = 2
                    else:
                        h_3_chosen = 2
                    # cv2_show_im(o_2["rgb"])

            # now modulate confidence based on # of tied acc classifiers
            # and whether or not those tied ones agree or not...
            if tied_count == 2:
                if h_1_chosen == h_2_chosen:
                    if h_1_chosen == 1:
                        o_x, o_y = get_obj_pos(o_1["rgb"])
                    else:
                        o_x, o_y = get_obj_pos(o_2["rgb"])
                else:
                    pref_confidence *= 0.5
                    print("[get_pref_obj_pos()] just pick one i.e. guessing at this point")
                    # can use cv2_show() to show which 2 were picked
                    o_x, o_y = get_obj_pos(o_1["rgb"])
            elif tied_count == 3:
                if h_1_chosen == h_2_chosen == h_3_chosen:
                    if h_1_chosen == 1:
                        o_x, o_y = get_obj_pos(o_1["rgb"])
                    else:
                        o_x, o_y = get_obj_pos(o_2["rgb"])
                elif h_1_chosen == h_2_chosen:
                    pref_confidence *= 0.66
                    if h_1_chosen == 1:
                        o_x, o_y = get_obj_pos(o_1["rgb"])
                    else:
                        o_x, o_y = get_obj_pos(o_2["rgb"])
                elif h_1_chosen == h_3_chosen:
                    pref_confidence *= 0.66
                    if h_1_chosen == 1:
                        o_x, o_y = get_obj_pos(o_1["rgb"])
                    else:
                        o_x, o_y = get_obj_pos(o_2["rgb"])
                else:
                    # must be h_2 and h_3 that agreed
                    pref_confidence *= 0.66
                    if h_2_chosen == 1:
                        o_x, o_y = get_obj_pos(o_1["rgb"])
                    else:
                        o_x, o_y = get_obj_pos(o_2["rgb"])

        return o_x, o_y
