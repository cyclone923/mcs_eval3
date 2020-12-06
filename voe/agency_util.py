import os
import cv2
import glob
import math
import random
import numpy as np
from copy import deepcopy
from skimage.feature import hog
import matplotlib.pyplot as plt
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement

def random_float_with_range(low, high):
    assert high > low
    return low + random.random() * (high - low)

def random_choice():
    choice = ["expected", "unexpected"]
    return random.choice(choice)

def random_confidence():
    return random.random()

def random_voe_list():
    n_point = random.randint(0, 9)
    return [{'x': random_float_with_range(0, 600), 'y': random_float_with_range(0, 400)} for _ in range(n_point)]

def find_walls(im, arena_mat, debug=False):
    wall_thresh = 100
    grid_len = 8
    # ratio of sampling and arena grids
    res_ratio = len(arena_mat[0]) / grid_len
    # assumes the image is square
    grid_inc = im.shape[0] / grid_len
    wall_indices = []    

    for i in range(grid_len**2):
        x = i // grid_len
        y = i % grid_len        
        top = int(y * grid_inc)
        bot = int(top + grid_inc)
        left = int(x * grid_inc)   
        right = int(left + grid_inc)
        # looking at the bottom-left two-thirds of the grid element
        _top = int(top + grid_inc//3)
        _right = int(right - grid_inc//3)

        # assumes square crop
        area = (_top-bot)**2
        num_poss_black_chs = 0 
        for i in range(3):
            avg_ch_color_value = np.sum(im[_top:bot, left:_right, i]) / area
            if avg_ch_color_value < wall_thresh:
                num_poss_black_chs += 1
        if num_poss_black_chs == 3:
            if debug:
                # draw in the walls
                im[top:bot, left:right, :] = 0
            if type(arena_mat) != type(None):
                # mark wall's position as occupied in arena grid
                for _i in range(5):
                    for _j in range(5):
                        _x = int(x * res_ratio + _i)
                        _y = int(y * res_ratio + _j)
                        arena_mat[_y][_x] = 0
            wall_indices.append((x,y))
            # extra corner bits to mimic agent pathing (e.g. wall/corner avoidance)
            extra_corners = [[0,-1],[-1,0],[4,-1],[5,0],[5,4],[4,5],[0,5],[-1,4],[-1,-1],[5,5],[-1,5],[5,-1]]
            
            if type(arena_mat) != type(None):
                for p_x, p_y in extra_corners:
                    p_x += int(x * res_ratio)
                    p_y += int(y * res_ratio)
                    valid_coors = p_x > 0 and p_x < len(arena_mat[0])-1
                    valid_coors = valid_coors and p_y > 0 and p_y < len(arena_mat[0])-1
                    if valid_coors:
                        arena_mat[p_y][p_x] = 0
    if debug:
        cv2.imwrite("walls_test.png", im)
    return im, arena_mat, wall_indices

# perspective transform to the top of the walls
def wall_trans(img, M):
    l_offset = 67
    r_offset = -63
    t_offset = 43
    b_offset = -107
    img = img[t_offset:b_offset, l_offset:r_offset, :]
    h, w = img.shape[:2]
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    warped = warped[:-25, :-245, :]
    # assumes the image is square
    #grid_inc = warped.shape[0] / 8
    # draws the grid
    # for j in range(8):
    #     cv2.line(warped, (int(j*grid_inc), 0), (int(j*grid_inc), h), (255, 0, 0), thickness=1)
    #     cv2.line(warped, (0, int(j*grid_inc)), (w, int(j*grid_inc)), (255, 0, 0), thickness=1)
    warped = warped[:,::-1,:]
    #cv2.imwrite("wall_trans.png", warped)

    return warped

# perspective transform to the ground plane
def gnd_trans(img, M):
    l_offset = 27
    r_offset = -63
    t_offset = 43
    b_offset = -57

    img = img[t_offset:b_offset, l_offset:r_offset, :]
    h, w = img.shape[:2]
    
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_NEAREST)
    # used opencv gui to see that each grid element had a width of ~28 pixels
    warped = warped[22:254, 22:252, :]
    # assumes the image is square
    # grid_inc = warped.shape[0] / 8
    # # draws the grid
    # for j in range(8):
    #     cv2.line(warped, (int(j*grid_inc), 0), (int(j*grid_inc), h), (255, 0, 0), thickness=1)
    #     cv2.line(warped, (0, int(j*grid_inc)), (w, int(j*grid_inc)), (255, 0, 0), thickness=1)
    warped = warped[:,::-1,:]
    #cv2.imwrite("gnd_trans.png", warped)

    return warped

def get_gnd_mask_color(im):
    '''
    Assumes the most common color of grid centers is the ground plane mask
    im: mask image pre-processed into 8x8 grid
    '''
    grid_len = 8
    # assumes the image is square
    grid_inc = im.shape[0] / grid_len
    colors = []

    for i in range(grid_len**2):
        x = i % grid_len        
        y = i // grid_len
        center_x = int(x * grid_inc + grid_inc//2)
        center_y = int(y * grid_inc + grid_inc//2)
        colors.append(tuple(im[center_y, center_x, ::-1]))
    most_common = max(set(colors), key=colors.count)

    return most_common

def get_mask_color(im, x, y):
    '''
    im: mask image pre-processed into 8x8 grid
    x: 0-7
    y: 0-7
    '''
    grid_len = 8
    # assumes the image is square
    grid_inc = im.shape[0] / grid_len

    center_x = int(x * grid_inc + grid_inc//2)
    center_y = int(y * grid_inc + grid_inc//2)
    return tuple(im[center_y, center_x, ::-1])

def add_border_colors(im, l):
    '''
    im: mask image 600x400 px
    l: list to add color tuples to
    '''
    # hard coded border mask positions
    borders_pos = [(535, 114), (456, 212), (141, 210), (64, 112)] # clockwise
    for x,y in borders_pos:
        l.append(tuple(im[y, x, ::-1]))
    return l

def get_unknowns(im, knowns, debug=False):
    '''
    Returns a list of color tuples for objects, agents, and home given a list of know mask colors
    im: mask opencv format image pre-processed into 8x8 grid
    '''
    new = []
    # search in a 25x25 grid for unknown mask colors
    search_len = 25
    # assumes image is square
    grid_len = im.shape[0] / search_len
    for i in range(search_len):
        for j in range(search_len):
            x = int((i+0.5) * grid_len)
            y = int((j+0.5) * grid_len)
            p_color = tuple(im[y, x, ::-1])
            if debug:
                im[y, x, :] = [255, 255, 255]
            if p_color not in knowns and p_color not in new:
                new.append(p_color)
    if debug:
        cv2_show_im(im)
    return new

def cv2_show_im(im, im2=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.subplots_adjust(hspace=.2, wspace=.1)
    ax1.imshow(im)
    if type(im2) == type(None):
        ax2.imshow(im)
    else:
        ax2.imshow(im2)
    plt.show()

def apply_color_mask(rgb, mask, color, show=False):
    if show:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        f.subplots_adjust(hspace=.2, wspace=.1)
        ax1.imshow(rgb)
    m = cv2.inRange(mask, np.array(list(color))[::-1], np.array(list(color))[::-1])
    rgb[m == 0] = 0
    if show:
        ax2.imshow(rgb)
        plt.show()
    return rgb, m

def get_mask_cardinals(im):
    # returns the extreme L,R,T,B indices of the mask
    positions = cv2.findNonZero(im.sum(axis=2))
    positions = np.squeeze(positions)
    num_px = positions.shape[0]
    left_most_px = np.argmin(positions[:,0])
    right_most_px = np.argmax(positions[:,0])
    top_most_px = np.argmin(positions[:,1])
    bot_most_px = np.argmax(positions[:,1])
    left = positions[left_most_px][0]
    right = positions[right_most_px][0]
    top = positions[top_most_px][1]
    bot = positions[bot_most_px][1]
    return left, right, top, bot

def get_home_pos(im):
    left, right, top, bot = get_mask_cardinals(im)
    avg_x = round((left+right) / 2)
    avg_y = round((top+bot) / 2)
    return avg_x, avg_y

def get_agent_pos(im, x_off=0, y_off=0):
    positions = cv2.findNonZero(im.sum(axis=2))
    positions = np.squeeze(positions)
    left_most_px = np.argmin(positions[:,0])
    bot_most_px = np.argmax(positions[:,1])
    p1 = positions[left_most_px]
    p2 = positions[bot_most_px]
    avg_x = math.ceil((p1[0] + p2[0]) / 2) + x_off
    avg_y = math.floor((p1[1] + p2[1]) / 2) + y_off
    return avg_x, avg_y

def get_obj_pos(im):
    try:
        positions = cv2.findNonZero(im.sum(axis=2))
        positions = np.squeeze(positions)
        num_px = positions.shape[0]
        avg_x = int(np.sum(positions[:,0]) / num_px)
        avg_y = int(np.sum(positions[:,1]) / num_px)
        # @TODO fix this hack, could replace with halfing towards the bottom left
        avg_x -= 8
        avg_y += 7
    except:
        avg_x = 0
        avg_y = 0
    return avg_x, avg_y

def get_ch_avgs(im):
    non_zero_c = cv2.countNonZero(im.sum(axis=2))
    ch_avgs = np.array([im[:,:,0].sum(), im[:,:,1].sum(), im[:,:,2].sum()])
    ch_avgs = ch_avgs / non_zero_c
    #print(ch_avgs, ch_avgs.shape)
    return ch_avgs

def calc_path(m, start, goal):
    m[start[1]][start[0]] = 1
    m[goal[1]][goal[0]] = 1
    grid = Grid(matrix=m)
    #print(start,goal)
    start = grid.node(start[0], start[1])
    end = grid.node(goal[0], goal[1])
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path, runs = finder.find_path(start, end, grid)
    #print('operations:', runs, 'path length:', len(path))
    #print(grid.grid_str(path=path, start=start, end=end))
    return path

def px_to_arena(p, im):
    # assumes 40x40 grid
    grid_len = 40
    # assumes the image is square
    grid_inc = grid_len / im.shape[0]
    x = int(p[0] * grid_inc)
    y = int(p[1] * grid_inc)
    return [x, y]

def is_wall(rgb, mask, c):
    obj_im, mask = apply_color_mask(np.copy(rgb), np.copy(mask), c, show=False)
    non_zero_c = cv2.countNonZero(obj_im.sum(axis=2))
    num_poss_black_chs = np.array([0, 0, 0]) 
    for i in range(3):
        avg_ch_color_value = np.sum(obj_im[:,:,i]) / non_zero_c
        #print(avg_ch_color_value)
        if avg_ch_color_value < 100:
            num_poss_black_chs[i] = avg_ch_color_value
        else: 
            return False
    if np.std(num_poss_black_chs) < 8:
        return True

    return False

def get_homographies():
    src_gnd = np.float32([(100, 90), # left
                        (272, 26),   # top
                        (272, 201),  # bottom
                        (444, 90)])  # right

    src_wall = np.float32([(0, 81),
                        (233, 4),
                        (233, 249),
                        (464, 82)])

    dst_gnd = np.float32([(225, 50),
                        (50, 50),
                        (225, 225),
                        (50, 225)])

    dst_wall = np.float32([(225, 0),
                        (0, 0),
                        (225, 225),
                        (0, 225)])

    M_wall = cv2.getPerspectiveTransform(src_wall, dst_wall)
    M_gnd = cv2.getPerspectiveTransform(src_gnd, dst_gnd)
    return M_wall, M_gnd

def is_blank_frame(f):
    return f.sum() < 10

def create_arena():
    _arena = []
    arena_size = 40
    # create 40x40 array
    for j in range(arena_size):
        row = []
        for k in range(arena_size):
            row.append(1)
        _arena.append(row)
    return _arena

def get_obj_hog(obj):
    # obj is an image
    flat_hog = hog(obj, feature_vector=True, orientations=8, multichannel=True)
    #print("flat hog shape:", flat_hog.shape)
    flat_hog = np.array(flat_hog)
    # there are 72 bin features/freq's for each subsection of the image 
    avg_hog_block = np.mean(flat_hog.reshape(-1, 72), axis=0)
    return avg_hog_block

def dist_agent_obj(a_pos, obj):
    a_x, a_y = a_pos
    o_x, o_y = get_obj_pos(obj)
    euclid_dist = ((o_x - a_x)**2 + (o_y - a_y)**2)**0.5
    return euclid_dist

# @TODO return a confidence and integrate with existing conf calculations
def get_pref_obj_pos(history, o_1, o_2, a_pos):    
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
    # print("short_acc:", short_acc)
    # print("far_acc:", far_acc)
    # print("color_acc:", color_acc)
    # print("hog_acc:", hog_acc)
    best_acc = max(hypotheses)
    # tied preference hypotheses affects our confidence
    tied_count = hypotheses.count(best_acc)
    # will be altered later if theirs a classifier tie
    pref_confidence = best_acc
    best_classifier = hypotheses.index(best_acc)
    
    if tied_count == 1:
        if best_classifier == 0 or best_classifier == 1:
            #print("preferred object is based on distance")
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
            #print("preferred object is a certain color")
            dist_1 = np.absolute(get_ch_avgs(o_1["rgb"]) - avg_chosen_c).sum()
            dist_2 = np.absolute(get_ch_avgs(o_2["rgb"]) - avg_chosen_c).sum()
            if dist_1 < dist_2:
                o_x, o_y = get_obj_pos(o_1["rgb"])
            else:
                o_x, o_y = get_obj_pos(o_2["rgb"])
        else:
            #print("preferred object is a certain shape")
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
            dist_1 = np.absolute(get_ch_avgs(o_1["rgb"]) - avg_chosen_c).sum()
            dist_2 = np.absolute(get_ch_avgs(o_2["rgb"]) - avg_chosen_c).sum()
            if dist_1 < dist_2:
                if type(h_1_chosen) == type(None):
                    h_1_chosen = 1  
                elif type(h_2_chosen) == type(None):
                    h_2_chosen = 1
                else:
                    h_3_chosen = 1
            else:
                if type(h_1_chosen) == type(None):
                    h_1_chosen = 2  
                elif type(h_2_chosen) == type(None):
                    h_2_chosen = 2
                else:
                    h_3_chosen = 2
        if hypotheses[3] == best_acc:
            dist_1 = np.absolute(get_obj_hog(o_1["rgb"]) - avg_chosen_h).sum()
            dist_2 = np.absolute(get_obj_hog(o_2["rgb"]) - avg_chosen_h).sum()
            if dist_1 < dist_2:
                if type(h_2_chosen) == type(None):
                    h_2_chosen = 1  
                else:
                    h_3_chosen = 1
            else:
                if type(h_2_chosen) == type(None):
                    h_2_chosen = 2
                else:
                    h_3_chosen = 2

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
                # just pick one i.e. guessing at this point
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

    return o_x, o_y, pref_confidence

# instead of a recursive solution, just hardcoded one since the nested structure isn't too deep
def jsonify_info_dict(info):    
    keys_to_keep = ["pref_dict", "trial_err", "step_num", "arena", "agent_ch_avgs", "objs_ch_avgs", "trial_1_objs_pos", "path", "wall_i_s", "trial_err"]
    json_friendly = {key:val for key, val in info.items() if key in keys_to_keep}
    for key, value in json_friendly.items():
        if type(value) == type(np.array([])):
            json_friendly[key] = value.tolist()
        elif type(value) == type({}):
            tmp_dict = {k:v for k, v in value.items()}
            for k, v in tmp_dict.items():
                if type(v) == type(np.array([])):
                    tmp_dict[k] = v.tolist()
            json_friendly[key] = tmp_dict
        elif type(value) == type((1, 2)):
            tmp_tuple = ()
            for v in value:
                if type(v) == type(np.array([])):
                    tmp_tuple += (v.tolist(), )
                else:
                    tmp_tuple += (v, )
            json_friendly[key] = tmp_tuple
    
    return json_friendly

def find_and_rm_jerk_walls(g_rgb, g_mask, structural_mask_c_s, a, o_1, o_2, h):
    '''
    Assumes that the agent's mask is the same color as a wall piece.
    Those wall pieces are called "jerk walls".
    This was developed with the oracle masks.
    The robustness to level2 data could be tested further.
    '''    

    x_ = h["center"][0]
    y_ = h["center"][1]
    agent_color = g_rgb[y_, x_, ::-1]
    agent_mask_color = g_mask[y_, x_, ::-1]
    # need at least 1 agent and 1 object
    if a["c"] != tuple(agent_mask_color):
        #print("possible wall with same mask color as agent!")
        # mask using inRange
        m = cv2.inRange(g_rgb, np.array(agent_color-10), np.array(agent_color+10))
        _obj_im, _ = apply_color_mask(np.copy(g_rgb), np.copy(g_mask), agent_mask_color, show=False)
        g_m = np.copy(g_mask)
        both_mask, _ = apply_color_mask(np.copy(g_mask), np.copy(g_mask), agent_color, show=False)
        both_mask[m == 0] = 255
        agent_mask = np.array(g_m, dtype=np.uint8)
        both_mask[agent_mask != 0] = 0
        _m = np.array(both_mask, dtype=np.uint8)
        g_mask[_m != 0] = 255
        #cv2_show_im(_m, g_mask)
        c = (255, 255, 255)
        structural_mask_c_s.append(c)
        if type(o_1["c"]) != type(None):
            # obj_1 and obj_2 were put into obj_1 and agent...
            o_2 = deepcopy(a)
        else:
            o_1 = deepcopy(a)    
        a["rgb"] = np.copy(_obj_im)
        a["mask"] = np.copy(agent_mask)
        a["c"] = tuple(np.copy(agent_mask_color))
        return _m, g_mask, structural_mask_c_s, a, o_1, o_2
    else:
        return None, g_mask, structural_mask_c_s, a, o_1, o_2