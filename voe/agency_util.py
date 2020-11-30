import os
import cv2
import glob
import math
import random
import numpy as np
from copy import deepcopy
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

# instead of a recursive solution, just hardcoded one since the nested structure isn't too deep
def jsonify_info_dict(info):    
    keys_to_keep = ["trial_err", "step_num", "arena", "agent_ch_avgs", "objs_ch_avgs", "trial_1_objs_pos", "path", "wall_i_s", "trial_err"]
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