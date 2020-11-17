from MCS_exploration.navigation.visibility_road_map import ObstaclePolygon
from shapely.geometry import MultiPolygon
from descartes import PolygonPatch
import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy
import json
import matplotlib
matplotlib.use('TkAgg')

AGENT_RADIUS = 0.35

TROPHY_OPTION = {
    "single_trophy_1": lambda o: (o[12], None),
    "single_trophy_2": lambda o: (o[14], None),
}

VALID_OPTIONS = {
    "sturdy_med": lambda o: (o[9], o[1]),
    "sturdy_large": lambda o: (o[10], o[2]),
    "suitcase_med": lambda o: (o[13], o[4]),
    "suitcase_large": lambda o: (o[14], o[5]),
    "treasure_chest_med": lambda o:(o[11], o[7]),
    "treasure_chest_large": lambda o:(o[12], o[8])
} # only 12 is a 'standing' trophy

INVALID_OPTIONS = {
    "suitcase_small": lambda o: (None, o[3]),
    "sturdy_small": lambda o: (None, o[0]),
    "treasure_chest_small": lambda o: (None, o[6]),
}

def set_goal_with_trophy(scene_config, box_config, trophy_prob=1):
    box_config = deepcopy(box_config)
    obstacles = []


    remove_walls = []
    for obj in scene_config['objects']:
        if "wall" not in obj['id']:
            remove_walls.append(obj)
    scene_config['objects'] = remove_walls

    for obj in scene_config['objects']:
        x_list = []
        y_list = []
        for i in range(4):
            x_list.append(obj['shows'][0]['bounding_box'][i]['x'])
            y_list.append(obj['shows'][0]['bounding_box'][i]['z'])
        obstacles.append(ObstaclePolygon(x_list, y_list))

    x_list = np.array([5.5, 5.5, -5.5, -5.5])
    y_list = np.array([5.5, 5, 5, 5.5])
    for i in [1, -1]:
        obstacles.append(ObstaclePolygon(x_list*i, y_list*i))
        obstacles.append(ObstaclePolygon(y_list*i, x_list*i))

    x, z = scene_config['performerStart']['position']['x'], scene_config['performerStart']['position']['z']
    x_list = [x+AGENT_RADIUS, x+AGENT_RADIUS, x-AGENT_RADIUS, x-AGENT_RADIUS]
    y_list = [z+AGENT_RADIUS, z-AGENT_RADIUS, z-AGENT_RADIUS, z+AGENT_RADIUS]
    obstacles.append(ObstaclePolygon(x_list, y_list))

    objs = pre_process_objects(box_config['objects'], tuple(obstacles), trophy_prob=trophy_prob)

    new_scene_config = scene_config.copy()
    for o in objs:
        new_scene_config['objects'].extend(o.get_objects())
    new_scene_config['goal'] = box_config['goal']

    return new_scene_config



def pre_process_objects(objects, all_obstacles, trophy_prob=1):

    if random.random() > 1 - trophy_prob: # turn this to 0.x to have 0.x probability to see a box contains a trophy
        # args = TROPHY_OPTION['single_trophy'](objects)
        valid_keys = random.choice(list(TROPHY_OPTION.keys()))
        args = TROPHY_OPTION[valid_keys](objects)
    else:
        valid_keys = random.choice(list(VALID_OPTIONS.keys()))
        args = VALID_OPTIONS[valid_keys](objects)

    random_pick = TrophyWithBox(*args)

    all_obstacles = random_pick.place_to_scene(all_obstacles)

    invalid_options = random.choice(list(INVALID_OPTIONS.keys()))
    args = INVALID_OPTIONS[invalid_options](objects)
    empty_box = TrophyWithBox(*args)
    all_obstacles = empty_box.place_to_scene(all_obstacles)

    invalid_options_2 = random.choice(list(VALID_OPTIONS.keys()))
    args_2 = VALID_OPTIONS[invalid_options_2](objects)

    empty_box_2 = TrophyWithBox(None, args_2[1])
    _ = empty_box.place_to_scene(all_obstacles)

    return random_pick, empty_box, empty_box_2 # there will always be 2 empty boxes in the scene, and one trophy or trophy with box


class TrophyWithBox:

    BOX_TROPHY_SIZE = json.load(open("interaction_scenes/box_trophy_size.json", "r"))
    RANDOM_ROTATE = [0, 90, 180, 270]

    def __init__(self, trophy, box, random_rotate=True):
        self.trophy = trophy
        self.box = box
        self.bdbox = None
        self.rotate_in_y = None
        if random_rotate:
            self.rotate_in_y = random.choice(self.RANDOM_ROTATE)
            self.rotate_in_y = 0
            # print("Rotate {}".format(self.rotate_in_y))
            for obj in [self.box, self.trophy]:
                if obj is not None:
                    self.random_rotate_in_y(obj['shows'][0], self.rotate_in_y)
        # if self.box and self.trophy:
        #     if "opened" not in self.box:
        #         self.box['opened'] = True

    def random_rotate_in_y(self, first_show, rotation):
        if 'rotation' not in first_show:
            first_show['rotation'] = {'x': 0, 'y':0, 'z':0}
        first_show['rotation']['y'] = (first_show['rotation']['y'] + rotation) % 360

    def get_bonding_box_radius(self):
        if self.box:
            r_x, r_z = self.BOX_TROPHY_SIZE[self.box['id']]['x_r'], self.BOX_TROPHY_SIZE[self.box['id']]['x_r']
        else:
            r_x, r_z = self.BOX_TROPHY_SIZE[self.trophy['id']]['x_r'], self.BOX_TROPHY_SIZE[self.trophy['id']]['z_r']

        if self.rotate_in_y:
            if self.rotate_in_y in [90, 270]:
                tmp = r_x
                r_x = r_z
                r_z = tmp

        return r_x, r_z

    def get_objects(self):
        ret = []
        if self.box:
            ret += [self.box]
        if self.trophy:
            ret += [self.trophy]
        return ret

    def get_location(self, obj):
        return obj['shows'][0]['position']['x'], obj['shows'][0]['position']['z']

    def set_location(self, obj, x, z):
        obj['shows'][0]['position']['x'] = x
        obj['shows'][0]['position']['z'] = z

    def place_to_scene(self, all_obstacles, plot=False):
        r_x, r_z = self.get_bonding_box_radius()
        new_all_obstacles = None
        while True:
            x, z = random.random() * 10 - 5, random.random() * 10 - 5
            x_list = [x + r_x, x + r_x, x - r_x, x - r_x]
            y_list = [z + r_z, z - r_z, z - r_z, z + r_z]
            bdbox = ObstaclePolygon(x_list, y_list)

            poly = MultiPolygon(all_obstacles)
            if plot:
                plt.cla()
                plt.xlim((-7, 7))
                plt.ylim((-7, 7))
                plt.gca().set_xlim((-7, 7))
                plt.gca().set_ylim((-7, 7))

                patch1 = PolygonPatch(poly, fc="green", ec="black", alpha=0.2, zorder=1)
                plt.gca().add_patch(patch1)
                bdbox.plot('blue')
                plt.pause(0.01)

            try:
                if not poly.intersection(bdbox):
                    center_x, center_z = x, z
                    self.bdbox = bdbox
                    new_all_obstacles = [i for i in all_obstacles]
                    new_all_obstacles.append(bdbox)
                    break
            except:
                pass

        box_x, box_z, trophy_x, trophy_z = None, None, None, None
        if self.trophy:
            trophy_x, trophy_z = self.get_location(self.trophy)

        if self.box:
            box_x, box_z = self.get_location(self.box)

        if self.box and self.trophy:
            offset_x = trophy_x - box_x
            offset_z = trophy_z - box_z
            self.set_location(self.box, center_x, center_z)
            self.set_location(self.trophy, center_x + offset_x, center_z + offset_z)
        elif self.trophy:
            self.set_location(self.trophy, center_x, center_z)
        elif self.box:
            self.set_location(self.box, center_x, center_z)

        if self.trophy:
            self.trophy['id'] = "trophy"

        assert new_all_obstacles
        assert len(new_all_obstacles) == len(all_obstacles) + 1
        return tuple(new_all_obstacles)