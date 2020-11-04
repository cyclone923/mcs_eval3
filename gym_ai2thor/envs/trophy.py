from MCS_exploration.navigation.visibility_road_map import ObstaclePolygon
from shapely.geometry import MultiPolygon
from descartes import PolygonPatch
import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy

def set_goal_with_trophy(scene_config, box_config, plot=True):
    box_config = deepcopy(box_config)
    obstacles = []

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

    agent_radious = 0.22
    x, z = scene_config['performerStart']['position']['x'], scene_config['performerStart']['position']['z']
    x_list = [x+agent_radious, x+agent_radious, x-agent_radious, x-agent_radious]
    y_list = [z+agent_radious, z-agent_radious, z-agent_radious, z+agent_radious]
    obstacles.append(ObstaclePolygon(x_list, y_list))
    all_obstacles = MultiPolygon(obstacles)

    random_target = pre_process_objects(box_config['objects'], all_obstacles)

    if plot:
        plt.cla()
        plt.xlim((-7, 7))
        plt.ylim((-7, 7))
        plt.gca().set_xlim((-7, 7))
        plt.gca().set_ylim((-7, 7))

        patch1 = PolygonPatch(all_obstacles, fc="green", ec="black", alpha=0.2, zorder=1)
        plt.gca().add_patch(patch1)
        random_target.bdbox.plot('blue')
        plt.pause(0.1)


    new_scene_config = scene_config.copy()
    new_scene_config['objects'].extend(random_target.get_objects().copy())
    new_scene_config['goal'] = box_config['goal'].copy()

    return new_scene_config


def pre_process_objects(objects, all_obstacles):
    object_ids = ['gift_box', 'sturdy_box', 'suitcase', 'trophy_1', 'trophy_2', 'trophy_3', 'trophy_4']
    for i, x in enumerate(objects):
        assert x['id'] == object_ids[i]

    single_trophy = TrophyWithBox(objects[6], None)
    box1 = TrophyWithBox(objects[3] ,objects[0])
    box2 = TrophyWithBox(objects[4], objects[1])
    box3 = TrophyWithBox(objects[5], objects[2])

    all_objs = [single_trophy, box1, box2, box3]
    # random_pick = random.choice(all_objs)
    random_pick = all_objs[2]
    trophy_radious = random_pick.get_bonding_box_radius()
    while True:
        x, z = random.random() * 10 - 5, random.random() * 10 - 5
        x_list = [x+trophy_radious, x+trophy_radious, x-trophy_radious, x-trophy_radious]
        y_list = [z+trophy_radious, z-trophy_radious, z-trophy_radious, z+trophy_radious]
        bdbox = ObstaclePolygon(x_list, y_list)
        if not all_obstacles.intersection(bdbox):
            trophy_box_x, trophy_box_z = x, z
            random_pick.bdbox = bdbox
            break

    trophy_x, trophy_z = TrophyWithBox.get_location(random_pick.trophy)

    if random_pick.box:
        box_x, box_z = TrophyWithBox.get_location(random_pick.box)
        offset_x = trophy_x - box_x
        offset_z = trophy_z - box_z
        TrophyWithBox.set_location(random_pick.box, trophy_box_x, trophy_box_z)
        TrophyWithBox.set_location(random_pick.trophy, trophy_box_x + offset_x, trophy_box_z + offset_z)
    else:
        TrophyWithBox.set_location(random_pick.trophy, trophy_box_x, trophy_box_z)

    random_pick.trophy['id'] = "trophy"

    return random_pick


class TrophyWithBox:

    def __init__(self, trophy, box):
        self.trophy = trophy
        self.box = box
        self.bdbox = None

    def get_bonding_box_radius(self):
        if self.box:
            r = 1
        else:
            r = 0.22
        return r

    def get_objects(self):
        ret = [self.trophy]
        if self.box:
            ret += [self.box]
        return ret


    @staticmethod
    def get_location(obj):
        return obj['shows'][0]['position']['x'], obj['shows'][0]['position']['x']


    @staticmethod
    def set_location(obj, x, z):
        obj['shows'][0]['position']['x'] = x
        obj['shows'][0]['position']['x'] = z



