import numpy as np
import time

from game_state import GameState
from navigation import bounding_box_navigator
from MCS_exploration.gym_ai2thor.envs.trophy import AGENT_RADIUS

import constants


class GraphAgent(object):
    def __init__(self, env,level, reuse=True, num_unrolls=1, game_state=None, net_scope=None):
        self.nav_radius = AGENT_RADIUS
        self.nav = bounding_box_navigator.BoundingBoxNavigator(self.nav_radius, maxStep=0.1)
        if game_state is None:
            print("Gamee State NONEEE MS")
            self.game_state = GameState(env=env,level=level)
            self.game_state.add_obstacle_func = self.nav.add_obstacle_from_bounding_boxes
            #self.game_state.add_obstacle_func = self.nav.add_obstacle_from_global_obstacles
            self.game_state.get_obstacles = self.nav.get_obstacles
        else:
            self.game_state = game_state
        #self.action_util = self.game_state.action_util
        self.gt_graph = None
        #self.sess = sess
        self.num_steps = 0
        self.global_step_id = 0
        self.num_unrolls = num_unrolls
        self.pose_indicator = np.zeros((constants.TERMINAL_CHECK_PADDING * 2 + 1,
                                        constants.TERMINAL_CHECK_PADDING * 2 + 1))
        self.times = np.zeros(2)

    def reset(self, scene_name=None, seed=None, config_filename="",event=None):
        if scene_name is not None:
            self.nav.reset()
            if self.game_state.env is not None and type(self.game_state) == GameState:
                self.game_state.reset(scene_name, use_gt=False, seed=seed,config_filename=config_filename,event=event)
            self.bounds = None
            #self.action = np.zeros(self.action_util.num_actions)
            self.memory = np.zeros((constants.SPATIAL_MAP_HEIGHT, constants.SPATIAL_MAP_WIDTH, constants.MEMORY_SIZE))
            self.gru_state = np.zeros((1, constants.GRU_SIZE))
            self.pose = self.game_state.pose
            self.is_possible = 1
            self.num_steps = 0
            self.times = np.zeros(2)
            self.impossible_spots = set()
            self.visited_spots = set()
        else:
            #print ("")
            self.game_state.reset(event=event)


    def step(self, action):
        print("MS other step used! ", action)
        t_start = time.time()
        self.game_state.step(action)
        self.times[1] += time.time() - t_start
        self.num_steps += 1
        return


    def get_plan(self):
        self.plan, self.path = self.game_state.graph.get_shortest_path(self.pose, self.game_state.end_point)
        return self.plan, self.path

    def get_label(self):
        #patch, curr_point = self.gt_graph.get_graph_patch(self.pose)
        patch = self.gt_graph.get_graph_patch(self.pose)
        patch = patch[:, :, 0]
        patch[patch < 2] = 0
        patch[patch > 1] = 1
        return patch
