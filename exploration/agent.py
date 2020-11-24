from exploration.navigator import Navigator
from exploration.obstacle_buffer import ObstacleBuffer
from exploration.strategy import ExploreStrategy
from exploration.trophy_goal import TrophyGoal
from exploration.plot import GeometryPlot
from exploration.dead_reckoning import DeadReckoningMcsWrapper

SHOW_ANIMATION = True


class ExploreAgent2D(DeadReckoningMcsWrapper, GeometryPlot):

    def __init__(self, env, level='oracle'):
        DeadReckoningMcsWrapper.__init__(self, env, level)
        GeometryPlot.__init__(self)
        self.obstacle_buffer = ObstacleBuffer(level)
        self.navigator = Navigator(self, robot_radius=0.27, max_step=0.1, success_distance=0.3)
        self.explore_strategy = ExploreStrategy(self)
        self.goal = TrophyGoal()
        self.level = level

        if self.level == 'oracle':
            self.add_obstacle_func = self.obstacle_buffer.add_obstacle_oracle
            self.update_trophy_goal = self.goal.update_trophy_goal_oracle
        elif self.level == 'level1':
            self.add_obstacle_func = self.obstacle_buffer.add_obstacle_level1
            self.update_trophy_goal = self.goal.update_trophy_goal_oracle


    # Make sure this is the only 'step' used, because obstacles are added here
    def step(self, **kwargs):
        DeadReckoningMcsWrapper.step(self, **kwargs)
        self.add_obstacle_func(self)
        self.explore_strategy.update_state()

        if SHOW_ANIMATION:
            self.show_plot(self)

        self.update_trophy_goal(self.step_output)


    def reset(self):
        DeadReckoningMcsWrapper.reset(self)
        self.obstacle_buffer.reset()
        self.explore_strategy.reset()
        self.goal.reset()

    @property
    def goal_found(self):
        return self.goal.goal_found

    #used in the navigator
    @property
    def scene_obstacles(self):
        return self.obstacle_buffer.scene_obstacles_dict

    @property
    def agent_radius(self):
        return self.navigator.radius

    def check_goal_found(self):
        if self.level == 'oracle':
            return


    def pick_trophy(self):
        self.explore_strategy.initial_exploration()

        while not self.goal_found:
            target_pose = self.explore_strategy.find_next_best_pose()
            self.navigator.go_to_goal(target_pose)

        self.navigator.go_to_goal(self.goal.position)








